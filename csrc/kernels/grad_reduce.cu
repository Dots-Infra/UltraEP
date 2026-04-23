#include <vector>

#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

// Vectorized memset for remote memory zeroing (supports up to GRAD_REDUCE_TILE_SIZE_BYTES)
__device__ __forceinline__ void memset_zero_tile(float* __restrict__ ptr, int num_elements) {
    // Use vectorized stores for efficiency (16 bytes = 4 floats per store)
    int num_float4s = num_elements / 4;
    float4* vec_ptr = reinterpret_cast<float4*>(ptr);
#pragma unroll 4
    for (int i = threadIdx.x; i < num_float4s; i += blockDim.x) {
        // Use streaming store to bypass L2 cache (data won't be reused)
        ptx::st_global_v4_u32_streaming(&vec_ptr[i], 0, 0, 0, 0);
    }

    // Handle remaining elements (0-3 floats)
    int remaining_start = num_float4s * 4;
    int remaining = num_elements - remaining_start;
    if (threadIdx.x < remaining) {
        ptr[remaining_start + threadIdx.x] = 0.0f;
    }
}

// ============================================================================
// Grad Reduce Kernel with Tile-Level Persistent Parallelism
// ============================================================================
//
// Design goals:
// 1. Keep the reduction path simple: one kernel, one scheduling strategy.
// 2. Use tile-level work stealing so CTA count can be tuned independently of task count.
// 3. Preserve overlap control by mapping an explicit SM budget to the launch grid.
// ============================================================================

// Structure describing a tile within a task (computed at runtime)
struct TileInfo {
    int task_idx;          // Which task this tile belongs to
    int tile_idx_in_task;  // Tile index within the task
    size_t global_offset;  // Offset in elements from task start
    int num_elements;      // Number of elements in this tile (may be < GRAD_REDUCE_TILE_ELEMENTS for last tile)
};

// Map a global tile index to task and tile within task
__device__ __forceinline__ TileInfo get_tile_info(const GradReduceTask* tasks,
                                                  const int* task_tile_offsets,
                                                  int num_tasks,
                                                  int global_tile_idx) {
    TileInfo info;

    // Binary search to find which task this tile belongs to
    int lo = 0, hi = num_tasks - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (task_tile_offsets[mid] <= global_tile_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    info.task_idx = lo;
    info.tile_idx_in_task = global_tile_idx - task_tile_offsets[lo];
    info.global_offset = (size_t)info.tile_idx_in_task * GRAD_REDUCE_TILE_ELEMENTS;

    // Compute number of elements in this tile
    size_t task_numel = tasks[info.task_idx].numel;
    size_t remaining = task_numel - info.global_offset;
    info.num_elements = min((size_t)GRAD_REDUCE_TILE_ELEMENTS, remaining);

    return info;
}

// task_metadata: device pointer to [total_tasks, total_tiles]
__global__ void grad_reduce_kernel(const GradReduceTask* grad_reduce_tasks,
                                   const int* task_tile_offsets,
                                   const int* task_metadata,
                                   int* global_tile_counter) {
    extern __shared__ float smem_pool[];

    ptx::mbarrier* mbarrier_ptr = ptx::create_mbarrier();
    __shared__ ptx::arrival_phase phase;

    // Read task metadata from device memory
    __shared__ int total_tasks;
    __shared__ int total_tiles;

    // Initialize mbarrier (thread 0 only)
    if (threadIdx.x == 0) {
        total_tasks = task_metadata[0];
        total_tiles = task_metadata[1];
        ptx::mbarrier_init(mbarrier_ptr, 1);  // Expect 1 arrival (from TMA completion)
        phase = 0;
    }
    __syncthreads();

    // Early exit if no work
    if (total_tasks == 0) {
        if (threadIdx.x == 0) {
            ptx::mbarrier_invalidate(mbarrier_ptr);
        }
        return;
    }

    // Persistent loop - each CTA grabs tiles until work is exhausted
    while (true) {
        // Fetch next tile index (leader thread only)
        __shared__ int my_tile_idx;
        if (threadIdx.x == 0) {
            my_tile_idx = atomicAdd(global_tile_counter, 1);
        }
        __syncthreads();

        if (my_tile_idx >= total_tiles)
            break;

        // Get tile information via binary search
        TileInfo tile = get_tile_info(grad_reduce_tasks, task_tile_offsets, total_tasks, my_tile_idx);
        GradReduceTask task = grad_reduce_tasks[tile.task_idx];

        // Calculate TMA transfer size (must be 16-byte aligned)
        size_t bytes = tile.num_elements * sizeof(float);
        bytes = (bytes + 15) & ~15;

        // Issue TMA load for this tile (thread 0 only)
        if (threadIdx.x == 0) {
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, bytes);
            ptx::tma_load_1d(smem_pool,
                             task.replica_remote_addr + tile.global_offset,
                             mbarrier_ptr,
                             bytes,
                             ptx::TMACacheHint::kEvictFirst);
        }

        // Wait for TMA completion (thread 0 waits, then sync all)
        if (threadIdx.x == 0) {
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        }
        __syncthreads();

        // Perform reduction: add loaded values to master buffer
        // Using atomicAdd to handle potential concurrent updates from multiple replicas
        for (int i = threadIdx.x; i < tile.num_elements; i += blockDim.x) {
            atomicAdd(&task.master_local_addr[tile.global_offset + i], smem_pool[i]);
        }
        __syncthreads();

        // Zero out the replica buffer region we just consumed
        memset_zero_tile(task.replica_remote_addr + tile.global_offset, tile.num_elements);

        // Memory fence to ensure remote stores are visible to other GPUs
        __threadfence_system();
    }

    // Invalidate mbarrier before exit
    __syncthreads();
    if (threadIdx.x == 0) {
        ptx::mbarrier_invalidate(mbarrier_ptr);
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

static void launch_grad_reduce(GradReduceTask* grad_reduce_tasks_gpu,
                               int* task_tile_offsets_gpu,
                               int* task_metadata_gpu,
                               int* global_tile_counter_gpu,
                               int num_ctas,
                               cudaStream_t stream) {
    dim3 grid(num_ctas);
    dim3 block(GRAD_REDUCE_THREADS_PER_BLOCK);
    int smem_size = GRAD_REDUCE_TILE_SIZE_BYTES;

    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(grad_reduce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    grad_reduce_kernel<<<grid, block, smem_size, stream>>>(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu);
}

// ============================================================================
// CPU task-build path
// ============================================================================

void run_grad_reduce(const GradReduceTask* grad_reduce_tasks_cpu,
                     GradReduceTask* grad_reduce_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     int* task_metadata_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     int num_sms) {
    if (total_tasks == 0)
        return;

    // Compute tile offsets on CPU (prefix sum of tile counts per task)
    std::vector<int> task_tile_offsets(total_tasks + 1);
    task_tile_offsets[0] = 0;
    for (int i = 0; i < total_tasks; ++i) {
        int num_tiles = (grad_reduce_tasks_cpu[i].numel + GRAD_REDUCE_TILE_ELEMENTS - 1) / GRAD_REDUCE_TILE_ELEMENTS;
        task_tile_offsets[i + 1] = task_tile_offsets[i] + num_tiles;
    }
    int total_tiles = task_tile_offsets[total_tasks];

    // Write task metadata
    int metadata[2] = {total_tasks, total_tiles};
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(task_metadata_gpu, metadata, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Copy tasks and tile offsets from CPU to GPU
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(grad_reduce_tasks_gpu,
                                       grad_reduce_tasks_cpu,
                                       total_tasks * sizeof(GradReduceTask),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(task_tile_offsets_gpu,
                                       task_tile_offsets.data(),
                                       (total_tasks + 1) * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_tile_counter_gpu, 0, sizeof(int), stream));

    launch_grad_reduce(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu, num_sms, stream);
}

// ============================================================================
// GPU task-build path (tasks already on GPU from build kernel)
// ============================================================================

void run_grad_reduce_from_gpu(GradReduceTask* grad_reduce_tasks_gpu,
                              int* task_tile_offsets_gpu,
                              int* task_metadata_gpu,
                              int* global_tile_counter_gpu,
                              cudaStream_t stream,
                              int num_sms) {
    launch_grad_reduce(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu, num_sms, stream);
}

}  // namespace ultra_ep::kernels
