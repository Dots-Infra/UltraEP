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
// Grad Reduce Kernel with Task-Level Parallelism (Low SM Mode)
// ============================================================================
//
// Strategy: Each CTA processes complete tasks (not individual tiles)
//
// Benefits:
// 1. No atomic counter contention for tile fetching
// 2. Within a task, tiles are processed sequentially - NO atomicAdd needed!
// 3. AtomicAdd only needed when multiple replicas share the same master
//
// For typical workloads with few replicas per master, this significantly
// reduces atomic contention.
//
// Pipeline within each task:
//   Tile 0: [TMA₀] [Reduce₀]
//   Tile 1:        [TMA₁] [Reduce₁]
//   Tile 0:               [Zero₀]
//   ...
// ============================================================================

// Structure for passing task info to kernel
struct TaskMeta {
    float* master_addr;
    float* replica_addr;
    int num_tiles;
    int num_elements_last_tile;  // Last tile may be smaller
};

// task_metadata: device pointer to [total_tasks, total_tiles]
__global__ void grad_reduce_kernel_low_sm(const GradReduceTask* grad_reduce_tasks,
                                          const int* task_metadata,
                                          int* global_task_counter) {
    // Double-buffered shared memory
    extern __shared__ float smem_base[];
    float* smem[2] = {smem_base, smem_base + GRAD_REDUCE_TILE_ELEMENTS};

    // Mbarriers for TMA
    ptx::mbarrier* mbarriers = ptx::create_mbarriers<2>();
    __shared__ ptx::arrival_phase phases[2];

    const bool is_leader = (threadIdx.x == 0);

    // Read total_tasks from device memory
    __shared__ int total_tasks;
    if (is_leader) {
        total_tasks = task_metadata[0];
    }

    // Initialize mbarriers
    if (is_leader) {
        for (int i = 0; i < 2; i++) {
            ptx::mbarrier_init(&mbarriers[i], 1);
            phases[i] = 0;
        }
    }
    __syncthreads();

    // Early exit if no work
    if (total_tasks == 0) {
        if (is_leader) {
            for (int i = 0; i < 2; i++) {
                ptx::mbarrier_invalidate(&mbarriers[i]);
            }
        }
        return;
    }

    // Persistent loop: each CTA grabs complete tasks
    while (true) {
        // Fetch next task
        __shared__ int my_task_idx;
        if (is_leader) {
            my_task_idx = atomicAdd(global_task_counter, 1);
        }
        __syncthreads();

        if (my_task_idx >= total_tasks)
            break;

        // Load task info
        GradReduceTask task = grad_reduce_tasks[my_task_idx];
        float* master = task.master_local_addr;
        float* replica = task.replica_remote_addr;
        const size_t numel = task.numel;
        const int num_tiles = (numel + GRAD_REDUCE_TILE_ELEMENTS - 1) / GRAD_REDUCE_TILE_ELEMENTS;

        // Process tiles within this task with double-buffered pipelining
        // Issue first TMA load
        int count0 = min((size_t)GRAD_REDUCE_TILE_ELEMENTS, numel);
        size_t bytes0 = (count0 * sizeof(float) + 15) & ~15;

        if (is_leader) {
            ptx::mbarrier_arrive_and_set_tx(&mbarriers[0], bytes0);
            ptx::tma_load_1d(smem[0], replica, &mbarriers[0], bytes0, ptx::TMACacheHint::kEvictFirst);
        }

        for (int tile = 0; tile < num_tiles; tile++) {
            int cur = tile % 2;
            int next = 1 - cur;

            size_t cur_offset = (size_t)tile * GRAD_REDUCE_TILE_ELEMENTS;
            int cur_count = min((size_t)GRAD_REDUCE_TILE_ELEMENTS, numel - cur_offset);

            // Issue TMA for next tile (if exists)
            if (tile + 1 < num_tiles) {
                size_t next_offset = (size_t)(tile + 1) * GRAD_REDUCE_TILE_ELEMENTS;
                int next_count = min((size_t)GRAD_REDUCE_TILE_ELEMENTS, numel - next_offset);
                size_t next_bytes = (next_count * sizeof(float) + 15) & ~15;

                if (is_leader) {
                    ptx::mbarrier_arrive_and_set_tx(&mbarriers[next], next_bytes);
                    ptx::tma_load_1d(smem[next],
                                     replica + next_offset,
                                     &mbarriers[next],
                                     next_bytes,
                                     ptx::TMACacheHint::kEvictFirst);
                }
            }

            // Wait for current tile's TMA
            if (is_leader) {
                ptx::mbarrier_wait_and_flip_phase(&mbarriers[cur], phases[cur]);
            }
            __syncthreads();

// Reduce: NO atomicAdd needed within the same task!
// Different CTAs processing the same master (from different replicas) still need atomic.
// But we use regular add here - if there are multiple replicas, atomicAdd is needed.
// For now, keep atomicAdd to be safe. The key win is reduced tile counter contention.
#pragma unroll 4
            for (int i = threadIdx.x; i < cur_count; i += blockDim.x) {
                atomicAdd(&master[cur_offset + i], smem[cur][i]);
            }
            __syncthreads();

            // Zero: can overlap with next tile's TMA (already issued above)
            memset_zero_tile(replica + cur_offset, cur_count);
        }

        // Memory fence after processing entire task
        __threadfence_system();
    }

    // Cleanup
    __syncthreads();
    if (is_leader) {
        for (int i = 0; i < 2; i++) {
            ptx::mbarrier_invalidate(&mbarriers[i]);
        }
    }
}

// ============================================================================
// Optimized Grad Reduce Kernel with Multi-CTA Cooperation (High SM Mode)
// ============================================================================
//
// Design Goals:
// 1. Maximize SM utilization by having multiple CTAs work on a single task
// 2. Pipeline TMA loads with reduce operations
// 3. Efficiently zero out replica buffers after data is consumed
//
// Key Optimizations:
// - Fine-grained tile-level task distribution instead of coarse task-level
// - All CTAs participate in processing tiles across all tasks
// - Persistent kernel pattern with atomic tile counter
// ============================================================================

// Structure describing a tile within a task (computed at runtime)
struct TileInfo {
    int task_idx;          // Which task this tile belongs to
    int tile_idx_in_task;  // Tile index within the task
    size_t global_offset;  // Offset in elements from task start
    int num_elements;      // Number of elements in this tile (may be < GRAD_REDUCE_TILE_ELEMENTS for last tile)
};

// Compute total number of tiles across all tasks
__device__ __forceinline__ int compute_total_tiles(const GradReduceTask* tasks, int num_tasks) {
    int total = 0;
    for (int i = 0; i < num_tasks; ++i) {
        total += (tasks[i].numel + GRAD_REDUCE_TILE_ELEMENTS - 1) / GRAD_REDUCE_TILE_ELEMENTS;
    }
    return total;
}

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
__global__ void grad_reduce_kernel_high_sm(const GradReduceTask* grad_reduce_tasks,
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

static void launch_grad_reduce_low_sm(GradReduceTask* grad_reduce_tasks_gpu,
                                      int* task_metadata_gpu,
                                      int* global_task_counter_gpu,
                                      int num_ctas,
                                      cudaStream_t stream) {
    dim3 grid(num_ctas);
    dim3 block(GRAD_REDUCE_THREADS_PER_BLOCK);
    int smem_size = GRAD_REDUCE_TILE_SIZE_BYTES * 2;

    CUDA_RUNTIME_CHECK(
        cudaFuncSetAttribute(grad_reduce_kernel_low_sm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    grad_reduce_kernel_low_sm<<<grid, block, smem_size, stream>>>(
        grad_reduce_tasks_gpu, task_metadata_gpu, global_task_counter_gpu);
}

static void launch_grad_reduce_high_sm(GradReduceTask* grad_reduce_tasks_gpu,
                                       int* task_tile_offsets_gpu,
                                       int* task_metadata_gpu,
                                       int* global_tile_counter_gpu,
                                       int num_ctas,
                                       cudaStream_t stream) {
    dim3 grid(num_ctas);
    dim3 block(GRAD_REDUCE_THREADS_PER_BLOCK);
    int smem_size = GRAD_REDUCE_TILE_SIZE_BYTES;

    CUDA_RUNTIME_CHECK(
        cudaFuncSetAttribute(grad_reduce_kernel_high_sm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    grad_reduce_kernel_high_sm<<<grid, block, smem_size, stream>>>(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu);
}

// ============================================================================
// CPU task-build path (existing)
// ============================================================================

void run_grad_reduce_low_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                            GradReduceTask* grad_reduce_tasks_gpu,
                            int* global_task_counter_gpu,
                            int* task_metadata_gpu,
                            const int total_tasks,
                            cudaStream_t stream,
                            const int num_device_sms) {
    if (total_tasks == 0)
        return;

    // Write task metadata
    int metadata[2] = {total_tasks, 0};  // total_tiles unused by low_sm
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(task_metadata_gpu, metadata, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Copy tasks to GPU
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(grad_reduce_tasks_gpu,
                                       grad_reduce_tasks_cpu,
                                       total_tasks * sizeof(GradReduceTask),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_task_counter_gpu, 0, sizeof(int), stream));

    int num_ctas = min(num_device_sms * 2, total_tasks);
    num_ctas = max(num_ctas, 1);

    launch_grad_reduce_low_sm(grad_reduce_tasks_gpu, task_metadata_gpu, global_task_counter_gpu, num_ctas, stream);
}

void run_grad_reduce_high_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                             GradReduceTask* grad_reduce_tasks_gpu,
                             int* global_tile_counter_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             const int total_tasks,
                             cudaStream_t stream,
                             const int num_device_sms,
                             int num_ctas_override) {
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

    int num_ctas;
    if (num_ctas_override > 0) {
        num_ctas = min(num_ctas_override, total_tiles);
    } else {
        num_ctas = min(num_device_sms * 2, total_tiles);
    }
    num_ctas = max(num_ctas, 1);

    launch_grad_reduce_high_sm(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu, num_ctas, stream);
}

// ============================================================================
// GPU task-build path (tasks already on GPU from build kernel)
// ============================================================================

void run_grad_reduce_low_sm_from_gpu(GradReduceTask* grad_reduce_tasks_gpu,
                                     int* global_task_counter_gpu,
                                     int* task_metadata_gpu,
                                     cudaStream_t stream,
                                     int num_device_sms,
                                     int max_possible_tasks) {
    // Use conservative upper bound; persistent kernel handles over-launch gracefully
    int num_ctas = min(num_device_sms * 2, max_possible_tasks);
    num_ctas = max(num_ctas, 1);

    launch_grad_reduce_low_sm(grad_reduce_tasks_gpu, task_metadata_gpu, global_task_counter_gpu, num_ctas, stream);
}

void run_grad_reduce_high_sm_from_gpu(GradReduceTask* grad_reduce_tasks_gpu,
                                      int* task_tile_offsets_gpu,
                                      int* task_metadata_gpu,
                                      int* global_tile_counter_gpu,
                                      cudaStream_t stream,
                                      int num_device_sms,
                                      int max_possible_tiles,
                                      int num_ctas_override) {
    int num_ctas;
    if (num_ctas_override > 0) {
        num_ctas = min(num_ctas_override, max_possible_tiles);
    } else {
        num_ctas = min(num_device_sms * 2, max_possible_tiles);
    }
    num_ctas = max(num_ctas, 1);

    launch_grad_reduce_high_sm(
        grad_reduce_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu, num_ctas, stream);
}

}  // namespace ultra_ep::kernels
