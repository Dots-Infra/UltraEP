#include "api.cuh"
#include "config.cuh"
#include "launch.cuh"
#include "ptx.cuh"

namespace ultra_ep::kernels {

// ---------------------------------------------------------------------------
// Grad Reduce Task Build
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(32) void build_grad_reduce_tasks_kernel(
    const TaskBuildConfig* __restrict__ config,
    const int32_t* __restrict__ p2l,
    const int32_t* __restrict__ l2p,
    const int32_t* __restrict__ lcnts,
    void* const* __restrict__ remote_grad_ptrs,
    const int64_t* __restrict__ local_master_fc1_ptrs,
    const int64_t* __restrict__ local_master_fc2_ptrs,
    GradReduceTask* __restrict__ tasks,
    int* __restrict__ tile_offsets,
    int* __restrict__ task_metadata) {
    if (threadIdx.x != 0)
        return;

    const int rank_idx = config->rank_idx;
    const int num_nvl_ranks = config->num_nvl_ranks;
    const int num_local_master = config->num_local_master_experts;
    const int num_local_physical = config->num_local_physical_experts;
    const int64_t fc1_numel = config->expert_fc1_numel;
    const int64_t fc2_numel = config->expert_fc2_numel;
    const int64_t total_numel = config->expert_total_numel;
    const int max_rep_dim = config->max_replicas_dim;

    int num_tasks = 0;

    for (int i = 0; i < num_local_master; ++i) {
        int master_phy = rank_idx * num_local_physical + i;
        int master_log = p2l[master_phy];
        int num_replicas = lcnts[master_log];

        float* local_fc1 = reinterpret_cast<float*>(local_master_fc1_ptrs[i]);
        float* local_fc2 = reinterpret_cast<float*>(local_master_fc2_ptrs[i]);

        for (int j = 1; j < num_replicas; ++j) {  // skip the master itself (index 0)
            int replica_phy = l2p[master_log * max_rep_dim + j];
            int replica_rank = replica_phy / num_local_physical;
            int replica_nvl_rank = replica_rank % num_nvl_ranks;
            int replica_local_offset = replica_phy % num_local_physical - num_local_master;

            float* remote_buf = reinterpret_cast<float*>(remote_grad_ptrs[replica_nvl_rank]);
            float* remote_expert_base = remote_buf + replica_local_offset * total_numel;

            // FC1 task
            tasks[num_tasks].master_local_addr = local_fc1;
            tasks[num_tasks].replica_remote_addr = remote_expert_base;
            tasks[num_tasks].numel = static_cast<size_t>(fc1_numel);
            num_tasks++;

            // FC2 task
            tasks[num_tasks].master_local_addr = local_fc2;
            tasks[num_tasks].replica_remote_addr = remote_expert_base + fc1_numel;
            tasks[num_tasks].numel = static_cast<size_t>(fc2_numel);
            num_tasks++;
        }
    }

    // Compute tile offsets (prefix sum) for tile-level grad_reduce scheduling
    tile_offsets[0] = 0;
    for (int t = 0; t < num_tasks; ++t) {
        int tiles = ceil_div(static_cast<int64_t>(tasks[t].numel), static_cast<int64_t>(kGradReduceTileElements));
        tile_offsets[t + 1] = tile_offsets[t] + tiles;
    }
    int total_tiles = (num_tasks > 0) ? tile_offsets[num_tasks] : 0;

    task_metadata[0] = num_tasks;
    task_metadata[1] = total_tiles;
}

void build_grad_reduce_tasks(const TaskBuildConfig* config,
                             const int32_t* physical_to_logical_map,
                             const int32_t* logical_to_physical_map,
                             const int32_t* logical_replica_counts,
                             void* const* remote_grad_ptrs,
                             const int64_t* local_master_fc1_ptrs,
                             const int64_t* local_master_fc2_ptrs,
                             GradReduceTask* tasks,
                             int* task_tile_offsets,
                             int* task_metadata,
                             int* global_task_or_tile_counter,
                             cudaStream_t stream) {
    const auto launch_config = make_launch_config(dim3(1), dim3(32), stream);
    launch_kernel(build_grad_reduce_tasks_kernel,
                  launch_config,
                  config,
                  physical_to_logical_map,
                  logical_to_physical_map,
                  logical_replica_counts,
                  remote_grad_ptrs,
                  local_master_fc1_ptrs,
                  local_master_fc2_ptrs,
                  tasks,
                  task_tile_offsets,
                  task_metadata);

    // Reset task/tile counter for the subsequent main kernel
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_task_or_tile_counter, 0, sizeof(int), stream));
}

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
    int num_elements;      // Number of elements in this tile (may be < kGradReduceTileElements for last tile)
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
    info.global_offset = static_cast<size_t>(info.tile_idx_in_task) * kGradReduceTileElements;

    // Compute number of elements in this tile
    size_t task_numel = tasks[info.task_idx].numel;
    size_t remaining = task_numel - info.global_offset;
    info.num_elements = min(static_cast<size_t>(kGradReduceTileElements), remaining);

    return info;
}

// task_metadata: device pointer to [total_tasks, total_tiles]
__global__ __launch_bounds__(kGradReduceThreadsPerBlock) void grad_reduce_kernel(
    const GradReduceTask* grad_reduce_tasks,
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

void run_grad_reduce(GradReduceTask* tasks,
                     int* task_tile_offsets,
                     int* task_metadata,
                     int* global_tile_counter,
                     cudaStream_t stream,
                     int num_sms) {
    const auto config = make_launch_config(
        dim3(num_sms), dim3(kGradReduceThreadsPerBlock), stream, kGradReduceTileSizeBytes);
    launch_kernel(grad_reduce_kernel, config, tasks, task_tile_offsets, task_metadata, global_tile_counter);
}

}  // namespace ultra_ep::kernels
