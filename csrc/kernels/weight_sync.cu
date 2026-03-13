#include <vector>

#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

// ============================================================================
// Weight Sync Kernel: Broadcast Master Weights to Replicas
// ============================================================================
//
// Design for hot master optimization:
// - Each task represents a master broadcasting to N replicas
// - For each tile:
//   1. TMA Load tile from local master to SMEM (async)
//   2. Issue N TMA stores to N different replica addresses
//   3. Pipeline: overlap TMA Load[N+1] with TMA Store[N]
//
// Timeline for consecutive tiles with double buffering:
//   Tile 0: [TMA_Load₀] [wait_load] [TMA_Store₀...]
//   Tile 1:                         [TMA_Load₁] [wait_load] [wait_store₀] [TMA_Store₁...]
//   Tile 2:                                                               [TMA_Load₂] ...
//
// This approach loads SMEM only once per tile, regardless of replica count.
// For hot masters with many replicas, this is significantly more efficient
// than separate load-store operations for each replica.
//
// Tile-level parallelism with persistent kernel:
// - Each CTA grabs tiles via atomic counter
// - Multiple CTAs can process different tiles of the same task concurrently
// ============================================================================

// Structure to help with tile-to-task mapping
struct WeightSyncTileInfo {
    int task_idx;
    int tile_idx_in_task;
    size_t element_offset;
    int num_elements;
};

// Map a global tile index to task and tile within task
__device__ __forceinline__ WeightSyncTileInfo get_weight_sync_tile_info(const WeightSyncTask* tasks,
                                                                        const int* task_tile_offsets,
                                                                        int num_tasks,
                                                                        int global_tile_idx) {
    WeightSyncTileInfo info;

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
    info.element_offset = (size_t)info.tile_idx_in_task * WEIGHT_SYNC_TILE_ELEMENTS;

    // Compute number of elements in this tile
    size_t task_numel = tasks[info.task_idx].numel;
    size_t remaining = task_numel - info.element_offset;
    info.num_elements = min((size_t)WEIGHT_SYNC_TILE_ELEMENTS, remaining);

    return info;
}

// Weight sync kernel with double buffering for true pipelining
// Pipeline: TMA_Load[N+1] overlaps with TMA_Store[N]
// This achieves true overlap of local HBM reads and remote NVLINK writes.
//
// task_metadata: device pointer to [total_tasks, total_tiles] (set by CPU or GPU task build)
__global__ void weight_sync_kernel(const WeightSyncTask* weight_sync_tasks,
                                   const int* task_tile_offsets,
                                   const int* task_metadata,
                                   int* global_tile_counter) {
    // Double-buffered shared memory
    extern __shared__ __nv_bfloat16 smem_base[];
    __nv_bfloat16* smem[2] = {smem_base, smem_base + WEIGHT_SYNC_TILE_ELEMENTS};

    // Mbarriers for TMA load synchronization
    ptx::mbarrier* mbarriers = ptx::create_mbarriers<2>();
    __shared__ ptx::arrival_phase phases[2];

    const bool is_leader = (threadIdx.x == 0);

    // Read task metadata from device memory
    __shared__ int total_tasks;
    __shared__ int total_tiles;
    if (is_leader) {
        total_tasks = task_metadata[0];
        total_tiles = task_metadata[1];
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

    // Shared tile indices
    __shared__ int tile_indices[2];
    __shared__ bool has_pending_store;

    if (is_leader) {
        has_pending_store = false;
    }

    // Fetch first tile
    if (is_leader) {
        tile_indices[0] = atomicAdd(global_tile_counter, 1);
    }
    __syncthreads();

    int cur_buf = 0;

    // Main pipeline loop
    while (tile_indices[cur_buf] < total_tiles) {
        int my_tile_idx = tile_indices[cur_buf];

        // Get current tile info
        WeightSyncTileInfo tile =
            get_weight_sync_tile_info(weight_sync_tasks, task_tile_offsets, total_tasks, my_tile_idx);
        WeightSyncTask task = weight_sync_tasks[tile.task_idx];

        size_t bytes = tile.num_elements * sizeof(__nv_bfloat16);
        bytes = (bytes + 15) & ~15;

        // Issue TMA Load for current tile
        if (is_leader) {
            ptx::mbarrier_arrive_and_set_tx(&mbarriers[cur_buf], bytes);
            ptx::tma_load_1d(smem[cur_buf],
                             task.master_local_addr + tile.element_offset,
                             &mbarriers[cur_buf],
                             bytes,
                             ptx::TMACacheHint::kEvictNormal);
        }

        // Prefetch next tile index while TMA Load is in flight
        int next_buf = 1 - cur_buf;
        if (is_leader) {
            tile_indices[next_buf] = atomicAdd(global_tile_counter, 1);
        }

        // Wait for current TMA Load to complete
        if (is_leader) {
            ptx::mbarrier_wait_and_flip_phase(&mbarriers[cur_buf], phases[cur_buf]);
        }
        __syncthreads();

        // If there's a pending store from previous iteration, wait for it
        // This ensures the previous buffer is free before we overwrite it
        if (has_pending_store) {
            if (is_leader) {
                ptx::tma_store_wait<0>();
            }
            __syncthreads();
            __threadfence_system();
        }

        // Fence and issue TMA stores for current tile
        if (is_leader) {
            ptx::tma_store_fence();
            for (int r = 0; r < task.num_replicas; ++r) {
                __nv_bfloat16* replica_addr = task.replica_remote_addrs[r] + tile.element_offset;
                ptx::tma_store_1d(replica_addr, smem[cur_buf], bytes, ptx::TMACacheHint::kEvictNormal);
            }
            ptx::tma_store_commit();
            has_pending_store = true;
        }
        __syncthreads();

        // Switch buffers
        cur_buf = next_buf;
    }

    // Wait for any remaining pending stores
    if (has_pending_store) {
        if (is_leader) {
            ptx::tma_store_wait<0>();
        }
        __syncthreads();
        __threadfence_system();
    }

    // Cleanup mbarriers
    __syncthreads();
    if (is_leader) {
        for (int i = 0; i < 2; i++) {
            ptx::mbarrier_invalidate(&mbarriers[i]);
        }
    }
}

// Helper to configure and launch the weight sync kernel
static void launch_weight_sync_kernel(WeightSyncTask* weight_sync_tasks_gpu,
                                      int* task_tile_offsets_gpu,
                                      int* task_metadata_gpu,
                                      int* global_tile_counter_gpu,
                                      int num_ctas,
                                      cudaStream_t stream) {
    dim3 grid(num_ctas);
    dim3 block(WEIGHT_SYNC_THREADS_PER_BLOCK);

    // Double buffer shared memory
    int smem_size = WEIGHT_SYNC_TILE_SIZE_BYTES * WEIGHT_SYNC_PIPELINE_STAGES;

    CUDA_RUNTIME_CHECK(
        cudaFuncSetAttribute(weight_sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    weight_sync_kernel<<<grid, block, smem_size, stream>>>(
        weight_sync_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu);
}

// CPU task-build path: H2D copy tasks + tile offsets, write metadata, then launch kernel
void run_weight_sync(const WeightSyncTask* weight_sync_tasks_cpu,
                     WeightSyncTask* weight_sync_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     int* task_tile_offsets_cpu,
                     int* task_metadata_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms) {
    if (total_tasks == 0)
        return;

    // Compute tile offsets on CPU (prefix sum of tile counts per task)
    task_tile_offsets_cpu[0] = 0;
    for (int i = 0; i < total_tasks; ++i) {
        int num_tiles = (weight_sync_tasks_cpu[i].numel + WEIGHT_SYNC_TILE_ELEMENTS - 1) / WEIGHT_SYNC_TILE_ELEMENTS;
        task_tile_offsets_cpu[i + 1] = task_tile_offsets_cpu[i] + num_tiles;
    }
    int total_tiles = task_tile_offsets_cpu[total_tasks];

    // Write task metadata to GPU
    int metadata[2] = {total_tasks, total_tiles};
    CUDA_RUNTIME_CHECK(
        cudaMemcpyAsync(task_metadata_gpu, metadata, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Copy tasks and tile offsets from CPU to GPU
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(weight_sync_tasks_gpu,
                                       weight_sync_tasks_cpu,
                                       total_tasks * sizeof(WeightSyncTask),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        task_tile_offsets_gpu, task_tile_offsets_cpu, (total_tasks + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_tile_counter_gpu, 0, sizeof(int), stream));

    // Configure kernel launch
    int num_ctas = min(num_device_sms * 2, total_tiles);
    num_ctas = max(num_ctas, 1);

    launch_weight_sync_kernel(
        weight_sync_tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu, num_ctas, stream);
}

// GPU task-build path: tasks already on GPU (written by build_weight_sync_tasks kernel)
void run_weight_sync_from_gpu(WeightSyncTask* tasks_gpu,
                              int* task_tile_offsets_gpu,
                              int* task_metadata_gpu,
                              int* global_tile_counter_gpu,
                              cudaStream_t stream,
                              int num_device_sms,
                              int max_possible_tiles) {
    // Use conservative upper bound for grid size; persistent kernel handles over-launch
    int num_ctas = min(num_device_sms * 2, max_possible_tiles);
    num_ctas = max(num_ctas, 1);

    launch_weight_sync_kernel(tasks_gpu, task_tile_offsets_gpu, task_metadata_gpu, global_tile_counter_gpu,
                              num_ctas, stream);
}

}  // namespace ultra_ep::kernels
