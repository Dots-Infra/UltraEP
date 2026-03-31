#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

// ============================================================================
// GPU Task Build Kernels
// ============================================================================
//
// Build WeightSyncTask / GradReduceTask arrays entirely on GPU, reading from
// the GPU-resident placement buffer.  This eliminates the CPU→GPU synchronization
// point in the weight_sync / grad_reduce hot path.
//
// Each kernel runs with a single warp (<<<1, 32>>>).  The workload is tiny
// (O(num_local_master × num_replicas)), so lane 0 does all the serial work
// while the remaining lanes are idle.  Total kernel time: ~2-5 µs.
//
// Outputs:
//   - task array  (WeightSyncTask[] or GradReduceTask[])
//   - tile_offsets prefix sum  (int[])
//   - task_metadata  ({total_tasks, total_tiles})
// ============================================================================

static __device__ __forceinline__ int ceil_div_i64(int64_t a, int64_t b) {
    return static_cast<int>((a + b - 1) / b);
}

// ---------------------------------------------------------------------------
// Weight Sync Task Build
// ---------------------------------------------------------------------------

__global__ void build_weight_sync_tasks_kernel(const TaskBuildConfig* __restrict__ config,
                                               const int32_t* __restrict__ p2l,
                                               const int32_t* __restrict__ l2p,
                                               const int32_t* __restrict__ lcnts,
                                               void* const* __restrict__ remote_weight_ptrs,
                                               const int64_t* __restrict__ local_master_fc1_ptrs,
                                               const int64_t* __restrict__ local_master_fc2_ptrs,
                                               WeightSyncTask* __restrict__ tasks,
                                               int* __restrict__ tile_offsets,
                                               int* __restrict__ task_metadata) {
    if (threadIdx.x != 0)
        return;

    // Load config into registers
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
        int num_replicas = lcnts[master_log] - 1;  // exclude the master itself

        if (num_replicas <= 0)
            continue;

        const bool use_relay = should_use_weight_sync_relay(num_replicas, *config);
        const int stage_targets = use_relay ? choose_weight_sync_relay_count(num_replicas, *config) : num_replicas;
        if (stage_targets <= 0) {
            continue;
        }

        // FC1 task
        WeightSyncTask& fc1 = tasks[num_tasks];
        fc1.master_local_addr = reinterpret_cast<__nv_bfloat16*>(local_master_fc1_ptrs[i]);
        fc1.num_replicas = stage_targets;
        fc1.numel = static_cast<size_t>(fc1_numel);

        // FC2 task
        WeightSyncTask& fc2 = tasks[num_tasks + 1];
        fc2.master_local_addr = reinterpret_cast<__nv_bfloat16*>(local_master_fc2_ptrs[i]);
        fc2.num_replicas = stage_targets;
        fc2.numel = static_cast<size_t>(fc2_numel);

        // Fill stage targets. Direct mode uses all replicas; relay mode only seeds
        // the chosen relay replicas, which then forward to their assigned leaves.
        for (int j = 0; j < stage_targets; ++j) {
            int replica_phy = l2p[master_log * max_rep_dim + j + 1];
            int replica_rank = replica_phy / num_local_physical;
            int replica_nvl_rank = replica_rank % num_nvl_ranks;
            int replica_local_offset = replica_phy % num_local_physical - num_local_master;

            __nv_bfloat16* remote_buf = reinterpret_cast<__nv_bfloat16*>(remote_weight_ptrs[replica_nvl_rank]);
            __nv_bfloat16* remote_expert_base = remote_buf + replica_local_offset * total_numel;

            fc1.replica_remote_addrs[j] = remote_expert_base;
            fc2.replica_remote_addrs[j] = remote_expert_base + fc1_numel;
        }

        num_tasks += 2;
    }

    // Compute tile offsets (prefix sum)
    tile_offsets[0] = 0;
    for (int t = 0; t < num_tasks; ++t) {
        int tiles = ceil_div_i64(static_cast<int64_t>(tasks[t].numel), WEIGHT_SYNC_TILE_ELEMENTS);
        tile_offsets[t + 1] = tile_offsets[t] + tiles;
    }
    int total_tiles = (num_tasks > 0) ? tile_offsets[num_tasks] : 0;

    // Write metadata for the main kernel
    task_metadata[0] = num_tasks;
    task_metadata[1] = total_tiles;
}

void build_weight_sync_tasks(const TaskBuildConfig* config_gpu,
                             const int32_t* p2l_gpu,
                             const int32_t* l2p_gpu,
                             const int32_t* lcnts_gpu,
                             void* const* remote_weight_ptrs_gpu,
                             const int64_t* local_master_fc1_ptrs_gpu,
                             const int64_t* local_master_fc2_ptrs_gpu,
                             WeightSyncTask* tasks_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             int* global_tile_counter_gpu,
                             cudaStream_t stream) {
    build_weight_sync_tasks_kernel<<<1, 32, 0, stream>>>(config_gpu,
                                                         p2l_gpu,
                                                         l2p_gpu,
                                                         lcnts_gpu,
                                                         remote_weight_ptrs_gpu,
                                                         local_master_fc1_ptrs_gpu,
                                                         local_master_fc2_ptrs_gpu,
                                                         tasks_gpu,
                                                         task_tile_offsets_gpu,
                                                         task_metadata_gpu);

    // Reset tile counter for the subsequent main kernel
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_tile_counter_gpu, 0, sizeof(int), stream));
}

__global__ void build_weight_sync_relay_tasks_kernel(const TaskBuildConfig* __restrict__ config,
                                                     const int32_t* __restrict__ p2l,
                                                     const int32_t* __restrict__ l2p,
                                                     const int32_t* __restrict__ lcnts,
                                                     void* const* __restrict__ remote_weight_ptrs,
                                                     __nv_bfloat16* __restrict__ local_replica_weight_buffer,
                                                     WeightSyncTask* __restrict__ tasks,
                                                     int* __restrict__ tile_offsets,
                                                     int* __restrict__ task_metadata) {
    if (threadIdx.x != 0)
        return;

    const int rank_idx = config->rank_idx;
    const int num_nvl_ranks = config->num_nvl_ranks;
    const int num_local_master = config->num_local_master_experts;
    const int num_local_physical = config->num_local_physical_experts;
    const int num_local_redundant = config->num_local_redundant_experts;
    const int64_t fc1_numel = config->expert_fc1_numel;
    const int64_t fc2_numel = config->expert_fc2_numel;
    const int64_t total_numel = config->expert_total_numel;
    const int max_rep_dim = config->max_replicas_dim;

    int num_tasks = 0;

    for (int local_replica_offset = 0; local_replica_offset < num_local_redundant; ++local_replica_offset) {
        const int local_phy = num_local_master + local_replica_offset;
        const int global_phy = rank_idx * num_local_physical + local_phy;
        const int logical_expert = p2l[global_phy];
        if (logical_expert < 0) {
            continue;
        }

        const int num_replicas = lcnts[logical_expert] - 1;
        if (!should_use_weight_sync_relay(num_replicas, *config)) {
            continue;
        }

        const int relay_count = choose_weight_sync_relay_count(num_replicas, *config);
        if (relay_count <= 0) {
            continue;
        }

        int replica_idx = -1;
        for (int j = 0; j < num_replicas; ++j) {
            if (l2p[logical_expert * max_rep_dim + j + 1] == global_phy) {
                replica_idx = j;
                break;
            }
        }
        if (replica_idx < 0 || replica_idx >= relay_count) {
            continue;
        }

        const int child_count = relay_stage_child_count(num_replicas, relay_count, replica_idx);
        if (child_count <= 0) {
            continue;
        }

        __nv_bfloat16* local_expert_base = local_replica_weight_buffer + local_replica_offset * total_numel;

        WeightSyncTask& fc1 = tasks[num_tasks];
        fc1.master_local_addr = local_expert_base;
        fc1.num_replicas = child_count;
        fc1.numel = static_cast<size_t>(fc1_numel);

        WeightSyncTask& fc2 = tasks[num_tasks + 1];
        fc2.master_local_addr = local_expert_base + fc1_numel;
        fc2.num_replicas = child_count;
        fc2.numel = static_cast<size_t>(fc2_numel);

        int child_idx = 0;
        const int leaf_count = num_replicas - relay_count;
        for (int leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
            if (relay_stage_leaf_owner(leaf_idx, relay_count) != replica_idx) {
                continue;
            }

            const int replica_phy = l2p[logical_expert * max_rep_dim + relay_count + leaf_idx + 1];
            const int replica_rank = replica_phy / num_local_physical;
            const int replica_nvl_rank = replica_rank % num_nvl_ranks;
            const int replica_local_offset = replica_phy % num_local_physical - num_local_master;

            __nv_bfloat16* remote_buf = reinterpret_cast<__nv_bfloat16*>(remote_weight_ptrs[replica_nvl_rank]);
            __nv_bfloat16* remote_expert_base = remote_buf + replica_local_offset * total_numel;

            fc1.replica_remote_addrs[child_idx] = remote_expert_base;
            fc2.replica_remote_addrs[child_idx] = remote_expert_base + fc1_numel;
            ++child_idx;
        }

        num_tasks += 2;
    }

    tile_offsets[0] = 0;
    for (int t = 0; t < num_tasks; ++t) {
        int tiles = ceil_div_i64(static_cast<int64_t>(tasks[t].numel), WEIGHT_SYNC_TILE_ELEMENTS);
        tile_offsets[t + 1] = tile_offsets[t] + tiles;
    }
    const int total_tiles = (num_tasks > 0) ? tile_offsets[num_tasks] : 0;
    task_metadata[0] = num_tasks;
    task_metadata[1] = total_tiles;
}

void build_weight_sync_relay_tasks(const TaskBuildConfig* config_gpu,
                                   const int32_t* p2l_gpu,
                                   const int32_t* l2p_gpu,
                                   const int32_t* lcnts_gpu,
                                   void* const* remote_weight_ptrs_gpu,
                                   __nv_bfloat16* local_replica_weight_buffer,
                                   WeightSyncTask* tasks_gpu,
                                   int* task_tile_offsets_gpu,
                                   int* task_metadata_gpu,
                                   int* global_tile_counter_gpu,
                                   cudaStream_t stream) {
    build_weight_sync_relay_tasks_kernel<<<1, 32, 0, stream>>>(config_gpu,
                                                               p2l_gpu,
                                                               l2p_gpu,
                                                               lcnts_gpu,
                                                               remote_weight_ptrs_gpu,
                                                               local_replica_weight_buffer,
                                                               tasks_gpu,
                                                               task_tile_offsets_gpu,
                                                               task_metadata_gpu);

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_tile_counter_gpu, 0, sizeof(int), stream));
}

// ---------------------------------------------------------------------------
// Grad Reduce Task Build
// ---------------------------------------------------------------------------

__global__ void build_grad_reduce_tasks_kernel(const TaskBuildConfig* __restrict__ config,
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

    // Compute tile offsets (prefix sum) — used by high_sm mode
    tile_offsets[0] = 0;
    for (int t = 0; t < num_tasks; ++t) {
        int tiles = ceil_div_i64(static_cast<int64_t>(tasks[t].numel), GRAD_REDUCE_TILE_ELEMENTS);
        tile_offsets[t + 1] = tile_offsets[t] + tiles;
    }
    int total_tiles = (num_tasks > 0) ? tile_offsets[num_tasks] : 0;

    task_metadata[0] = num_tasks;
    task_metadata[1] = total_tiles;
}

void build_grad_reduce_tasks(const TaskBuildConfig* config_gpu,
                             const int32_t* p2l_gpu,
                             const int32_t* l2p_gpu,
                             const int32_t* lcnts_gpu,
                             void* const* remote_grad_ptrs_gpu,
                             const int64_t* local_master_fc1_ptrs_gpu,
                             const int64_t* local_master_fc2_ptrs_gpu,
                             GradReduceTask* tasks_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             int* global_task_or_tile_counter_gpu,
                             cudaStream_t stream) {
    build_grad_reduce_tasks_kernel<<<1, 32, 0, stream>>>(config_gpu,
                                                         p2l_gpu,
                                                         l2p_gpu,
                                                         lcnts_gpu,
                                                         remote_grad_ptrs_gpu,
                                                         local_master_fc1_ptrs_gpu,
                                                         local_master_fc2_ptrs_gpu,
                                                         tasks_gpu,
                                                         task_tile_offsets_gpu,
                                                         task_metadata_gpu);

    // Reset task/tile counter for the subsequent main kernel
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_task_or_tile_counter_gpu, 0, sizeof(int), stream));
}

}  // namespace ultra_ep::kernels
