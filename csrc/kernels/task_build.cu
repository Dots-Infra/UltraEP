#include "api.cuh"
#include "config.cuh"
#include "launch.cuh"

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

static __device__ __forceinline__ void init_weight_sync_task(WeightSyncTask& task) {
    task.master_local_addr = nullptr;
    task.num_replicas = 0;
    task.numel = 0;
    task.wait_ready_slot = -1;
    task.num_ready_signals = 0;
}

__global__ __launch_bounds__(32) void build_weight_sync_task_lists_kernel(
    const TaskBuildConfig* __restrict__ config,
    const int32_t* __restrict__ p2l,
    const int32_t* __restrict__ l2p,
    const int32_t* __restrict__ lcnts,
    void* const* __restrict__ remote_weight_ptrs,
    const int64_t* __restrict__ local_master_fc1_ptrs,
    const int64_t* __restrict__ local_master_fc2_ptrs,
    __nv_bfloat16* __restrict__ local_replica_weight_buffer,
    WeightSyncTask* __restrict__ stage1_tasks,
    int* __restrict__ stage1_tile_offsets,
    int* __restrict__ stage1_task_metadata,
    int* __restrict__ stage1_remaining_tiles,
    WeightSyncTask* __restrict__ stage2_tasks,
    int* __restrict__ stage2_tile_offsets,
    int* __restrict__ stage2_task_metadata) {
    if (threadIdx.x != 0) {
        return;
    }

    const int rank_idx = config->rank_idx;
    const int domain_base_rank = rank_idx - config->nvl_rank_idx;
    const int num_nvl_ranks = config->num_nvl_ranks;
    const int num_local_master = config->num_local_master_experts;
    const int num_local_physical = config->num_local_physical_experts;
    const int64_t total_numel = config->expert_total_numel;
    const int max_rep_dim = config->max_replicas_dim;
    const int64_t weight_bytes_per_expert = config->expert_total_numel * WEIGHT_ELEMENT_SIZE;
    const size_t shard_numels[2] = {
        static_cast<size_t>(config->expert_fc1_numel),
        static_cast<size_t>(config->expert_fc2_numel),
    };
    const size_t shard_offsets[2] = {
        0,
        static_cast<size_t>(config->expert_fc1_numel),
    };

    int64_t sender_load_bytes[MAX_NVL_DOMAIN_SIZE] = {0};
    int stage1_num_tasks = 0;
    int stage2_num_tasks = 0;

    for (int domain_nvl_rank = 0; domain_nvl_rank < num_nvl_ranks; ++domain_nvl_rank) {
        const int master_rank = domain_base_rank + domain_nvl_rank;
        for (int local_master_idx = 0; local_master_idx < num_local_master; ++local_master_idx) {
            const int master_global_phy = master_rank * num_local_physical + local_master_idx;
            const int logical_expert = p2l[master_global_phy];
            if (logical_expert < 0) {
                continue;
            }

            const int num_replicas = lcnts[logical_expert] - 1;
            if (num_replicas <= 0) {
                continue;
            }

            const bool use_relay = should_use_weight_sync_relay(num_replicas, *config);
            if (!use_relay) {
                sender_load_bytes[domain_nvl_rank] += static_cast<int64_t>(num_replicas) * weight_bytes_per_expert;
                if (master_rank == rank_idx) {
                    __nv_bfloat16* local_master_addrs[2] = {
                        reinterpret_cast<__nv_bfloat16*>(local_master_fc1_ptrs[local_master_idx]),
                        reinterpret_cast<__nv_bfloat16*>(local_master_fc2_ptrs[local_master_idx]),
                    };
                    for (int shard_idx = 0; shard_idx < 2; ++shard_idx) {
                        WeightSyncTask& task = stage1_tasks[stage1_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr = local_master_addrs[shard_idx];
                        task.num_replicas = num_replicas;
                        task.numel = shard_numels[shard_idx];

                        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                            const int replica_phy = l2p[logical_expert * max_rep_dim + replica_idx + 1];
                            const int replica_rank = replica_phy / num_local_physical;
                            const int replica_nvl_rank = replica_rank % num_nvl_ranks;
                            const int replica_local_offset = replica_phy % num_local_physical - num_local_master;

                            __nv_bfloat16* remote_buf =
                                reinterpret_cast<__nv_bfloat16*>(remote_weight_ptrs[replica_nvl_rank]);
                            __nv_bfloat16* remote_expert_base =
                                remote_buf + replica_local_offset * total_numel + shard_offsets[shard_idx];
                            task.replica_remote_addrs[replica_idx] = remote_expert_base;
                        }
                    }
                }
                continue;
            }

            const int relay_count = choose_weight_sync_relay_count(num_replicas, *config);
            if (relay_count <= 0) {
                continue;
            }

            bool replica_selected[MAX_NVL_DOMAIN_SIZE - 1] = {false};
            int relay_replica_indices[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int relay_global_ranks[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int relay_nvl_ranks[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int relay_local_offsets[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int relay_child_counts[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int leaf_owner_relay[MAX_NVL_DOMAIN_SIZE - 1];
            for (int replica_idx = 0; replica_idx < MAX_NVL_DOMAIN_SIZE - 1; ++replica_idx) {
                leaf_owner_relay[replica_idx] = -1;
            }

            for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                int best_replica_idx = -1;
                int best_rank_used_penalty = 2;
                int64_t best_sender_load = 0;
                int best_rank = 0;
                int best_nvl_rank = 0;
                int best_local_offset = 0;

                for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                    if (replica_selected[replica_idx]) {
                        continue;
                    }

                    const int replica_phy = l2p[logical_expert * max_rep_dim + replica_idx + 1];
                    const int replica_rank = replica_phy / num_local_physical;
                    const int replica_nvl_rank = replica_rank % num_nvl_ranks;
                    const int replica_local_offset = replica_phy % num_local_physical - num_local_master;

                    bool rank_used = false;
                    for (int prev = 0; prev < relay_idx; ++prev) {
                        if (relay_global_ranks[prev] == replica_rank) {
                            rank_used = true;
                            break;
                        }
                    }
                    const int rank_used_penalty = rank_used ? 1 : 0;
                    const int64_t candidate_sender_load = sender_load_bytes[replica_nvl_rank];

                    const bool is_better = best_replica_idx < 0 || rank_used_penalty < best_rank_used_penalty ||
                        (rank_used_penalty == best_rank_used_penalty &&
                         (candidate_sender_load < best_sender_load ||
                          (candidate_sender_load == best_sender_load &&
                           (replica_rank < best_rank ||
                            (replica_rank == best_rank && replica_idx < best_replica_idx)))));
                    if (!is_better) {
                        continue;
                    }

                    best_replica_idx = replica_idx;
                    best_rank_used_penalty = rank_used_penalty;
                    best_sender_load = candidate_sender_load;
                    best_rank = replica_rank;
                    best_nvl_rank = replica_nvl_rank;
                    best_local_offset = replica_local_offset;
                }

                if (best_replica_idx < 0) {
                    break;
                }

                replica_selected[best_replica_idx] = true;
                relay_replica_indices[relay_idx] = best_replica_idx;
                relay_global_ranks[relay_idx] = best_rank;
                relay_nvl_ranks[relay_idx] = best_nvl_rank;
                relay_local_offsets[relay_idx] = best_local_offset;
            }

            int leaf_replica_indices[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            int leaf_count = 0;
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                if (!replica_selected[replica_idx]) {
                    leaf_replica_indices[leaf_count++] = replica_idx;
                }
            }

            int64_t projected_relay_loads[MAX_NVL_DOMAIN_SIZE - 1] = {0};
            for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                projected_relay_loads[relay_idx] = sender_load_bytes[relay_nvl_ranks[relay_idx]];
            }

            for (int leaf_order = 0; leaf_order < leaf_count; ++leaf_order) {
                const int replica_idx = leaf_replica_indices[leaf_order];
                int owner_relay = -1;
                if (leaf_order < relay_count) {
                    owner_relay = leaf_order;
                } else {
                    for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                        const bool is_better = owner_relay < 0 ||
                            projected_relay_loads[relay_idx] < projected_relay_loads[owner_relay] ||
                            (projected_relay_loads[relay_idx] == projected_relay_loads[owner_relay] &&
                             (relay_child_counts[relay_idx] < relay_child_counts[owner_relay] ||
                              (relay_child_counts[relay_idx] == relay_child_counts[owner_relay] &&
                               (relay_global_ranks[relay_idx] < relay_global_ranks[owner_relay] ||
                                (relay_global_ranks[relay_idx] == relay_global_ranks[owner_relay] &&
                                 relay_replica_indices[relay_idx] < relay_replica_indices[owner_relay])))));
                        if (is_better) {
                            owner_relay = relay_idx;
                        }
                    }
                }

                if (owner_relay < 0) {
                    continue;
                }

                leaf_owner_relay[replica_idx] = owner_relay;
                relay_child_counts[owner_relay] += 1;
                projected_relay_loads[owner_relay] += weight_bytes_per_expert;
            }

            sender_load_bytes[domain_nvl_rank] += static_cast<int64_t>(relay_count) * weight_bytes_per_expert;
            for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                sender_load_bytes[relay_nvl_ranks[relay_idx]] +=
                    static_cast<int64_t>(relay_child_counts[relay_idx]) * weight_bytes_per_expert;
            }

            if (master_rank == rank_idx) {
                __nv_bfloat16* local_master_addrs[2] = {
                    reinterpret_cast<__nv_bfloat16*>(local_master_fc1_ptrs[local_master_idx]),
                    reinterpret_cast<__nv_bfloat16*>(local_master_fc2_ptrs[local_master_idx]),
                };
                for (int shard_idx = 0; shard_idx < 2; ++shard_idx) {
                    const int num_chunks = weight_sync_num_chunks(shard_numels[shard_idx]);
                    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        WeightSyncTask& task = stage1_tasks[stage1_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr =
                            local_master_addrs[shard_idx] + weight_sync_chunk_offset_elements(chunk_idx);
                        task.num_replicas = relay_count;
                        task.numel = weight_sync_chunk_numel(shard_numels[shard_idx], chunk_idx);
                        task.num_ready_signals = relay_count;

                        for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                            __nv_bfloat16* remote_buf =
                                reinterpret_cast<__nv_bfloat16*>(remote_weight_ptrs[relay_nvl_ranks[relay_idx]]);
                            __nv_bfloat16* remote_expert_base =
                                remote_buf + relay_local_offsets[relay_idx] * total_numel + shard_offsets[shard_idx];
                            task.replica_remote_addrs[relay_idx] =
                                remote_expert_base + weight_sync_chunk_offset_elements(chunk_idx);
                            task.ready_signal_slots[relay_idx] = weight_sync_ready_flag_slot(
                                *config, relay_local_offsets[relay_idx], shard_idx, chunk_idx);
                            task.ready_signal_nvl_ranks[relay_idx] = relay_nvl_ranks[relay_idx];
                        }
                    }
                }
            }

            for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                if (relay_global_ranks[relay_idx] != rank_idx || relay_child_counts[relay_idx] <= 0) {
                    continue;
                }

                __nv_bfloat16* local_relay_base =
                    local_replica_weight_buffer + relay_local_offsets[relay_idx] * total_numel;
                for (int shard_idx = 0; shard_idx < 2; ++shard_idx) {
                    const int num_chunks = weight_sync_num_chunks(shard_numels[shard_idx]);
                    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        WeightSyncTask& task = stage2_tasks[stage2_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr =
                            local_relay_base + shard_offsets[shard_idx] + weight_sync_chunk_offset_elements(chunk_idx);
                        task.num_replicas = relay_child_counts[relay_idx];
                        task.numel = weight_sync_chunk_numel(shard_numels[shard_idx], chunk_idx);
                        task.wait_ready_slot =
                            weight_sync_ready_flag_slot(*config, relay_local_offsets[relay_idx], shard_idx, chunk_idx);

                        int child_idx = 0;
                        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                            if (leaf_owner_relay[replica_idx] != relay_idx) {
                                continue;
                            }

                            const int replica_phy = l2p[logical_expert * max_rep_dim + replica_idx + 1];
                            const int replica_rank = replica_phy / num_local_physical;
                            const int replica_nvl_rank = replica_rank % num_nvl_ranks;
                            const int replica_local_offset = replica_phy % num_local_physical - num_local_master;

                            __nv_bfloat16* remote_buf =
                                reinterpret_cast<__nv_bfloat16*>(remote_weight_ptrs[replica_nvl_rank]);
                            __nv_bfloat16* remote_expert_base =
                                remote_buf + replica_local_offset * total_numel + shard_offsets[shard_idx];
                            task.replica_remote_addrs[child_idx++] =
                                remote_expert_base + weight_sync_chunk_offset_elements(chunk_idx);
                        }
                    }
                }
            }
        }
    }

    stage1_tile_offsets[0] = 0;
    for (int task_idx = 0; task_idx < stage1_num_tasks; ++task_idx) {
        const int num_tiles = weight_sync_num_tiles(stage1_tasks[task_idx].numel);
        stage1_tile_offsets[task_idx + 1] = stage1_tile_offsets[task_idx] + num_tiles;
        stage1_remaining_tiles[task_idx] = num_tiles;
    }
    stage1_task_metadata[0] = stage1_num_tasks;
    stage1_task_metadata[1] = stage1_num_tasks > 0 ? stage1_tile_offsets[stage1_num_tasks] : 0;

    stage2_tile_offsets[0] = 0;
    for (int task_idx = 0; task_idx < stage2_num_tasks; ++task_idx) {
        const int num_tiles = weight_sync_num_tiles(stage2_tasks[task_idx].numel);
        stage2_tile_offsets[task_idx + 1] = stage2_tile_offsets[task_idx] + num_tiles;
    }
    stage2_task_metadata[0] = stage2_num_tasks;
    stage2_task_metadata[1] = stage2_num_tasks > 0 ? stage2_tile_offsets[stage2_num_tasks] : 0;
}

void build_weight_sync_task_lists(const TaskBuildConfig* config_gpu,
                                  const int32_t* p2l_gpu,
                                  const int32_t* l2p_gpu,
                                  const int32_t* lcnts_gpu,
                                  void* const* remote_weight_ptrs_gpu,
                                  const int64_t* local_master_fc1_ptrs_gpu,
                                  const int64_t* local_master_fc2_ptrs_gpu,
                                  __nv_bfloat16* local_replica_weight_buffer,
                                  WeightSyncTask* stage1_tasks_gpu,
                                  int* stage1_task_tile_offsets_gpu,
                                  int* stage1_task_metadata_gpu,
                                  int* stage1_task_remaining_tiles_gpu,
                                  int* stage1_global_tile_counter_gpu,
                                  WeightSyncTask* stage2_tasks_gpu,
                                  int* stage2_task_tile_offsets_gpu,
                                  int* stage2_task_metadata_gpu,
                                  int* stage2_global_tile_counter_gpu,
                                  cudaStream_t stream) {
    const auto config = make_launch_config(dim3(1), dim3(32), stream);
    launch_kernel(build_weight_sync_task_lists_kernel,
                          config,
                          config_gpu,
                          p2l_gpu,
                          l2p_gpu,
                          lcnts_gpu,
                          remote_weight_ptrs_gpu,
                          local_master_fc1_ptrs_gpu,
                          local_master_fc2_ptrs_gpu,
                          local_replica_weight_buffer,
                          stage1_tasks_gpu,
                          stage1_task_tile_offsets_gpu,
                          stage1_task_metadata_gpu,
                          stage1_task_remaining_tiles_gpu,
                          stage2_tasks_gpu,
                          stage2_task_tile_offsets_gpu,
                          stage2_task_metadata_gpu);

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(stage1_global_tile_counter_gpu, 0, sizeof(int), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(stage2_global_tile_counter_gpu, 0, sizeof(int), stream));
}

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
    const auto config = make_launch_config(dim3(1), dim3(32), stream);
    launch_kernel(build_grad_reduce_tasks_kernel,
                          config,
                          config_gpu,
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
