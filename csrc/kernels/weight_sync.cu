#include "api.cuh"
#include "config.cuh"
#include "launch.cuh"
#include "ptx.cuh"

namespace ultra_ep::kernels {

// Helper functions

static constexpr int kWeightSyncThreadCopyMaxNvlRanks = 8;
static constexpr int kWeightSyncThreadCopyCtaMultiplier = 4;

static __host__ __device__ __forceinline__ size_t weight_sync_chunk_offset_bytes(const int chunk_idx) {
    return static_cast<size_t>(chunk_idx) * kWeightSyncRelayChunkTiles * kWeightSyncTileSizeBytes;
}

static __host__ __device__ __forceinline__ size_t weight_sync_chunk_num_bytes(const size_t total_bytes,
                                                                              const int chunk_idx) {
    const size_t chunk_offset = weight_sync_chunk_offset_bytes(chunk_idx);
    if (chunk_offset >= total_bytes) {
        return 0;
    }

    const size_t chunk_capacity = static_cast<size_t>(kWeightSyncRelayChunkTiles) * kWeightSyncTileSizeBytes;
    const size_t remaining = total_bytes - chunk_offset;
    return remaining < chunk_capacity ? remaining : chunk_capacity;
}

static __host__ __device__ __forceinline__ int floor_sqrt_int(const int x) {
    int root = 0;
    while ((root + 1) * (root + 1) <= x) {
        ++root;
    }
    return root;
}

static __host__ __device__ __forceinline__ int weight_sync_num_shards(const TaskBuildConfig& config) {
    return config.expert_weight_scale_total_numel > 0 ? 4 : 2;
}

static __host__ __device__ __forceinline__ size_t weight_sync_shard_num_bytes(const TaskBuildConfig& config,
                                                                              const int shard_idx) {
    if (shard_idx == 0) {
        return static_cast<size_t>(config.expert_fc1_numel) * config.weight_data_element_bytes;
    }
    if (shard_idx == 1) {
        return static_cast<size_t>(config.expert_fc2_numel) * config.weight_data_element_bytes;
    }
    if (shard_idx == 2) {
        return static_cast<size_t>(config.expert_fc1_weight_scale_bytes);
    }
    return static_cast<size_t>(config.expert_fc2_weight_scale_bytes);
}

static __host__ __device__ __forceinline__ size_t weight_sync_shard_offset_bytes(const TaskBuildConfig& config,
                                                                                 const int shard_idx) {
    if (shard_idx == 0 || shard_idx == 2) {
        return 0;
    }
    if (shard_idx == 1) {
        return static_cast<size_t>(config.expert_fc1_numel) * config.weight_data_element_bytes;
    }
    return static_cast<size_t>(config.expert_weight_scale_fc2_offset_bytes);
}

static __host__ __device__ __forceinline__ size_t weight_sync_expert_buffer_bytes(const TaskBuildConfig& config,
                                                                                  const bool scale_shard) {
    if (scale_shard) {
        return static_cast<size_t>(config.expert_weight_scale_stride_bytes);
    }
    return static_cast<size_t>(config.expert_total_numel) * config.weight_data_element_bytes;
}

static __host__ __device__ __forceinline__ size_t weight_sync_expert_total_bytes(const TaskBuildConfig& config) {
    size_t total = 0;
    const int num_shards = weight_sync_num_shards(config);
    for (int shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        total += weight_sync_shard_num_bytes(config, shard_idx);
    }
    return total;
}

static __device__ __forceinline__ const uint8_t* weight_sync_local_master_addr(
    const int64_t* __restrict__ local_master_fc1_ptrs,
    const int64_t* __restrict__ local_master_fc2_ptrs,
    const int64_t* __restrict__ local_master_fc1_scale_ptrs,
    const int64_t* __restrict__ local_master_fc2_scale_ptrs,
    const int local_master_idx,
    const int shard_idx) {
    if (shard_idx == 0) {
        return reinterpret_cast<const uint8_t*>(local_master_fc1_ptrs[local_master_idx]);
    }
    if (shard_idx == 1) {
        return reinterpret_cast<const uint8_t*>(local_master_fc2_ptrs[local_master_idx]);
    }
    if (shard_idx == 2) {
        return reinterpret_cast<const uint8_t*>(local_master_fc1_scale_ptrs[local_master_idx]);
    }
    return reinterpret_cast<const uint8_t*>(local_master_fc2_scale_ptrs[local_master_idx]);
}

static __device__ __forceinline__ uint8_t* weight_sync_remote_shard_addr(const TaskBuildConfig& config,
                                                                         void* const* __restrict__ remote_weight_ptrs,
                                                                         void* const* __restrict__ remote_scale_ptrs,
                                                                         const int replica_nvl_rank,
                                                                         const int replica_local_offset,
                                                                         const int shard_idx) {
    const bool scale_shard = shard_idx >= 2;
    void* const* remote_ptrs = scale_shard ? remote_scale_ptrs : remote_weight_ptrs;
    uint8_t* remote_buf = reinterpret_cast<uint8_t*>(remote_ptrs[replica_nvl_rank]);
    return remote_buf +
        static_cast<size_t>(replica_local_offset) * weight_sync_expert_buffer_bytes(config, scale_shard) +
        weight_sync_shard_offset_bytes(config, shard_idx);
}

static __device__ __forceinline__ uint8_t* weight_sync_local_replica_shard_addr(
    const TaskBuildConfig& config,
    uint8_t* __restrict__ local_replica_weight_buffer,
    uint8_t* __restrict__ local_replica_weight_scale_buffer,
    const int replica_local_offset,
    const int shard_idx) {
    const bool scale_shard = shard_idx >= 2;
    uint8_t* local_buf = scale_shard ? local_replica_weight_scale_buffer : local_replica_weight_buffer;
    return local_buf +
        static_cast<size_t>(replica_local_offset) * weight_sync_expert_buffer_bytes(config, scale_shard) +
        weight_sync_shard_offset_bytes(config, shard_idx);
}

static __host__ __device__ __forceinline__ int choose_weight_sync_relay_count(const int num_replicas,
                                                                              const TaskBuildConfig& config) {
    if (num_replicas <= 1) {
        return 0;
    }

    int relay_count = floor_sqrt_int(num_replicas);
    if (relay_count < 1) {
        relay_count = 1;
    }
    if (config.weight_sync_relay_max_relays > 0 && relay_count > config.weight_sync_relay_max_relays) {
        relay_count = config.weight_sync_relay_max_relays;
    }
    if (relay_count >= num_replicas) {
        relay_count = num_replicas - 1;
    }
    return relay_count;
}

static __host__ __device__ __forceinline__ int max_weight_sync_relay_chunks_per_shard(const TaskBuildConfig& config) {
    size_t max_bytes = 0;
    const int num_shards = weight_sync_num_shards(config);
    for (int shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
        const size_t shard_bytes = weight_sync_shard_num_bytes(config, shard_idx);
        max_bytes = max_bytes > shard_bytes ? max_bytes : shard_bytes;
    }
    return weight_sync_num_chunks(max_bytes);
}

static __host__ __device__ __forceinline__ int weight_sync_ready_flag_slot(const TaskBuildConfig& config,
                                                                           const int local_replica_offset,
                                                                           const int shard_idx,
                                                                           const int chunk_idx) {
    return ((local_replica_offset * weight_sync_num_shards(config) + shard_idx) *
            max_weight_sync_relay_chunks_per_shard(config)) +
        chunk_idx;
}

static __host__ __device__ __forceinline__ bool should_use_weight_sync_relay(const int num_replicas,
                                                                             const TaskBuildConfig& config) {
    if (config.weight_sync_plan_mode == static_cast<int>(WeightSyncPlanMode::kDirect)) {
        return false;
    }

    const int relay_count = choose_weight_sync_relay_count(num_replicas, config);
    if (relay_count <= 0) {
        return false;
    }

    if (config.weight_sync_plan_mode == static_cast<int>(WeightSyncPlanMode::kForceRelay)) {
        return true;
    }

    if (num_replicas < config.weight_sync_relay_min_replicas) {
        return false;
    }

    const int relay_sender_fanout = relay_count;
    const int relay_child_fanout = ceil_div(num_replicas - relay_count, relay_count);
    const int relay_critical_fanout =
        relay_sender_fanout > relay_child_fanout ? relay_sender_fanout : relay_child_fanout;
    return (num_replicas - relay_critical_fanout) >= config.weight_sync_relay_min_fanout_gain;
}

// ---------------------------------------------------------------------------
// Weight Sync Task Build
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void init_weight_sync_task(WeightSyncTask& task) {
    task.master_local_addr = nullptr;
    task.num_replicas = 0;
    task.num_bytes = 0;
    task.wait_ready_slot = -1;
    task.num_ready_signals = 0;
}

__global__ __launch_bounds__(32) void build_weight_sync_task_lists_kernel(
    const TaskBuildConfig* __restrict__ config,
    const int32_t* __restrict__ p2l,
    const int32_t* __restrict__ l2p,
    const int32_t* __restrict__ lcnts,
    void* const* __restrict__ remote_weight_ptrs,
    void* const* __restrict__ remote_weight_scale_ptrs,
    const int64_t* __restrict__ local_master_fc1_ptrs,
    const int64_t* __restrict__ local_master_fc2_ptrs,
    const int64_t* __restrict__ local_master_fc1_scale_ptrs,
    const int64_t* __restrict__ local_master_fc2_scale_ptrs,
    uint8_t* __restrict__ local_replica_weight_buffer,
    uint8_t* __restrict__ local_replica_weight_scale_buffer,
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
    const int max_rep_dim = config->max_replicas_dim;
    const int num_weight_shards = weight_sync_num_shards(*config);
    const int64_t weight_bytes_per_expert = static_cast<int64_t>(weight_sync_expert_total_bytes(*config));

    int64_t sender_load_bytes[kMaxNvlDomainSize] = {0};
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
                    for (int shard_idx = 0; shard_idx < num_weight_shards; ++shard_idx) {
                        const size_t shard_bytes = weight_sync_shard_num_bytes(*config, shard_idx);
                        if (shard_bytes == 0) {
                            continue;
                        }
                        WeightSyncTask& task = stage1_tasks[stage1_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr = weight_sync_local_master_addr(local_master_fc1_ptrs,
                                                                               local_master_fc2_ptrs,
                                                                               local_master_fc1_scale_ptrs,
                                                                               local_master_fc2_scale_ptrs,
                                                                               local_master_idx,
                                                                               shard_idx);
                        task.num_replicas = num_replicas;
                        task.num_bytes = shard_bytes;

                        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                            const int replica_phy = l2p[logical_expert * max_rep_dim + replica_idx + 1];
                            const int replica_rank = replica_phy / num_local_physical;
                            const int replica_nvl_rank = replica_rank % num_nvl_ranks;
                            const int replica_local_offset = replica_phy % num_local_physical - num_local_master;

                            task.replica_remote_addrs[replica_idx] =
                                weight_sync_remote_shard_addr(*config,
                                                              remote_weight_ptrs,
                                                              remote_weight_scale_ptrs,
                                                              replica_nvl_rank,
                                                              replica_local_offset,
                                                              shard_idx);
                        }
                    }
                }
                continue;
            }

            const int relay_count = choose_weight_sync_relay_count(num_replicas, *config);
            if (relay_count <= 0) {
                continue;
            }

            bool replica_selected[kMaxNvlDomainSize - 1] = {false};
            int relay_replica_indices[kMaxNvlDomainSize - 1] = {0};
            int relay_global_ranks[kMaxNvlDomainSize - 1] = {0};
            int relay_nvl_ranks[kMaxNvlDomainSize - 1] = {0};
            int relay_local_offsets[kMaxNvlDomainSize - 1] = {0};
            int relay_child_counts[kMaxNvlDomainSize - 1] = {0};
            int leaf_owner_relay[kMaxNvlDomainSize - 1];
            for (int replica_idx = 0; replica_idx < kMaxNvlDomainSize - 1; ++replica_idx) {
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

            int leaf_replica_indices[kMaxNvlDomainSize - 1] = {0};
            int leaf_count = 0;
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                if (!replica_selected[replica_idx]) {
                    leaf_replica_indices[leaf_count++] = replica_idx;
                }
            }

            int64_t projected_relay_loads[kMaxNvlDomainSize - 1] = {0};
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
                for (int shard_idx = 0; shard_idx < num_weight_shards; ++shard_idx) {
                    const size_t shard_bytes = weight_sync_shard_num_bytes(*config, shard_idx);
                    if (shard_bytes == 0) {
                        continue;
                    }
                    const int num_chunks = weight_sync_num_chunks(shard_bytes);
                    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        WeightSyncTask& task = stage1_tasks[stage1_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr = weight_sync_local_master_addr(local_master_fc1_ptrs,
                                                                               local_master_fc2_ptrs,
                                                                               local_master_fc1_scale_ptrs,
                                                                               local_master_fc2_scale_ptrs,
                                                                               local_master_idx,
                                                                               shard_idx) +
                            weight_sync_chunk_offset_bytes(chunk_idx);
                        task.num_replicas = relay_count;
                        task.num_bytes = weight_sync_chunk_num_bytes(shard_bytes, chunk_idx);
                        task.num_ready_signals = relay_count;

                        for (int relay_idx = 0; relay_idx < relay_count; ++relay_idx) {
                            uint8_t* remote_expert_base = weight_sync_remote_shard_addr(*config,
                                                                                        remote_weight_ptrs,
                                                                                        remote_weight_scale_ptrs,
                                                                                        relay_nvl_ranks[relay_idx],
                                                                                        relay_local_offsets[relay_idx],
                                                                                        shard_idx);
                            task.replica_remote_addrs[relay_idx] =
                                remote_expert_base + weight_sync_chunk_offset_bytes(chunk_idx);
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

                for (int shard_idx = 0; shard_idx < num_weight_shards; ++shard_idx) {
                    const size_t shard_bytes = weight_sync_shard_num_bytes(*config, shard_idx);
                    if (shard_bytes == 0) {
                        continue;
                    }
                    uint8_t* local_relay_base = weight_sync_local_replica_shard_addr(*config,
                                                                                     local_replica_weight_buffer,
                                                                                     local_replica_weight_scale_buffer,
                                                                                     relay_local_offsets[relay_idx],
                                                                                     shard_idx);
                    const int num_chunks = weight_sync_num_chunks(shard_bytes);
                    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                        WeightSyncTask& task = stage2_tasks[stage2_num_tasks++];
                        init_weight_sync_task(task);
                        task.master_local_addr = local_relay_base + weight_sync_chunk_offset_bytes(chunk_idx);
                        task.num_replicas = relay_child_counts[relay_idx];
                        task.num_bytes = weight_sync_chunk_num_bytes(shard_bytes, chunk_idx);
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

                            uint8_t* remote_expert_base = weight_sync_remote_shard_addr(*config,
                                                                                        remote_weight_ptrs,
                                                                                        remote_weight_scale_ptrs,
                                                                                        replica_nvl_rank,
                                                                                        replica_local_offset,
                                                                                        shard_idx);
                            task.replica_remote_addrs[child_idx++] =
                                remote_expert_base + weight_sync_chunk_offset_bytes(chunk_idx);
                        }
                    }
                }
            }
        }
    }

    stage1_tile_offsets[0] = 0;
    for (int task_idx = 0; task_idx < stage1_num_tasks; ++task_idx) {
        const int num_tiles = weight_sync_num_tiles(stage1_tasks[task_idx].num_bytes);
        stage1_tile_offsets[task_idx + 1] = stage1_tile_offsets[task_idx] + num_tiles;
        stage1_remaining_tiles[task_idx] = num_tiles;
    }
    stage1_task_metadata[0] = stage1_num_tasks;
    stage1_task_metadata[1] = stage1_num_tasks > 0 ? stage1_tile_offsets[stage1_num_tasks] : 0;

    stage2_tile_offsets[0] = 0;
    for (int task_idx = 0; task_idx < stage2_num_tasks; ++task_idx) {
        const int num_tiles = weight_sync_num_tiles(stage2_tasks[task_idx].num_bytes);
        stage2_tile_offsets[task_idx + 1] = stage2_tile_offsets[task_idx] + num_tiles;
    }
    stage2_task_metadata[0] = stage2_num_tasks;
    stage2_task_metadata[1] = stage2_num_tasks > 0 ? stage2_tile_offsets[stage2_num_tasks] : 0;
}

void build_weight_sync_task_lists(const TaskBuildConfig* config,
                                  const int32_t* physical_to_logical_map,
                                  const int32_t* logical_to_physical_map,
                                  const int32_t* logical_replica_counts,
                                  void* const* remote_weight_ptrs,
                                  void* const* remote_weight_scale_ptrs,
                                  const int64_t* local_master_fc1_ptrs,
                                  const int64_t* local_master_fc2_ptrs,
                                  const int64_t* local_master_fc1_scale_ptrs,
                                  const int64_t* local_master_fc2_scale_ptrs,
                                  uint8_t* local_replica_weight_buffer,
                                  uint8_t* local_replica_weight_scale_buffer,
                                  WeightSyncTask* stage1_tasks,
                                  int* stage1_task_tile_offsets,
                                  int* stage1_task_metadata,
                                  int* stage1_task_remaining_tiles,
                                  int* stage1_global_tile_counter,
                                  WeightSyncTask* stage2_tasks,
                                  int* stage2_task_tile_offsets,
                                  int* stage2_task_metadata,
                                  int* stage2_global_tile_counter,
                                  cudaStream_t stream) {
    const auto launch_config = make_launch_config(dim3(1), dim3(32), stream);
    launch_kernel(build_weight_sync_task_lists_kernel,
                  launch_config,
                  config,
                  physical_to_logical_map,
                  logical_to_physical_map,
                  logical_replica_counts,
                  remote_weight_ptrs,
                  remote_weight_scale_ptrs,
                  local_master_fc1_ptrs,
                  local_master_fc2_ptrs,
                  local_master_fc1_scale_ptrs,
                  local_master_fc2_scale_ptrs,
                  local_replica_weight_buffer,
                  local_replica_weight_scale_buffer,
                  stage1_tasks,
                  stage1_task_tile_offsets,
                  stage1_task_metadata,
                  stage1_task_remaining_tiles,
                  stage2_tasks,
                  stage2_task_tile_offsets,
                  stage2_task_metadata);

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(stage1_global_tile_counter, 0, sizeof(int), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(stage2_global_tile_counter, 0, sizeof(int), stream));
}

// ============================================================================
// Weight Sync Kernel: Synchronize weights across the current task plan
// ============================================================================
//
// Design:
// - Each task is a single-source, multi-destination transfer for one weight shard.
// - The source can be either:
//     1. a local master weight (direct fan-out / stage 1), or
//     2. a local relay replica buffer (relay forwarding / stage 2).
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
// This approach loads SMEM only once per tile, regardless of destination count.
// The higher-level planner may choose either a flat fan-out or a staged relay
// topology; both are executed by the same persistent kernel.
//
// Tile-level parallelism with persistent kernel:
// - Each CTA grabs tiles via atomic counter
// - Multiple CTAs can process different tiles of the same task concurrently
// ============================================================================

// Structure to help with tile-to-task mapping
struct WeightSyncTileInfo {
    int task_idx;
    int tile_idx_in_task;
    size_t byte_offset;
    int num_bytes;
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
    info.byte_offset = static_cast<size_t>(info.tile_idx_in_task) * kWeightSyncTileSizeBytes;

    const size_t task_bytes = tasks[info.task_idx].num_bytes;
    const size_t remaining = task_bytes - info.byte_offset;
    info.num_bytes = min(static_cast<size_t>(kWeightSyncTileSizeBytes), remaining);

    return info;
}

__device__ __forceinline__ void finalize_completed_weight_sync_task(const WeightSyncTask* tasks,
                                                                    const int completed_task_idx,
                                                                    int* task_remaining_tiles,
                                                                    uint64_t* local_ready_flags,
                                                                    uint64_t* const* remote_ready_flag_ptrs,
                                                                    const uint64_t current_epoch) {
    if (completed_task_idx < 0 || task_remaining_tiles == nullptr || remote_ready_flag_ptrs == nullptr) {
        return;
    }

    const WeightSyncTask& task = tasks[completed_task_idx];
    if (task.num_ready_signals <= 0) {
        return;
    }

    const int old_remaining_tiles = atomicSub(task_remaining_tiles + completed_task_idx, 1);
    if (old_remaining_tiles != 1) {
        return;
    }

    for (int signal_idx = 0; signal_idx < task.num_ready_signals; ++signal_idx) {
        uint64_t* remote_ready_flag =
            remote_ready_flag_ptrs[task.ready_signal_nvl_ranks[signal_idx]] + task.ready_signal_slots[signal_idx];
        *remote_ready_flag = current_epoch;
    }
    __threadfence_system();
}

__device__ __forceinline__ void wait_for_weight_sync_task_ready(const WeightSyncTask& task,
                                                                uint64_t* local_ready_flags,
                                                                const uint64_t current_epoch) {
    if (task.wait_ready_slot < 0 || local_ready_flags == nullptr) {
        return;
    }

    unsigned long long* ready_flag = reinterpret_cast<unsigned long long*>(local_ready_flags + task.wait_ready_slot);
    while (atomicAdd(ready_flag, 0ULL) < current_epoch) {
    }
    __threadfence_system();
}

__device__ __forceinline__ bool can_use_tma_for_weight_sync_tile(const WeightSyncTask& task,
                                                                 const WeightSyncTileInfo& tile) {
    std::uintptr_t alignment_bits = reinterpret_cast<std::uintptr_t>(task.master_local_addr + tile.byte_offset);
    for (int replica_idx = 0; replica_idx < task.num_replicas; ++replica_idx) {
        alignment_bits |= reinterpret_cast<std::uintptr_t>(task.replica_remote_addrs[replica_idx] + tile.byte_offset);
    }
    return (alignment_bits & (sizeof(uint4) - 1)) == 0;
}

__device__ __forceinline__ void copy_weight_sync_tile_threaded(const uint8_t* __restrict__ src,
                                                               uint8_t* __restrict__ dst,
                                                               const int num_bytes) {
    const std::uintptr_t alignment_bits = reinterpret_cast<std::uintptr_t>(src) | reinterpret_cast<std::uintptr_t>(dst);

    if ((alignment_bits & (sizeof(uint4) - 1)) == 0) {
        const int num_vecs = num_bytes / static_cast<int>(sizeof(uint4));
        const uint4* __restrict__ src_vec = reinterpret_cast<const uint4*>(src);
        uint4* __restrict__ dst_vec = reinterpret_cast<uint4*>(dst);
        for (int vec_idx = threadIdx.x; vec_idx < num_vecs; vec_idx += blockDim.x) {
            const uint4 value = src_vec[vec_idx];
            ptx::st_global_v4_u32_streaming(dst_vec + vec_idx, value.x, value.y, value.z, value.w);
        }

        const int tail_start = num_vecs * static_cast<int>(sizeof(uint4));
        for (int byte_idx = tail_start + threadIdx.x; byte_idx < num_bytes; byte_idx += blockDim.x) {
            dst[byte_idx] = src[byte_idx];
        }
        return;
    }

    for (int byte_idx = threadIdx.x; byte_idx < num_bytes; byte_idx += blockDim.x) {
        dst[byte_idx] = src[byte_idx];
    }
}

__global__ __launch_bounds__(kWeightSyncThreadsPerBlock) void weight_sync_thread_copy_kernel(
    const WeightSyncTask* weight_sync_tasks,
    const int* task_tile_offsets,
    const int* task_metadata,
    int* global_tile_counter,
    int* task_remaining_tiles,
    uint64_t* local_ready_flags,
    uint64_t* const* remote_ready_flag_ptrs,
    uint64_t current_epoch) {
    const bool is_leader = (threadIdx.x == 0);

    __shared__ int total_tasks;
    __shared__ int total_tiles;
    __shared__ int tile_idx;
    if (is_leader) {
        total_tasks = task_metadata[0];
        total_tiles = task_metadata[1];
    }
    __syncthreads();

    if (total_tasks == 0) {
        return;
    }

    while (true) {
        if (is_leader) {
            tile_idx = atomicAdd(global_tile_counter, 1);
        }
        __syncthreads();

        if (tile_idx >= total_tiles) {
            break;
        }

        const WeightSyncTileInfo tile =
            get_weight_sync_tile_info(weight_sync_tasks, task_tile_offsets, total_tasks, tile_idx);
        const WeightSyncTask& task = weight_sync_tasks[tile.task_idx];

        if (is_leader) {
            wait_for_weight_sync_task_ready(task, local_ready_flags, current_epoch);
        }
        __syncthreads();

        const uint8_t* src = task.master_local_addr + tile.byte_offset;
        for (int replica_idx = 0; replica_idx < task.num_replicas; ++replica_idx) {
            uint8_t* dst = task.replica_remote_addrs[replica_idx] + tile.byte_offset;
            copy_weight_sync_tile_threaded(src, dst, tile.num_bytes);
        }
        __syncthreads();

        if (is_leader && task_remaining_tiles != nullptr && task.num_ready_signals > 0) {
            __threadfence_system();
            finalize_completed_weight_sync_task(weight_sync_tasks,
                                                tile.task_idx,
                                                task_remaining_tiles,
                                                local_ready_flags,
                                                remote_ready_flag_ptrs,
                                                current_epoch);
        }
        __syncthreads();
    }
}

// Weight sync kernel with double buffering for true pipelining
// Pipeline: TMA_Load[N+1] overlaps with TMA_Store[N]
// This achieves true overlap of local HBM reads and remote NVLINK writes.
//
// task_metadata: device pointer to [total_tasks, total_tiles] (set by CPU or GPU task build)
__global__ __launch_bounds__(kWeightSyncThreadsPerBlock) void weight_sync_kernel(
    const WeightSyncTask* weight_sync_tasks,
    const int* task_tile_offsets,
    const int* task_metadata,
    int* global_tile_counter,
    int* task_remaining_tiles,
    uint64_t* local_ready_flags,
    uint64_t* const* remote_ready_flag_ptrs,
    uint64_t current_epoch) {
    // Double-buffered shared memory
    extern __shared__ uint8_t smem_base[];
    uint8_t* smem[2] = {smem_base, smem_base + kWeightSyncTileSizeBytes};

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
    int pending_task_idx = -1;

    // Main pipeline loop
    while (tile_indices[cur_buf] < total_tiles) {
        int my_tile_idx = tile_indices[cur_buf];

        // Get current tile info
        WeightSyncTileInfo tile =
            get_weight_sync_tile_info(weight_sync_tasks, task_tile_offsets, total_tasks, my_tile_idx);
        const WeightSyncTask& task = weight_sync_tasks[tile.task_idx];

        if (is_leader) {
            wait_for_weight_sync_task_ready(task, local_ready_flags, current_epoch);
        }
        __syncthreads();

        const bool use_tma = can_use_tma_for_weight_sync_tile(task, tile);
        const int tma_bytes = (tile.num_bytes / static_cast<int>(sizeof(uint4))) * static_cast<int>(sizeof(uint4));
        const int tail_bytes = tile.num_bytes - tma_bytes;
        const int next_buf = 1 - cur_buf;

        if (!use_tma || tma_bytes == 0) {
            if (has_pending_store) {
                if (is_leader) {
                    ptx::tma_store_wait<0>();
                    __threadfence_system();
                    finalize_completed_weight_sync_task(weight_sync_tasks,
                                                        pending_task_idx,
                                                        task_remaining_tiles,
                                                        local_ready_flags,
                                                        remote_ready_flag_ptrs,
                                                        current_epoch);
                    has_pending_store = false;
                }
                __syncthreads();
            }

            const uint8_t* src = task.master_local_addr + tile.byte_offset;
            for (int replica_idx = 0; replica_idx < task.num_replicas; ++replica_idx) {
                uint8_t* dst = task.replica_remote_addrs[replica_idx] + tile.byte_offset;
                copy_weight_sync_tile_threaded(src, dst, tile.num_bytes);
            }
            __syncthreads();

            if (is_leader) {
                __threadfence_system();
                finalize_completed_weight_sync_task(weight_sync_tasks,
                                                    tile.task_idx,
                                                    task_remaining_tiles,
                                                    local_ready_flags,
                                                    remote_ready_flag_ptrs,
                                                    current_epoch);
                tile_indices[cur_buf] = atomicAdd(global_tile_counter, 1);
            }
            __syncthreads();
            continue;
        }

        const size_t bytes = static_cast<size_t>(tma_bytes);

        // Issue TMA Load for the 16-byte aligned bulk of the current tile.
        if (is_leader) {
            ptx::mbarrier_arrive_and_set_tx(&mbarriers[cur_buf], bytes);
            ptx::tma_load_1d(smem[cur_buf],
                             task.master_local_addr + tile.byte_offset,
                             &mbarriers[cur_buf],
                             bytes,
                             ptx::TMACacheHint::kEvictNormal);
        }

        // Prefetch next tile index while TMA Load is in flight
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
                __threadfence_system();
                finalize_completed_weight_sync_task(weight_sync_tasks,
                                                    pending_task_idx,
                                                    task_remaining_tiles,
                                                    local_ready_flags,
                                                    remote_ready_flag_ptrs,
                                                    current_epoch);
            }
            __syncthreads();
        }

        // Fence and issue TMA stores for current tile.
        if (is_leader) {
            ptx::tma_store_fence();
            for (int r = 0; r < task.num_replicas; ++r) {
                uint8_t* replica_addr = task.replica_remote_addrs[r] + tile.byte_offset;
                ptx::tma_store_1d(replica_addr, smem[cur_buf], bytes, ptx::TMACacheHint::kEvictNormal);
            }
            ptx::tma_store_commit();
            has_pending_store = true;
            pending_task_idx = tile.task_idx;
        }
        if (tail_bytes > 0) {
            const uint8_t* tail_src = task.master_local_addr + tile.byte_offset + tma_bytes;
            for (int replica_idx = 0; replica_idx < task.num_replicas; ++replica_idx) {
                uint8_t* tail_dst = task.replica_remote_addrs[replica_idx] + tile.byte_offset + tma_bytes;
                copy_weight_sync_tile_threaded(tail_src, tail_dst, tail_bytes);
            }
        }
        __syncthreads();

        // Switch buffers
        cur_buf = next_buf;
    }

    // Wait for any remaining pending stores
    if (has_pending_store) {
        if (is_leader) {
            ptx::tma_store_wait<0>();
            __threadfence_system();
            finalize_completed_weight_sync_task(weight_sync_tasks,
                                                pending_task_idx,
                                                task_remaining_tiles,
                                                local_ready_flags,
                                                remote_ready_flag_ptrs,
                                                current_epoch);
        }
        __syncthreads();
    }

    // Cleanup mbarriers
    __syncthreads();
    if (is_leader) {
        for (int i = 0; i < 2; i++) {
            ptx::mbarrier_invalidate(&mbarriers[i]);
        }
    }
}

void run_weight_sync(WeightSyncTask* tasks,
                     int* task_tile_offsets,
                     int* task_metadata,
                     int* global_tile_counter,
                     int* task_remaining_tiles,
                     uint64_t* local_ready_flags,
                     uint64_t* const* remote_ready_flag_ptrs,
                     uint64_t current_epoch,
                     cudaStream_t stream,
                     int num_device_sms,
                     int num_nvl_ranks,
                     int max_possible_tiles,
                     int cta_multiplier) {
    if (num_nvl_ranks <= kWeightSyncThreadCopyMaxNvlRanks && task_remaining_tiles != nullptr) {
        const int thread_copy_cta_multiplier =
            cta_multiplier > kWeightSyncThreadCopyCtaMultiplier ? cta_multiplier : kWeightSyncThreadCopyCtaMultiplier;
        const int num_ctas = clamp_num_ctas(num_device_sms * thread_copy_cta_multiplier, max_possible_tiles);
        const auto config = make_launch_config(dim3(num_ctas), dim3(kWeightSyncThreadsPerBlock), stream, 0);

        launch_kernel(weight_sync_thread_copy_kernel,
                      config,
                      tasks,
                      task_tile_offsets,
                      task_metadata,
                      global_tile_counter,
                      task_remaining_tiles,
                      local_ready_flags,
                      remote_ready_flag_ptrs,
                      current_epoch);
        return;
    }

    // Use conservative upper bound for grid size; persistent kernel handles over-launch
    const int num_ctas = clamp_num_ctas(num_device_sms * cta_multiplier, max_possible_tiles);
    const auto config = make_launch_config(
        dim3(num_ctas), dim3(kWeightSyncThreadsPerBlock), stream, kWeightSyncTileSizeBytes * kWeightSyncPipelineStages);

    launch_kernel(weight_sync_kernel,
                  config,
                  tasks,
                  task_tile_offsets,
                  task_metadata,
                  global_tile_counter,
                  task_remaining_tiles,
                  local_ready_flags,
                  remote_ready_flag_ptrs,
                  current_epoch);
}

}  // namespace ultra_ep::kernels
