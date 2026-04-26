#include "../api.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

#include "../../utils/exception.cuh"

namespace ultra_ep::kernels::legacy {

namespace {

struct ReplicaEntry {
    int logical_id;
    double load_per_replica;
};

void solve_on_host(const int32_t* expert_loads,
                   int32_t* physical_to_logical_map,
                   int32_t* logical_to_physical_map,
                   int32_t* logical_replica_counts,
                   int num_global_logical_experts,
                   int num_ranks,
                   int num_local_master_experts,
                   int num_local_redundant_experts,
                   int num_nvl_ranks,
                   int max_replicas_dim,
                   float balance_threshold) {
    const int num_local_physical_experts = num_local_master_experts + num_local_redundant_experts;
    const int num_global_physical_experts = num_local_physical_experts * num_ranks;
    const int num_nvl_domains = num_ranks / num_nvl_ranks;
    const int num_logical_per_nvl = num_local_master_experts * num_nvl_ranks;
    const int num_redundant_per_nvl = num_local_redundant_experts * num_nvl_ranks;
    const int max_extra_replicas = num_nvl_ranks - 1;

    std::memset(physical_to_logical_map, 0xFF, static_cast<size_t>(num_global_physical_experts) * sizeof(int32_t));
    std::memset(logical_to_physical_map,
                0xFF,
                static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t));
    std::memset(logical_replica_counts, 0, static_cast<size_t>(num_global_logical_experts) * sizeof(int32_t));

    for (int logical_id = 0; logical_id < num_global_logical_experts; ++logical_id) {
        const int rank = logical_id / num_local_master_experts;
        const int local_idx = logical_id % num_local_master_experts;
        const int physical_id = rank * num_local_physical_experts + local_idx;
        physical_to_logical_map[physical_id] = logical_id;
        logical_to_physical_map[logical_id * max_replicas_dim] = physical_id;
        logical_replica_counts[logical_id] = 1;
    }

    if (num_local_redundant_experts == 0 || num_nvl_ranks <= 1) {
        return;
    }

    std::vector<ReplicaEntry> replicas;
    replicas.reserve(num_redundant_per_nvl);
    std::vector<double> rank_loads(num_nvl_ranks);
    std::vector<int> rank_slots_used(num_nvl_ranks);
    std::vector<uint8_t> expert_on_rank(static_cast<size_t>(num_nvl_ranks) * num_logical_per_nvl);

    for (int domain_idx = 0; domain_idx < num_nvl_domains; ++domain_idx) {
        const int domain_start_rank = domain_idx * num_nvl_ranks;
        const int domain_start_logical = domain_start_rank * num_local_master_experts;

        double total_load = 0.0;
        for (int offset = 0; offset < num_logical_per_nvl; ++offset) {
            total_load += expert_loads[domain_start_logical + offset];
        }
        const double avg_per_slot =
            total_load > 0.0 ? total_load / (num_nvl_ranks * num_local_master_experts) : 0.0;

        for (int slot = 0; slot < num_redundant_per_nvl; ++slot) {
            int best_logical = -1;
            double best_score = -1.0;
            for (int offset = 0; offset < num_logical_per_nvl; ++offset) {
                const int logical_id = domain_start_logical + offset;
                if (logical_replica_counts[logical_id] - 1 >= max_extra_replicas) {
                    continue;
                }
                const double score =
                    static_cast<double>(expert_loads[logical_id]) / logical_replica_counts[logical_id];
                if (score > best_score || (score == best_score && (best_logical < 0 || logical_id < best_logical))) {
                    best_score = score;
                    best_logical = logical_id;
                }
            }

            if (balance_threshold > 1.0f && avg_per_slot > 0.0 && best_score <= avg_per_slot * balance_threshold) {
                break;
            }
            if (best_logical < 0) {
                break;
            }
            logical_replica_counts[best_logical] += 1;
        }

        replicas.clear();
        for (int offset = 0; offset < num_logical_per_nvl; ++offset) {
            const int logical_id = domain_start_logical + offset;
            const int num_extra_replicas = logical_replica_counts[logical_id] - 1;
            if (num_extra_replicas <= 0) {
                continue;
            }
            const double load_per_replica =
                static_cast<double>(expert_loads[logical_id]) / logical_replica_counts[logical_id];
            for (int replica_idx = 0; replica_idx < num_extra_replicas; ++replica_idx) {
                replicas.push_back({logical_id, load_per_replica});
            }
        }

        std::sort(replicas.begin(), replicas.end(), [](const ReplicaEntry& lhs, const ReplicaEntry& rhs) {
            if (lhs.load_per_replica != rhs.load_per_replica) {
                return lhs.load_per_replica > rhs.load_per_replica;
            }
            return lhs.logical_id < rhs.logical_id;
        });

        std::fill(rank_loads.begin(), rank_loads.end(), 0.0);
        std::fill(rank_slots_used.begin(), rank_slots_used.end(), 0);
        std::fill(expert_on_rank.begin(), expert_on_rank.end(), 0);

        for (int rank_offset = 0; rank_offset < num_nvl_ranks; ++rank_offset) {
            for (int local_master_idx = 0; local_master_idx < num_local_master_experts; ++local_master_idx) {
                const int logical_id = (domain_start_rank + rank_offset) * num_local_master_experts + local_master_idx;
                rank_loads[rank_offset] +=
                    static_cast<double>(expert_loads[logical_id]) / logical_replica_counts[logical_id];
                const int bitmap_base = rank_offset * num_logical_per_nvl;
                const int master_offset = rank_offset * num_local_master_experts + local_master_idx;
                expert_on_rank[bitmap_base + master_offset] = 1;
            }
        }

        for (const ReplicaEntry& replica : replicas) {
            const int local_logical_id = replica.logical_id - domain_start_logical;
            int best_rank = -1;
            double best_rank_load = 1e18;

            for (int rank_offset = 0; rank_offset < num_nvl_ranks; ++rank_offset) {
                if (rank_slots_used[rank_offset] >= num_local_redundant_experts) {
                    continue;
                }
                if (expert_on_rank[rank_offset * num_logical_per_nvl + local_logical_id]) {
                    continue;
                }
                if (rank_loads[rank_offset] < best_rank_load ||
                    (rank_loads[rank_offset] == best_rank_load && (best_rank < 0 || rank_offset < best_rank))) {
                    best_rank_load = rank_loads[rank_offset];
                    best_rank = rank_offset;
                }
            }

            if (best_rank < 0) {
                logical_replica_counts[replica.logical_id] -= 1;
                continue;
            }

            const int global_rank = domain_start_rank + best_rank;
            const int physical_id =
                global_rank * num_local_physical_experts + num_local_master_experts + rank_slots_used[best_rank];
            physical_to_logical_map[physical_id] = replica.logical_id;

            int32_t* logical_row = logical_to_physical_map + replica.logical_id * max_replicas_dim;
            for (int replica_slot = 1; replica_slot < max_replicas_dim; ++replica_slot) {
                if (logical_row[replica_slot] == -1) {
                    logical_row[replica_slot] = physical_id;
                    break;
                }
            }

            rank_loads[best_rank] += replica.load_per_replica;
            rank_slots_used[best_rank] += 1;
            expert_on_rank[best_rank * num_logical_per_nvl + local_logical_id] = 1;
        }
    }
}

}  // namespace

void solve_placement(const int32_t* expert_loads,
                   const int32_t* expert_loads_per_rank,
                   int32_t* physical_to_logical_map,
                   int32_t* logical_to_physical_map,
                   int32_t* logical_replica_counts,
                   int32_t* logical_instance_quota,
                   int32_t* logical_instance_quota_prefix,
                   int32_t* rank_quota_prefix,
                   cudaStream_t stream,
                   int num_global_logical_experts,
                   int num_ranks,
                   int num_local_master_experts,
                   int num_local_redundant_experts,
                   int num_nvl_ranks,
                   int max_replicas_dim,
                   float balance_threshold,
                   int32_t min_tokens_per_replica,
                   bool allow_zero_master_quota,
                   bool locality_aware,
                   float oracle_eps,
                   int kernel_stage) {
    (void)expert_loads_per_rank;
    (void)min_tokens_per_replica;
    (void)allow_zero_master_quota;
    (void)locality_aware;
    (void)oracle_eps;
    (void)kernel_stage;

    const int num_local_physical_experts = num_local_master_experts + num_local_redundant_experts;
    const int num_global_physical_experts = num_local_physical_experts * num_ranks;
    const size_t loads_bytes = static_cast<size_t>(num_global_logical_experts) * sizeof(int32_t);
    const size_t p2l_bytes = static_cast<size_t>(num_global_physical_experts) * sizeof(int32_t);
    const size_t l2p_bytes =
        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t);
    const size_t lcnts_bytes = static_cast<size_t>(num_global_logical_experts) * sizeof(int32_t);
    const size_t quota_bytes =
        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t);

    std::vector<int32_t> host_loads(num_global_logical_experts);
    std::vector<int32_t> host_p2l(num_global_physical_experts);
    std::vector<int32_t> host_l2p(static_cast<size_t>(num_global_logical_experts) * max_replicas_dim);
    std::vector<int32_t> host_lcnts(num_global_logical_experts);

    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        host_loads.data(), expert_loads, loads_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    solve_on_host(host_loads.data(),
                  host_p2l.data(),
                  host_l2p.data(),
                  host_lcnts.data(),
                  num_global_logical_experts,
                  num_ranks,
                  num_local_master_experts,
                  num_local_redundant_experts,
                  num_nvl_ranks,
                  max_replicas_dim,
                  balance_threshold);

    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        physical_to_logical_map, host_p2l.data(), p2l_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        logical_to_physical_map, host_l2p.data(), l2p_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        logical_replica_counts, host_lcnts.data(), lcnts_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(logical_instance_quota, 0, quota_bytes, stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(logical_instance_quota_prefix, 0, quota_bytes, stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(rank_quota_prefix, 0, quota_bytes, stream));
}

}  // namespace ultra_ep::kernels::legacy
