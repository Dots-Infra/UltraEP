#include <algorithm>
#include <cassert>
#include <cstring>

#include "api.hpp"

namespace ultra_ep::solver {

// ============================================================================
// Constructor – pre-allocate all scratch buffers once
// ============================================================================
PlacementSolver::PlacementSolver(int num_global_logical_experts,
                                 int num_ranks,
                                 int num_local_master_experts,
                                 int num_local_redundant_experts,
                                 int num_nvl_ranks,
                                 int max_replicas_dim)
    : num_global_logical_experts_(num_global_logical_experts),
      num_ranks_(num_ranks),
      num_local_master_(num_local_master_experts),
      num_local_redundant_(num_local_redundant_experts),
      num_nvl_ranks_(num_nvl_ranks),
      max_replicas_dim_(max_replicas_dim),
      num_local_physical_(num_local_master_experts + num_local_redundant_experts),
      num_global_physical_((num_local_master_experts + num_local_redundant_experts) * num_ranks),
      num_nvl_domains_(num_ranks / num_nvl_ranks),
      num_logical_per_nvl_(num_local_master_experts * num_nvl_ranks),
      num_redundant_per_nvl_(num_local_redundant_experts * num_nvl_ranks),
      // Each logical expert can appear at most once per rank.
      // Master takes 1 rank, so replicas can go to at most (num_nvl_ranks - 1) other ranks.
      max_extra_replicas_(num_nvl_ranks - 1) {
    // Pre-allocate scratch buffers at their maximum sizes
    replicas_.reserve(num_redundant_per_nvl_);
    gpu_load_.resize(num_nvl_ranks_);
    gpu_slots_used_.resize(num_nvl_ranks_);
    expert_on_rank_.resize(static_cast<size_t>(num_nvl_ranks_) * num_logical_per_nvl_);
}

// ============================================================================
// solve() – hot path, zero allocations
// ============================================================================

void PlacementSolver::solve(const int32_t* __restrict__ expert_loads,
                            int32_t* __restrict__ p2l_map,
                            int32_t* __restrict__ l2p_map,
                            int32_t* __restrict__ lcnts,
                            float balance_threshold) const {
    // ------------------------------------------------------------------
    // Initialize output maps
    // ------------------------------------------------------------------
    std::memset(p2l_map, 0xFF, static_cast<size_t>(num_global_physical_) * sizeof(int32_t));
    std::memset(l2p_map, 0xFF, static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t));
    std::memset(lcnts, 0, static_cast<size_t>(num_global_logical_experts_) * sizeof(int32_t));

    // ------------------------------------------------------------------
    // Step 1: Place masters (fixed positions, never change)
    // ------------------------------------------------------------------
    for (int l = 0; l < num_global_logical_experts_; ++l) {
        const int rank = l / num_local_master_;
        const int local_idx = l % num_local_master_;
        const int p = rank * num_local_physical_ + local_idx;

        p2l_map[p] = l;
        l2p_map[l * max_replicas_dim_ + 0] = p;
        lcnts[l] = 1;
    }

    // Early exit: no redundant slots or single-GPU NVL domains
    if (num_local_redundant_ == 0 || num_nvl_ranks_ <= 1) {
        return;
    }

    // ------------------------------------------------------------------
    // Step 2: Per NVL domain — replicate and pack
    // ------------------------------------------------------------------
    for (int d = 0; d < num_nvl_domains_; ++d) {
        const int domain_start_rank = d * num_nvl_ranks_;
        const int domain_start_log = domain_start_rank * num_local_master_;

        // ==============================================================
        // Phase A: Greedy replication (EPLB replicate_experts strategy)
        //   Hard cap per expert: max_extra_replicas_ == num_nvl_ranks_ - 1
        // ==============================================================

        // Compute avg load per slot for early-stop check
        double total_load = 0;
        for (int i = 0; i < num_logical_per_nvl_; ++i) {
            total_load += expert_loads[domain_start_log + i];
        }
        const double avg_per_slot = (total_load > 0) ? total_load / (num_nvl_ranks_ * num_local_master_) : 0.0;

        for (int slot = 0; slot < num_redundant_per_nvl_; ++slot) {
            int best_l = -1;
            double best_score = -1.0;

            for (int i = 0; i < num_logical_per_nvl_; ++i) {
                const int l = domain_start_log + i;
                // Skip if this expert already uses all available ranks
                if (lcnts[l] - 1 >= max_extra_replicas_)
                    continue;

                const double score = static_cast<double>(expert_loads[l]) / lcnts[l];
                // Deterministic tie-break: higher score wins; equal score → lower index
                if (score > best_score || (score == best_score && (best_l == -1 || l < best_l))) {
                    best_score = score;
                    best_l = l;
                }
            }

            // Early-stop: all experts' per-replica load is within threshold of average
            if (balance_threshold > 1.0f && avg_per_slot > 0.0 && best_score <= avg_per_slot * balance_threshold) {
                break;
            }

            if (best_l < 0)
                break;  // all experts at capacity
            lcnts[best_l]++;
        }

        // ==============================================================
        // Phase B: Build sorted replica list (LPT order)
        // ==============================================================
        replicas_.clear();
        for (int i = 0; i < num_logical_per_nvl_; ++i) {
            const int l = domain_start_log + i;
            const int num_extra = lcnts[l] - 1;
            if (num_extra <= 0)
                continue;

            const double lpr = static_cast<double>(expert_loads[l]) / lcnts[l];
            for (int j = 0; j < num_extra; ++j) {
                replicas_.push_back({l, lpr});
            }
        }

        // Sort descending by load; tie-break on logical_id (ascending)
        std::sort(replicas_.begin(), replicas_.end(), [](const ReplicaEntry& a, const ReplicaEntry& b) {
            if (a.load_per_replica != b.load_per_replica)
                return a.load_per_replica > b.load_per_replica;
            return a.logical_id < b.logical_id;
        });

        // ==============================================================
        // Phase C: Pack replicas to GPUs (greedy bin-packing)
        //
        //   Constraints checked per GPU candidate:
        //     1. Redundant slot available  (gpu_slots_used < num_local_redundant)
        //     2. Expert not already on this rank  (expert_on_rank bitmap)
        //        — subsumes the "not master rank" check
        // ==============================================================

        // Initialize GPU loads (master contribution, post-replication)
        std::fill(gpu_load_.begin(), gpu_load_.end(), 0.0);
        std::fill(gpu_slots_used_.begin(), gpu_slots_used_.end(), 0);

        for (int i = 0; i < num_nvl_ranks_; ++i) {
            for (int j = 0; j < num_local_master_; ++j) {
                const int l = (domain_start_rank + i) * num_local_master_ + j;
                gpu_load_[i] += static_cast<double>(expert_loads[l]) / lcnts[l];
            }
        }

        // Initialize expert-on-rank bitmap (masters)
        const size_t bitmap_bytes = static_cast<size_t>(num_nvl_ranks_) * num_logical_per_nvl_;
        std::memset(expert_on_rank_.data(), 0, bitmap_bytes);
        for (int i = 0; i < num_nvl_ranks_; ++i) {
            const int base = i * num_logical_per_nvl_;
            const int master_start = i * num_local_master_;  // local offset within domain
            for (int j = 0; j < num_local_master_; ++j) {
                expert_on_rank_[base + master_start + j] = 1;
            }
        }

        // Assign each replica
        for (const auto& rep : replicas_) {
            const int l_local = rep.logical_id - domain_start_log;

            // Find GPU with minimum load that satisfies both constraints
            int best_gpu = -1;
            double best_load = 1e18;
            for (int i = 0; i < num_nvl_ranks_; ++i) {
                if (gpu_slots_used_[i] >= num_local_redundant_)
                    continue;
                if (expert_on_rank_[i * num_logical_per_nvl_ + l_local])
                    continue;
                // Deterministic tie-break: lower index wins
                if (gpu_load_[i] < best_load || (gpu_load_[i] == best_load && (best_gpu == -1 || i < best_gpu))) {
                    best_load = gpu_load_[i];
                    best_gpu = i;
                }
            }

            if (best_gpu == -1) {
                // Cannot place — revert replication decision
                lcnts[rep.logical_id]--;
                continue;
            }

            // Physical index
            const int global_rank = domain_start_rank + best_gpu;
            const int phys_idx = global_rank * num_local_physical_ + num_local_master_ + gpu_slots_used_[best_gpu];

            // Update p2l
            p2l_map[phys_idx] = rep.logical_id;

            // Update l2p (next free slot for this expert, scanning from 1)
            int32_t* l2p_row = l2p_map + rep.logical_id * max_replicas_dim_;
            for (int k = 1; k < max_replicas_dim_; ++k) {
                if (l2p_row[k] == -1) {
                    l2p_row[k] = phys_idx;
                    break;
                }
            }

            // Update tracking
            gpu_load_[best_gpu] += rep.load_per_replica;
            gpu_slots_used_[best_gpu]++;
            expert_on_rank_[best_gpu * num_logical_per_nvl_ + l_local] = 1;
        }
    }
}

}  // namespace ultra_ep::solver
