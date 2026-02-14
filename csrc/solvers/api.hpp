#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace ultra_ep::solver {

/**
 * PlacementSolver: computes expert replication and placement for a single MoE layer.
 *
 * Designed to be instantiated once (per Manager) and reused across layers and
 * training steps.  All scratch buffers are pre-allocated in the constructor so
 * that solve() performs zero heap allocations on the hot path.
 *
 * Algorithm (per NVL domain):
 *   Phase A – Greedy replication: repeatedly pick the logical expert with the
 *             highest (load / current_count) and give it one more replica.
 *             Hard cap: each expert has at most (num_nvl_ranks − 1) replicas,
 *             because every rank may hold at most ONE copy of any logical expert
 *             (to keep grouped-GEMM legal), and the master already occupies one rank.
 *   Phase B – LPT bin-packing: sort replicas by descending per-replica load, then
 *             greedily assign each to the non-master rank with the smallest current
 *             total load that still has a free redundant slot AND does not already
 *             host this logical expert.
 *
 * Deterministic: identical inputs → identical outputs on every rank.
 * CPU-only, no CUDA / NVSHMEM dependency.
 */
class PlacementSolver {
public:
    PlacementSolver(int num_global_logical_experts,
                    int num_ranks,
                    int num_local_master_experts,
                    int num_local_redundant_experts,
                    int num_nvl_ranks,
                    int max_replicas_dim);

    /**
     * Compute placement for one layer.
     *
     * @param expert_loads  [num_global_logical_experts] int32 – per-expert token counts
     * @param p2l_map       [num_global_physical_experts] int32 – output
     * @param l2p_map       [num_global_logical_experts * max_replicas_dim] int32 – output (row-major)
     * @param lcnts         [num_global_logical_experts] int32 – output
     */
    void solve(const int32_t* __restrict__ expert_loads,
               int32_t* __restrict__ p2l_map,
               int32_t* __restrict__ l2p_map,
               int32_t* __restrict__ lcnts) const;

private:
    // ---- Configuration (immutable after construction) ----
    int num_global_logical_experts_;
    int num_ranks_;
    int num_local_master_;
    int num_local_redundant_;
    int num_nvl_ranks_;
    int max_replicas_dim_;

    int num_local_physical_;
    int num_global_physical_;
    int num_nvl_domains_;
    int num_logical_per_nvl_;
    int num_redundant_per_nvl_;
    int max_extra_replicas_;  // == num_nvl_ranks_ - 1

    // ---- Pre-allocated scratch buffers (reused across solve() calls) ----
    // Mutable because solve() is logically const (same config → same result).
    struct ReplicaEntry {
        int logical_id;
        double load_per_replica;
    };

    mutable std::vector<ReplicaEntry> replicas_;
    mutable std::vector<double> gpu_load_;
    mutable std::vector<int> gpu_slots_used_;

    // Per-rank expert occupancy bitmap, flat [num_nvl_ranks * num_logical_per_nvl].
    // 1 = this expert already occupies this rank (master or replica).
    mutable std::vector<uint8_t> expert_on_rank_;
};

inline void register_apis(pybind11::module_& m) {
    pybind11::class_<PlacementSolver>(m, "PlacementSolver")
        .def(pybind11::init<int, int, int, int, int, int>())
        .def("solve",
             [](const PlacementSolver& self,
                torch::Tensor& expert_loads,
                torch::Tensor& p2l_map,
                torch::Tensor& l2p_map,
                torch::Tensor& lcnts) {
                 self.solve(expert_loads.data<int32_t>(),
                            p2l_map.data<int32_t>(),
                            l2p_map.data<int32_t>(),
                            lcnts.data<int32_t>());
             });
}

}  // namespace ultra_ep::solver