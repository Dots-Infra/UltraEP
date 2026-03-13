#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "../utils/exception.cuh"

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
               int32_t* __restrict__ lcnts,
               float balance_threshold = 1.0f) const;

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

/**
 * PlacementSolverGPU: GPU-based expert replication and placement solver (V3).
 *
 * Runs entirely on GPU after the NVSHMEM allreduce, eliminating:
 *   - D2H copy of expert_loads
 *   - cudaStreamSynchronize (blocking CPU)
 *   - H2D copy of placement results
 *
 * V3 optimizations (NCU data-driven):
 *   - 2-warp cooperative kernel (64 threads) for G <= 64:
 *     hides shuffle latency (Stall Wait) and smem latency (Short Scoreboard)
 *   - Phase C parallel argmin: each warp handles 32 GPUs independently,
 *     reducing shuffle count per round from 11 to 6
 *   - Template specialization on EPL and COMPACT_EOR:
 *     eliminates runtime branches, enables precise loop unrolling
 *   - EOR union: compact (uint64) and packed (uint32) share smem via union
 *   - External cudaMemsetAsync for p2l/l2p/lcnts initialization
 *
 * Algorithm (per NVL domain, in a single CUDA kernel):
 *   Phase 0 — Place masters, init smem (2-warp: warp 0 masters, warp 1 smem init)
 *   Phase A — Binary search for replica counts (both warps redundantly execute)
 *   Phase B — Build + bitonic-sort replica list (64 threads parallel sort)
 *   Phase C — Greedy bin-pack with 2-warp cooperative argmin
 *
 * Deterministic: same expert_loads → identical p2l/l2p/lcnts on every rank.
 * Results satisfy the same 8 invariants as PlacementSolver (CPU).
 */
class PlacementSolverGPU {
public:
    PlacementSolverGPU(int num_global_logical_experts,
                       int num_ranks,
                       int num_local_master_experts,
                       int num_local_redundant_experts,
                       int num_nvl_ranks,
                       int max_replicas_dim);

    /**
     * Compute placement for one layer entirely on GPU.
     *
     * @param expert_loads_gpu  [num_global_logical_experts] int32 device ptr
     * @param p2l_gpu           [num_global_physical_experts] int32 device ptr – output
     * @param l2p_gpu           [num_global_logical_experts * max_replicas_dim] int32 device ptr – output
     * @param lcnts_gpu         [num_global_logical_experts] int32 device ptr – output
     * @param stream            CUDA stream to launch the kernel on
     */
    void solve(const int32_t* expert_loads_gpu,
               int32_t* p2l_gpu,
               int32_t* l2p_gpu,
               int32_t* lcnts_gpu,
               cudaStream_t stream,
               float balance_threshold = 1.0f) const;

private:
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
};

/**
 * PlacementSolverQuota: quota-aware placement solver.
 *
 * Runs entirely on GPU.  For each NVL domain, a single CUDA block:
 *   - binary-searches the minimum feasible quota threshold
 *   - uses warp-cooperative feasibility oracles and multi-warp probing to
 *     accelerate the skewed-load search path
 *   - materializes p2l/l2p/lcnts and per-instance quotas
 *   - fuses the locality-aware per-rank quota decomposition
 *
 * The locality phase bulk-loads per-rank expert loads into shared memory and
 * writes the current rank's `rank_quota_prefix` slice directly on device, so
 * the dense reroute path stays on the pure-CUDA fast path.
 */
class PlacementSolverQuota {
public:
    PlacementSolverQuota(int num_global_logical_experts,
                         int num_ranks,
                         int num_local_master_experts,
                         int num_local_redundant_experts,
                         int num_nvl_ranks,
                         int max_replicas_dim);
    ~PlacementSolverQuota();

    void solve(const int32_t* expert_loads_gpu,
               const int32_t* expert_loads_per_rank_gpu,
               int32_t* p2l_gpu,
               int32_t* l2p_gpu,
               int32_t* lcnts_gpu,
               int32_t* quota_gpu,
               int32_t* quota_prefix_gpu,
               int32_t* rank_quota_prefix_gpu,
               cudaStream_t stream,
               float balance_threshold = 1.0f,
               int32_t min_tokens_per_replica = 1,
               bool allow_zero_master_quota = true,
               bool locality_aware = true) const;

private:
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
};

/**
 * RerouteSolver: expands a logical routing map to a physical routing map using
 * deterministic round-robin dispatch.
 *
 * For each logical expert l with C_l = lcnts[l] physical instances (1 master +
 * replicas), all tokens routed to l are numbered in global token-index order.
 * The k-th token is assigned to physical expert l2p[l, k % C_l].
 *
 * This produces a bijective mapping from each active (token, logical_expert) pair
 * to a unique (token, physical_expert) pair.  The mapping is represented as three
 * parallel index arrays (token_indices, logical_indices, physical_indices) of
 * length N (number of active routing pairs).
 *
 * Design rationale (CPU-side computation):
 *   - routing_map.nonzero() runs on GPU (CUB-optimised), yielding a compact [N, 2]
 *     index tensor that is D2H-copied (~128 KB for T=4096, topk=2).
 *   - The round-robin assignment is a single sequential O(N) scan with per-expert
 *     counters — ideal for CPU cache and branch prediction.
 *   - l2p and lcnts already reside on CPU (pinned); no extra copy needed.
 *   - Result index arrays (~192 KB) are H2D-copied back to GPU.
 *   - Total memcpy ≈ 300 KB, latency ≈ tens of microseconds.
 *
 * Deterministic: identical inputs → identical outputs, regardless of GPU timing.
 * Thread-safe: solve() is logically const (internal buffers are mutable scratch).
 */
class RerouteSolver {
public:
    RerouteSolver(int num_global_logical_experts, int num_global_physical_experts, int max_replicas_dim);

    /**
     * Compute the round-robin reroute mapping.
     *
     * @param routing_map  [num_tokens, num_logical] bool – logical routing map (CPU or GPU)
     * @param l2p          [num_logical, max_replicas] int32 – logical-to-physical map (CPU)
     * @param lcnts        [num_logical] int32 – per-expert replica counts (CPU)
     *
     * @return (token_indices, logical_indices, physical_indices)
     *         each [N] int64, on the same device as routing_map.
     *         N = total number of active (token, logical_expert) pairs.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> solve(const torch::Tensor& routing_map,
                                                                  const int32_t* __restrict__ l2p_map,
                                                                  const int32_t* __restrict__ lcnts) const;

private:
    int num_logical_;
    int num_physical_;
    int max_replicas_;

    // Pre-allocated per-expert counters, reused across solve() calls.
    mutable std::vector<int32_t> counters_;
};

inline void register_apis(pybind11::module_& m) {
    pybind11::class_<PlacementSolver>(m, "PlacementSolver")
        .def(pybind11::init<int, int, int, int, int, int>())
        .def("solve",
             [](const PlacementSolver& self,
                torch::Tensor& expert_loads,
                torch::Tensor& p2l_map,
                torch::Tensor& l2p_map,
                torch::Tensor& lcnts,
                float balance_threshold) {
                 self.solve(expert_loads.data<int32_t>(),
                            p2l_map.data<int32_t>(),
                            l2p_map.data<int32_t>(),
                            lcnts.data<int32_t>(),
                            balance_threshold);
             },
             pybind11::arg("expert_loads"),
             pybind11::arg("p2l_map"),
             pybind11::arg("l2p_map"),
             pybind11::arg("lcnts"),
             pybind11::arg("balance_threshold") = 1.0f);

    pybind11::class_<PlacementSolverGPU>(m, "PlacementSolverGPU")
        .def(pybind11::init<int, int, int, int, int, int>())
        .def("solve",
             [](const PlacementSolverGPU& self,
                torch::Tensor& expert_loads,
                torch::Tensor& p2l_map,
                torch::Tensor& l2p_map,
                torch::Tensor& lcnts,
                float balance_threshold) {
                 EP_HOST_ASSERT(expert_loads.device().is_cuda() && expert_loads.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(p2l_map.device().is_cuda() && p2l_map.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(l2p_map.device().is_cuda() && l2p_map.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(lcnts.device().is_cuda() && lcnts.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(expert_loads.is_contiguous() && p2l_map.is_contiguous());
                 EP_HOST_ASSERT(l2p_map.is_contiguous() && lcnts.is_contiguous());
                 auto stream = at::cuda::getCurrentCUDAStream();
                 self.solve(expert_loads.data_ptr<int32_t>(),
                            p2l_map.data_ptr<int32_t>(),
                            l2p_map.data_ptr<int32_t>(),
                            lcnts.data_ptr<int32_t>(),
                            stream.stream(),
                            balance_threshold);
             },
             pybind11::arg("expert_loads"),
             pybind11::arg("p2l_map"),
             pybind11::arg("l2p_map"),
             pybind11::arg("lcnts"),
             pybind11::arg("balance_threshold") = 1.0f);

    pybind11::class_<PlacementSolverQuota>(m, "PlacementSolverQuota")
        .def(pybind11::init<int, int, int, int, int, int>())
        .def("solve",
             [](const PlacementSolverQuota& self,
                torch::Tensor& expert_loads,
                torch::Tensor& expert_loads_per_rank,
                torch::Tensor& p2l_map,
                torch::Tensor& l2p_map,
                torch::Tensor& lcnts,
                torch::Tensor& quota,
                torch::Tensor& quota_prefix,
                torch::Tensor& rank_quota_prefix,
                float balance_threshold,
                int32_t min_tokens_per_replica,
                bool allow_zero_master_quota,
                bool locality_aware) {
                 EP_HOST_ASSERT(expert_loads.device().is_cuda() && expert_loads.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(expert_loads_per_rank.device().is_cuda() &&
                                expert_loads_per_rank.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(p2l_map.device().is_cuda() && p2l_map.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(l2p_map.device().is_cuda() && l2p_map.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(lcnts.device().is_cuda() && lcnts.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(quota.device().is_cuda() && quota.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(quota_prefix.device().is_cuda() && quota_prefix.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(rank_quota_prefix.device().is_cuda() &&
                                rank_quota_prefix.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(expert_loads.is_contiguous() && expert_loads_per_rank.is_contiguous());
                 EP_HOST_ASSERT(p2l_map.is_contiguous() && l2p_map.is_contiguous() && lcnts.is_contiguous());
                 EP_HOST_ASSERT(quota.is_contiguous() && quota_prefix.is_contiguous() &&
                                rank_quota_prefix.is_contiguous());
                 auto stream = at::cuda::getCurrentCUDAStream();
                 self.solve(expert_loads.data_ptr<int32_t>(),
                            expert_loads_per_rank.data_ptr<int32_t>(),
                            p2l_map.data_ptr<int32_t>(),
                            l2p_map.data_ptr<int32_t>(),
                            lcnts.data_ptr<int32_t>(),
                            quota.data_ptr<int32_t>(),
                            quota_prefix.data_ptr<int32_t>(),
                            rank_quota_prefix.data_ptr<int32_t>(),
                            stream.stream(),
                            balance_threshold,
                            min_tokens_per_replica,
                            allow_zero_master_quota,
                            locality_aware);
             },
             pybind11::arg("expert_loads"),
             pybind11::arg("expert_loads_per_rank"),
             pybind11::arg("p2l_map"),
             pybind11::arg("l2p_map"),
             pybind11::arg("lcnts"),
             pybind11::arg("quota"),
             pybind11::arg("quota_prefix"),
             pybind11::arg("rank_quota_prefix"),
             pybind11::arg("balance_threshold") = 1.0f,
             pybind11::arg("min_tokens_per_replica") = 1,
             pybind11::arg("allow_zero_master_quota") = true,
             pybind11::arg("locality_aware") = true);

    pybind11::class_<RerouteSolver>(m, "RerouteSolver")
        .def(pybind11::init<int, int, int>())
        .def("solve",
             [](const RerouteSolver& self,
                const torch::Tensor& routing_map,
                torch::Tensor& l2p_map,
                torch::Tensor& lcnts) {
                 EP_HOST_ASSERT(l2p_map.device().is_cpu());
                 EP_HOST_ASSERT(lcnts.device().is_cpu());
                 EP_HOST_ASSERT(l2p_map.is_contiguous());
                 EP_HOST_ASSERT(lcnts.is_contiguous());
                 EP_HOST_ASSERT(l2p_map.dtype() == torch::kInt32);
                 EP_HOST_ASSERT(lcnts.dtype() == torch::kInt32);
                 return self.solve(routing_map, l2p_map.data<int32_t>(), lcnts.data<int32_t>());
             });
}

}  // namespace ultra_ep::solver
