/**
 * RerouteSolver: deterministic round-robin token dispatch from logical to physical experts.
 *
 * Algorithm:
 *   For each logical expert l with C_l physical instances (master + replicas),
 *   the tokens routed to l are numbered 0, 1, 2, ... in ascending token-index order.
 *   Token k is dispatched to physical expert l2p[l, k % C_l].
 *
 * Implementation:
 *   1. routing_map.nonzero() on GPU → compact [N, 2] index tensor (row-major order).
 *   2. D2H copy of the [N, 2] tensor (~N * 16 bytes).
 *   3. Sequential O(N) scan on CPU with per-expert counters.
 *      - For each (token, logical) pair, compute physical = l2p[l, counter[l] % lcnts[l]].
 *   4. H2D copy of the three result index arrays (~N * 24 bytes).
 *
 * All scratch buffers are pre-allocated in the constructor; solve() performs
 * zero heap allocations on the hot path (aside from torch::empty for output tensors,
 * which are small and unavoidable).
 */

#include <cstring>

#include "api.hpp"

namespace ultra_ep::solver {

RerouteSolver::RerouteSolver(int num_global_logical_experts, int num_global_physical_experts, int max_replicas_dim)
    : num_logical_(num_global_logical_experts),
      num_physical_(num_global_physical_experts),
      max_replicas_(max_replicas_dim),
      counters_(num_global_logical_experts, 0) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RerouteSolver::solve(
    const torch::Tensor& routing_map,
    const int32_t* __restrict__ l2p_ptr,
    const int32_t* __restrict__ lcnts_ptr) const {
    EP_HOST_ASSERT(routing_map.dim() == 2);
    EP_HOST_ASSERT(routing_map.size(1) == num_logical_);
    EP_HOST_ASSERT(routing_map.is_cuda());

    auto device = routing_map.device();

    // Step 1: Get active (token, logical_expert) pairs via nonzero()
    //   nonzero() returns [N, 2] int64, sorted in row-major (C) order:
    //   first by token index, then by logical expert index.
    //   This guarantees that within each logical expert, tokens appear
    //   in ascending order — exactly the ordering required for round-robin.
    torch::Tensor nz = routing_map.nonzero();  // [N, 2], same device as routing_map
    const int64_t N = nz.size(0);

    // Early return for empty routing
    if (N == 0) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        return std::make_tuple(torch::empty({0}, opts), torch::empty({0}, opts), torch::empty({0}, opts));
    }

    // Step 2: D2H copy of nonzero indices for CPU processing
    // Must use contiguous() or accessor: nonzero() returns a transposed view with strides [1, N]
    torch::Tensor nz_cpu = nz.cpu();
    auto nz_accessor = nz_cpu.accessor<int64_t, 2>();
    // nz_data layout: [t0, l0, t1, l1, ...] (row-major, N rows x 2 cols)

    // Step 3: Deterministic round-robin assignment (CPU, O(N))
    // Reset per-expert counters
    std::memset(counters_.data(), 0, static_cast<size_t>(num_logical_) * sizeof(int32_t));

    // Allocate output index tensors on CPU
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);

    torch::Tensor token_idx = torch::empty({N}, cpu_opts);
    torch::Tensor logical_idx = torch::empty({N}, cpu_opts);
    torch::Tensor physical_idx = torch::empty({N}, cpu_opts);

    int64_t* tok_out = token_idx.data<int64_t>();
    int64_t* log_out = logical_idx.data<int64_t>();
    int64_t* phy_out = physical_idx.data<int64_t>();

    for (int64_t i = 0; i < N; ++i) {
        const int64_t t = nz_accessor[i][0];                // token index
        const int l = static_cast<int>(nz_accessor[i][1]);  // logical expert index
        EP_HOST_ASSERT(l >= 0 && l < num_logical_);

        const int32_t cnt = lcnts_ptr[l];
        EP_HOST_ASSERT(cnt >= 1 && cnt <= max_replicas_);
        const int32_t c = counters_[l];
        const int32_t replica_idx = c % cnt;
        EP_HOST_ASSERT(replica_idx >= 0 && replica_idx < max_replicas_);
        const int32_t phys = l2p_ptr[l * max_replicas_ + replica_idx];

        tok_out[i] = t;
        log_out[i] = static_cast<int64_t>(l);
        phy_out[i] = static_cast<int64_t>(phys);

        counters_[l] = c + 1;
    }

    // H2D transfer
    token_idx = token_idx.to(device, /*non_blocking=*/false);
    logical_idx = logical_idx.to(device, /*non_blocking=*/false);
    physical_idx = physical_idx.to(device, /*non_blocking=*/false);

    return std::make_tuple(token_idx, logical_idx, physical_idx);
}

}  // namespace ultra_ep::solver
