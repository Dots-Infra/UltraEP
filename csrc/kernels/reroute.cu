/**
 * CUDA kernels for reroute: expand logical routing map to physical routing map.
 *
 * Algorithm (per logical expert, one warp per expert):
 *   1. Cooperative load: transpose a tile of routing_map[tile:tile+TILE_T, block_experts]
 *      from global memory into shared memory for bank-conflict-free column access.
 *   2. Warp scan: each warp processes its expert column in groups of 32 tokens using
 *      __ballot_sync / __popc for a warp-level prefix sum that gives each active token
 *      its deterministic round-robin rank.
 *   3. Scatter (forward) or gather (backward): use the rank to compute the physical
 *      expert index via l2p_map[expert, rank % C], then read/write probs.
 *
 * The mapping is deterministic: tokens are processed in ascending index order per expert,
 * and the round-robin counter increments monotonically.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

// ============================================================================
// Forward kernel: scatter probs from [T,L] logical to [T,P] physical space
// ============================================================================
template <typename scalar_t, int TILE_T, int WARPS_PER_BLOCK, int SMEM_PAD>
__global__ void reroute_forward_kernel(const bool* __restrict__ routing_map,
                                       const scalar_t* __restrict__ probs,
                                       const int32_t* __restrict__ l2p_map,
                                       const int32_t* __restrict__ lcnts,
                                       bool* __restrict__ expanded_routing_map,
                                       scalar_t* __restrict__ expanded_probs,
                                       const int T,
                                       const int L,
                                       const int P,
                                       const int max_replicas) {
    // Transposed SMEM tile: smem[expert_in_block][token_in_tile + PAD]
    // PAD eliminates bank conflicts: with bool (1B) and 32 banks of 4B,
    // stride = TILE_T + PAD avoids all-same-bank access patterns.
    constexpr int SMEM_STRIDE = TILE_T + SMEM_PAD;
    __shared__ bool smem_routing[WARPS_PER_BLOCK][SMEM_STRIDE];

    const int warp_id = threadIdx.x >> 5;  // threadIdx.x / 32
    const int lane = threadIdx.x & 31;     // threadIdx.x % 32
    const int block_expert_base = blockIdx.x * WARPS_PER_BLOCK;
    const int expert_id = block_expert_base + warp_id;
    const bool warp_active = (expert_id < L);

    // Load per-expert constants into registers
    int C = 0;
    int counter = 0;
    if (warp_active) {
        C = lcnts[expert_id];
    }

    // Number of active experts in this block (for cooperative load bounds)
    const int num_experts_this_block = min(WARPS_PER_BLOCK, L - block_expert_base);

    // Process tokens in tiles
    for (int tile_start = 0; tile_start < T; tile_start += TILE_T) {
        const int tile_T = min(TILE_T, T - tile_start);

        // Cooperative load: all threads in the block load the routing_map tile
        // and transpose it into SMEM for bank-conflict-free column reads.
        // Iterate over TILE_T * WARPS_PER_BLOCK slots (power-of-2 for fast index math);
        // the bounds check skips padding slots for the last block / last tile.
        for (int i = threadIdx.x; i < TILE_T * WARPS_PER_BLOCK; i += blockDim.x) {
            const int local_e = i & (WARPS_PER_BLOCK - 1);  // i % WARPS_PER_BLOCK
            const int local_t = i >> 3;                     // i / WARPS_PER_BLOCK (=8)
            if (local_e < num_experts_this_block && local_t < tile_T) {
                smem_routing[local_e][local_t] = routing_map[(tile_start + local_t) * L + block_expert_base + local_e];
            }
        }
        __syncthreads();

        // Warp scan: each warp processes its expert column
        if (warp_active) {
            for (int offset = 0; offset < tile_T; offset += 32) {
                const int local_t = offset + lane;
                const int global_t = tile_start + local_t;
                const bool active = (local_t < tile_T) ? smem_routing[warp_id][local_t] : false;

                // Warp-level prefix sum using ballot + population count
                const unsigned ballot = __ballot_sync(0xFFFFFFFF, active);
                const int preceding = __popc(ballot & ((1u << lane) - 1));
                const int total_active = __popc(ballot);
                const int my_rank = counter + preceding;

                if (active && global_t < T) {
                    const int replica_idx = my_rank % C;
                    const int phys = l2p_map[expert_id * max_replicas + replica_idx];
                    expanded_routing_map[global_t * P + phys] = true;
                    expanded_probs[global_t * P + phys] = probs[global_t * L + expert_id];
                }

                counter += total_active;
            }
        }

        __syncthreads();  // ensure SMEM is safe to overwrite for next tile
    }
}

// ============================================================================
// Backward kernel: gather gradients from [T,P] physical to [T,L] logical
// ============================================================================
template <typename scalar_t, int TILE_T, int WARPS_PER_BLOCK, int SMEM_PAD>
__global__ void reroute_backward_kernel(const scalar_t* __restrict__ grad_expanded_probs,
                                        const bool* __restrict__ routing_map,
                                        const int32_t* __restrict__ l2p_map,
                                        const int32_t* __restrict__ lcnts,
                                        scalar_t* __restrict__ grad_probs,
                                        const int T,
                                        const int L,
                                        const int P,
                                        const int max_replicas) {
    constexpr int SMEM_STRIDE = TILE_T + SMEM_PAD;
    __shared__ bool smem_routing[WARPS_PER_BLOCK][SMEM_STRIDE];

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int block_expert_base = blockIdx.x * WARPS_PER_BLOCK;
    const int expert_id = block_expert_base + warp_id;
    const bool warp_active = (expert_id < L);

    int C = 0;
    int counter = 0;
    if (warp_active) {
        C = lcnts[expert_id];
    }

    const int num_experts_this_block = min(WARPS_PER_BLOCK, L - block_expert_base);

    for (int tile_start = 0; tile_start < T; tile_start += TILE_T) {
        const int tile_T = min(TILE_T, T - tile_start);

        // Cooperative load of routing_map tile (same as forward)
        for (int i = threadIdx.x; i < TILE_T * WARPS_PER_BLOCK; i += blockDim.x) {
            const int local_e = i & (WARPS_PER_BLOCK - 1);
            const int local_t = i >> 3;
            if (local_e < num_experts_this_block && local_t < tile_T) {
                smem_routing[local_e][local_t] = routing_map[(tile_start + local_t) * L + block_expert_base + local_e];
            }
        }
        __syncthreads();

        if (warp_active) {
            for (int offset = 0; offset < tile_T; offset += 32) {
                const int local_t = offset + lane;
                const int global_t = tile_start + local_t;
                const bool active = (local_t < tile_T) ? smem_routing[warp_id][local_t] : false;

                const unsigned ballot = __ballot_sync(0xFFFFFFFF, active);
                const int preceding = __popc(ballot & ((1u << lane) - 1));
                const int total_active = __popc(ballot);
                const int my_rank = counter + preceding;

                if (active && global_t < T) {
                    const int replica_idx = my_rank % C;
                    const int phys = l2p_map[expert_id * max_replicas + replica_idx];
                    // Gather: grad_probs[t, logical] = grad_expanded_probs[t, physical]
                    grad_probs[global_t * L + expert_id] = grad_expanded_probs[global_t * P + phys];
                }

                counter += total_active;
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Host-side launch functions
// ============================================================================

void run_reroute_forward(const bool* routing_map,
                         const void* probs,
                         const int32_t* l2p_map,
                         const int32_t* lcnts,
                         bool* expanded_routing_map,
                         void* expanded_probs,
                         int T,
                         int L,
                         int P,
                         int max_replicas,
                         at::ScalarType dtype,
                         cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    const int num_blocks = (L + REROUTE_WARPS_PER_BLOCK - 1) / REROUTE_WARPS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(REROUTE_THREADS_PER_BLOCK);

    // Kernel uses static __shared__ memory; no dynamic SMEM needed.
    // Static SMEM per block: WARPS_PER_BLOCK * (TILE_T + PAD) bytes ≈ 2 KB.

    // Dispatch on scalar type
    if (dtype == at::ScalarType::Float) {
        reroute_forward_kernel<float, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(routing_map,
                                         static_cast<const float*>(probs),
                                         l2p_map,
                                         lcnts,
                                         expanded_routing_map,
                                         static_cast<float*>(expanded_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else if (dtype == at::ScalarType::BFloat16) {
        reroute_forward_kernel<__nv_bfloat16, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(routing_map,
                                         static_cast<const __nv_bfloat16*>(probs),
                                         l2p_map,
                                         lcnts,
                                         expanded_routing_map,
                                         static_cast<__nv_bfloat16*>(expanded_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else if (dtype == at::ScalarType::Half) {
        reroute_forward_kernel<__half, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(routing_map,
                                         static_cast<const __half*>(probs),
                                         l2p_map,
                                         lcnts,
                                         expanded_routing_map,
                                         static_cast<__half*>(expanded_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else {
        EP_HOST_ASSERT(false && "Unsupported dtype for reroute_cuda_forward");
    }
}

void run_reroute_backward(const void* grad_expanded_probs,
                          const bool* routing_map,
                          const int32_t* l2p_map,
                          const int32_t* lcnts,
                          void* grad_probs,
                          int T,
                          int L,
                          int P,
                          int max_replicas,
                          at::ScalarType dtype,
                          cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    const int num_blocks = (L + REROUTE_WARPS_PER_BLOCK - 1) / REROUTE_WARPS_PER_BLOCK;
    dim3 grid(num_blocks);
    dim3 block(REROUTE_THREADS_PER_BLOCK);

    if (dtype == at::ScalarType::Float) {
        reroute_backward_kernel<float, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(static_cast<const float*>(grad_expanded_probs),
                                         routing_map,
                                         l2p_map,
                                         lcnts,
                                         static_cast<float*>(grad_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else if (dtype == at::ScalarType::BFloat16) {
        reroute_backward_kernel<__nv_bfloat16, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(grad_expanded_probs),
                                         routing_map,
                                         l2p_map,
                                         lcnts,
                                         static_cast<__nv_bfloat16*>(grad_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else if (dtype == at::ScalarType::Half) {
        reroute_backward_kernel<__half, REROUTE_TILE_T, REROUTE_WARPS_PER_BLOCK, REROUTE_SMEM_PAD>
            <<<grid, block, 0, stream>>>(static_cast<const __half*>(grad_expanded_probs),
                                         routing_map,
                                         l2p_map,
                                         lcnts,
                                         static_cast<__half*>(grad_probs),
                                         T,
                                         L,
                                         P,
                                         max_replicas);
    } else {
        EP_HOST_ASSERT(false && "Unsupported dtype for reroute_cuda_backward");
    }
}

}  // namespace ultra_ep::kernels
