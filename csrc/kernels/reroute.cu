/**
 * Optimized CUDA kernels for reroute: expand logical routing map to physical routing map.
 *
 * Forward (two-pass, deterministic round-robin):
 *   Pass 1 — Count: each warp counts active tokens for one (expert, tile) pair
 *            using __ballot_sync / __popc.  Fully parallel across all (expert, tile) pairs.
 *   Pass 2 — Scatter: each warp computes its base rank via warp-parallel prefix-sum of
 *            tile_counts, then scatters probs to the physical expert determined by
 *            l2p_map[expert, rank % C].  Fully parallel.
 *
 * Backward (row-parallel gather):
 *   Each thread handles one (token, expert) pair.  For active pairs, it searches the
 *   forward's expanded_routing_map to find which physical replica was assigned, then
 *   gathers the gradient.  This eliminates the serial round-robin recomputation entirely.
 *   For typical replica counts (1–4), the search is 1–4 iterations.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

// ============================================================================
// Forward pass 1: count active tokens per (expert, tile)
// ============================================================================
template <int TILE_T, int WARPS_PER_BLOCK>
__global__ void reroute_forward_count_kernel(const bool* __restrict__ routing_map,
                                             int32_t* __restrict__ tile_counts,
                                             const int T,
                                             const int L,
                                             const int num_tiles) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int expert_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int tile_id = blockIdx.y;

    if (expert_id >= L)
        return;

    const int tile_start = tile_id * TILE_T;
    const int tile_end = min(tile_start + TILE_T, T);
    int count = 0;

    for (int offset = tile_start; offset < tile_end; offset += 32) {
        const int t = offset + lane;
        const bool active = (t < tile_end) && routing_map[t * L + expert_id];
        const unsigned ballot = __ballot_sync(0xFFFFFFFF, active);
        if (lane == 0)
            count += __popc(ballot);
    }

    if (lane == 0) {
        tile_counts[expert_id * num_tiles + tile_id] = count;
    }
}

// ============================================================================
// Forward pass 2: prefix-sum of tile counts + scatter
// ============================================================================
template <typename scalar_t, int TILE_T, int WARPS_PER_BLOCK, bool QUOTA_MODE>
__global__ void reroute_forward_scatter_kernel(const bool* __restrict__ routing_map,
                                               const scalar_t* __restrict__ probs,
                                               const int32_t* __restrict__ l2p_map,
                                               const int32_t* __restrict__ lcnts,
                                               const int32_t* __restrict__ rank_quota_prefix,
                                               const int32_t* __restrict__ tile_counts,
                                               bool* __restrict__ expanded_routing_map,
                                               scalar_t* __restrict__ expanded_probs,
                                               const int T,
                                               const int L,
                                               const int P,
                                               const int max_replicas,
                                               const int num_tiles) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int expert_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int tile_id = blockIdx.y;

    if (expert_id >= L)
        return;

    const int C = lcnts[expert_id];
    constexpr int PREFETCH_REPLICAS = 8;
    int local_prefix[PREFETCH_REPLICAS];
    int local_l2p[PREFETCH_REPLICAS];
    const int prefetch_count = min(C, PREFETCH_REPLICAS);

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        local_prefix[j] = 0;
        local_l2p[j] = -1;
    }

    if constexpr (QUOTA_MODE) {
#pragma unroll
        for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
            if (j < prefetch_count) {
                local_prefix[j] = rank_quota_prefix[expert_id * max_replicas + j];
                local_l2p[j] = l2p_map[expert_id * max_replicas + j];
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
            if (j < prefetch_count) {
                local_l2p[j] = l2p_map[expert_id * max_replicas + j];
            }
        }
    }

    // Warp-parallel prefix-sum: sum tile_counts[expert][0..tile_id-1] to get base_rank.
    // Each round loads 32 values via coalesced warp read and reduces with shuffles.
    int base_rank = 0;
    const int32_t* my_tile_counts = tile_counts + expert_id * num_tiles;
    for (int base = 0; base < tile_id; base += 32) {
        const int idx = base + lane;
        int val = (idx < tile_id) ? my_tile_counts[idx] : 0;
// Warp-level sum reduction
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, s);
        }
        if (lane == 0)
            base_rank += val;
    }
    base_rank = __shfl_sync(0xFFFFFFFF, base_rank, 0);

    int counter = base_rank;
    const int tile_start = tile_id * TILE_T;
    const int tile_end = min(tile_start + TILE_T, T);

    for (int offset = tile_start; offset < tile_end; offset += 32) {
        const int t = offset + lane;
        const bool active = (t < tile_end) && routing_map[t * L + expert_id];

        const unsigned ballot = __ballot_sync(0xFFFFFFFF, active);
        const int preceding = __popc(ballot & ((1u << lane) - 1));
        const int total_active = __popc(ballot);
        const int my_rank = counter + preceding;

        if (active) {
            int replica_idx = 0;
            int phys = -1;
            if constexpr (QUOTA_MODE) {
                // Branchless upper_bound: count how many prefix values <= my_rank.
                // This avoids warp divergence from a break-based scan (D5 design).
#pragma unroll
                for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
                    if (j < prefetch_count) {
                        replica_idx += (my_rank >= local_prefix[j]) ? 1 : 0;
                    }
                }
                for (int j = PREFETCH_REPLICAS; j < C; ++j) {
                    replica_idx += (my_rank >= rank_quota_prefix[expert_id * max_replicas + j]) ? 1 : 0;
                }
                // Clamp to valid range [0, C-1]
                replica_idx = min(replica_idx, max(C - 1, 0));
                phys = (replica_idx < PREFETCH_REPLICAS) ? local_l2p[replica_idx]
                                                         : l2p_map[expert_id * max_replicas + replica_idx];
            } else {
                replica_idx = my_rank % C;
                phys = (replica_idx < PREFETCH_REPLICAS) ? local_l2p[replica_idx]
                                                         : l2p_map[expert_id * max_replicas + replica_idx];
            }
            expanded_routing_map[t * P + phys] = true;
            expanded_probs[t * P + phys] = probs[t * L + expert_id];
        }

        counter += total_active;
    }
}

// ============================================================================
// Backward kernel: row-parallel gather using expanded_routing_map lookup
// ============================================================================
template <typename scalar_t>
__global__ void reroute_backward_gather_kernel(const scalar_t* __restrict__ grad_expanded_probs,
                                               const bool* __restrict__ routing_map,
                                               const bool* __restrict__ expanded_routing_map,
                                               const int32_t* __restrict__ l2p_map,
                                               const int32_t* __restrict__ lcnts,
                                               scalar_t* __restrict__ grad_probs,
                                               const int T,
                                               const int L,
                                               const int P,
                                               const int max_replicas) {
    const int t = blockIdx.x * blockDim.y + threadIdx.y;
    const int l = blockIdx.y * blockDim.x + threadIdx.x;

    if (t >= T || l >= L)
        return;

    const int idx = t * L + l;
    if (routing_map[idx]) {
        const int C = lcnts[l];
        for (int r = 0; r < C; ++r) {
            const int phys = l2p_map[l * max_replicas + r];
            if (expanded_routing_map[t * P + phys]) {
                grad_probs[idx] = grad_expanded_probs[t * P + phys];
                break;
            }
        }
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
                         int32_t* tile_counts,
                         int T,
                         int L,
                         int P,
                         int max_replicas,
                         cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    constexpr int TILE_T = REROUTE_FWD_TILE_T;
    constexpr int WPB = REROUTE_FWD_WARPS_PER_BLOCK;
    constexpr int TPB = REROUTE_FWD_THREADS_PER_BLOCK;

    const int num_tiles = (T + TILE_T - 1) / TILE_T;
    const int num_expert_blocks = (L + WPB - 1) / WPB;
    dim3 grid(num_expert_blocks, num_tiles);
    dim3 block(TPB);

    // Pass 1: count active tokens per (expert, tile)
    reroute_forward_count_kernel<TILE_T, WPB><<<grid, block, 0, stream>>>(routing_map, tile_counts, T, L, num_tiles);

    // Pass 2: prefix-sum + scatter (type-dispatched)
    reroute_forward_scatter_kernel<float, TILE_T, WPB, false>
        <<<grid, block, 0, stream>>>(routing_map,
                                     static_cast<const float*>(probs),
                                     l2p_map,
                                     lcnts,
                                     nullptr,
                                     tile_counts,
                                     expanded_routing_map,
                                     static_cast<float*>(expanded_probs),
                                     T,
                                     L,
                                     P,
                                     max_replicas,
                                     num_tiles);
}

void run_reroute_forward_quota(const bool* routing_map,
                               const void* probs,
                               const int32_t* l2p_map,
                               const int32_t* lcnts,
                               const int32_t* rank_quota_prefix,
                               bool* expanded_routing_map,
                               void* expanded_probs,
                               int32_t* tile_counts,
                               int T,
                               int L,
                               int P,
                               int max_replicas,
                               cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    constexpr int TILE_T = REROUTE_FWD_TILE_T;
    constexpr int WPB = REROUTE_FWD_WARPS_PER_BLOCK;
    constexpr int TPB = REROUTE_FWD_THREADS_PER_BLOCK;

    const int num_tiles = (T + TILE_T - 1) / TILE_T;
    const int num_expert_blocks = (L + WPB - 1) / WPB;
    dim3 grid(num_expert_blocks, num_tiles);
    dim3 block(TPB);

    reroute_forward_count_kernel<TILE_T, WPB><<<grid, block, 0, stream>>>(routing_map, tile_counts, T, L, num_tiles);

    reroute_forward_scatter_kernel<float, TILE_T, WPB, true>
        <<<grid, block, 0, stream>>>(routing_map,
                                     static_cast<const float*>(probs),
                                     l2p_map,
                                     lcnts,
                                     rank_quota_prefix,
                                     tile_counts,
                                     expanded_routing_map,
                                     static_cast<float*>(expanded_probs),
                                     T,
                                     L,
                                     P,
                                     max_replicas,
                                     num_tiles);
}

void run_reroute_backward(const void* grad_expanded_probs,
                          const bool* routing_map,
                          const bool* expanded_routing_map,
                          const int32_t* l2p_map,
                          const int32_t* lcnts,
                          void* grad_probs,
                          int T,
                          int L,
                          int P,
                          int max_replicas,
                          cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    // 2D block: x = expert dimension (up to 256), y = rows per block
    const int block_x = min(L, 256);
    const int block_y = min(REROUTE_BWD_ROWS_PER_BLOCK, 1024 / block_x);
    dim3 block(block_x, block_y);
    dim3 grid((T + block_y - 1) / block_y, (L + block_x - 1) / block_x);

    reroute_backward_gather_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float*>(grad_expanded_probs),
                                                                      routing_map,
                                                                      expanded_routing_map,
                                                                      l2p_map,
                                                                      lcnts,
                                                                      static_cast<float*>(grad_probs),
                                                                      T,
                                                                      L,
                                                                      P,
                                                                      max_replicas);
}

// ---------------------------------------------------------------------------
// reroute_sparse_kernel
//
// In-place remaps topk_ids from logical expert IDs to physical expert IDs
// using round-robin dispatch across replicas.
// Each thread handles one topk entry.  An atomicAdd on a per-expert counter
// determines the round-robin rank; the modulo selects the physical replica.
// ---------------------------------------------------------------------------

__global__ void reroute_sparse_kernel(int64_t* __restrict__ topk_ids,
                                      const int32_t* __restrict__ l2p_map,
                                      const int32_t* __restrict__ replica_counts,
                                      int* __restrict__ counters,
                                      const int num_entries,
                                      const int max_replicas,
                                      const int num_experts) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries; idx += gridDim.x * blockDim.x) {
        int64_t logical_id = topk_ids[idx];
        if (logical_id < 0 || logical_id >= num_experts)
            continue;

        int C = replica_counts[logical_id];
        if (C <= 0)
            continue;

        int rank = atomicAdd(&counters[logical_id], 1);
        int replica_idx = rank % C;
        int32_t physical_id = l2p_map[logical_id * max_replicas + replica_idx];
        topk_ids[idx] = static_cast<int64_t>(physical_id);
    }
}

void run_reroute_sparse(int64_t* topk_ids_ptr,
                        const int32_t* l2p_map_gpu,
                        const int32_t* lcnts_gpu,
                        int* counters_gpu,
                        const int num_tokens,
                        const int top_k,
                        const int num_global_logical_experts,
                        const int max_replicas,
                        cudaStream_t stream) {
    int num_entries = num_tokens * top_k;
    int L = num_global_logical_experts;

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(counters_gpu, 0, L * sizeof(int), stream));

    if (num_entries > 0) {
        constexpr int BLOCK = 256;
        int num_blocks = min(256, (num_entries + BLOCK - 1) / BLOCK);
        reroute_sparse_kernel<<<num_blocks, BLOCK, 0, stream>>>(
            topk_ids_ptr, l2p_map_gpu, lcnts_gpu, counters_gpu, num_entries, max_replicas, L);
    }
}

__global__ void reduce_per_rank_loads_kernel(const int32_t* __restrict__ loads_per_rank,
                                             int32_t* __restrict__ global_loads,
                                             int G,
                                             int L) {
    for (int l = blockIdx.x * blockDim.x + threadIdx.x; l < L; l += blockDim.x * gridDim.x) {
        int32_t total = 0;
        for (int r = 0; r < G; ++r) {
            total += loads_per_rank[r * L + l];
        }
        global_loads[l] = total;
    }
}

void reduce_per_rank_loads(const int32_t* loads_per_rank, int32_t* global_loads, int G, int L, cudaStream_t stream) {
    if (G <= 0 || L <= 0)
        return;
    constexpr int BLOCK = 256;
    int grid = min(1024, (L + BLOCK - 1) / BLOCK);
    reduce_per_rank_loads_kernel<<<grid, BLOCK, 0, stream>>>(loads_per_rank, global_loads, G, L);
}

}  // namespace ultra_ep::kernels
