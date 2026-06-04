#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "api.cuh"
#include "config.cuh"
#include "launch.cuh"

namespace ultra_ep::kernels {

__device__ __forceinline__ int gcd_int(int a, int b) {
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// ============================================================================
// Forward pass 1: count active tokens per (expert, tile)
// ============================================================================
template <int TILE_T, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK *
                             32) void reroute_forward_count_kernel(const bool* __restrict__ routing_map,
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
template <typename scalar_t, int TILE_T, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32) void dense_rr_reroute_scatter_kernel(
    const bool* __restrict__ routing_map,
    const scalar_t* __restrict__ probs,
    const int32_t* __restrict__ logical_to_physical_map,
    const int32_t* __restrict__ logical_replica_counts,
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

    const int C = logical_replica_counts[expert_id];
    constexpr int PREFETCH_REPLICAS = 8;
    int local_l2p[PREFETCH_REPLICAS];
    const int prefetch_count = min(C, PREFETCH_REPLICAS);

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        local_l2p[j] = -1;
    }

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        if (j < prefetch_count) {
            local_l2p[j] = logical_to_physical_map[expert_id * max_replicas + j];
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
            const int replica_idx = my_rank % C;
            const int phys = (replica_idx < PREFETCH_REPLICAS)
                ? local_l2p[replica_idx]
                : logical_to_physical_map[expert_id * max_replicas + replica_idx];
            expanded_routing_map[t * P + phys] = true;
            expanded_probs[t * P + phys] = probs[t * L + expert_id];
        }

        counter += total_active;
    }
}

template <typename scalar_t, int TILE_T, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32) void dense_quota_reroute_scatter_kernel(
    const bool* __restrict__ routing_map,
    const scalar_t* __restrict__ probs,
    const int32_t* __restrict__ logical_to_physical_map,
    const int32_t* __restrict__ logical_replica_counts,
    const int32_t* __restrict__ rank_quota_prefix,
    const int32_t* __restrict__ tile_counts,
    bool* __restrict__ expanded_routing_map,
    scalar_t* __restrict__ expanded_probs,
    const int T,
    const int L,
    const int P,
    const int max_replicas,
    const int num_tiles,
    const bool interleave_by_rank_quota) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int expert_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int tile_id = blockIdx.y;

    if (expert_id >= L)
        return;

    const int C = logical_replica_counts[expert_id];
    constexpr int PREFETCH_REPLICAS = 8;
    int local_prefix[PREFETCH_REPLICAS];
    int local_l2p[PREFETCH_REPLICAS];
    const int prefetch_count = min(C, PREFETCH_REPLICAS);

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        local_prefix[j] = 0;
        local_l2p[j] = -1;
    }

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        if (j < prefetch_count) {
            local_prefix[j] = rank_quota_prefix[expert_id * max_replicas + j];
            local_l2p[j] = logical_to_physical_map[expert_id * max_replicas + j];
        }
    }

    int quota_local_total = 0;
    int quota_perm_stride = 1;
    int quota_perm_offset = 0;
    if (C > 0) {
        quota_local_total =
            (C <= PREFETCH_REPLICAS) ? local_prefix[C - 1] : rank_quota_prefix[expert_id * max_replicas + C - 1];
    }
    if (interleave_by_rank_quota && quota_local_total > 1) {
        if (lane == 0) {
            int stride = quota_local_total / 2 + 1;
            if (stride >= quota_local_total) {
                stride = quota_local_total - 1;
            }
            if (stride < 1) {
                stride = 1;
            }
            while (gcd_int(stride, quota_local_total) != 1) {
                ++stride;
                if (stride >= quota_local_total) {
                    stride = 1;
                }
            }
            quota_perm_stride = stride;
            quota_perm_offset = expert_id % quota_local_total;
        }
        quota_perm_stride = __shfl_sync(0xFFFFFFFF, quota_perm_stride, 0);
        quota_perm_offset = __shfl_sync(0xFFFFFFFF, quota_perm_offset, 0);
    }

    int base_rank = 0;
    const int32_t* my_tile_counts = tile_counts + expert_id * num_tiles;
    for (int base = 0; base < tile_id; base += 32) {
        const int idx = base + lane;
        int val = (idx < tile_id) ? my_tile_counts[idx] : 0;
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
            int quota_rank = my_rank;
            if (interleave_by_rank_quota && quota_local_total > 1) {
                quota_rank = static_cast<int>((static_cast<int64_t>(my_rank) * quota_perm_stride + quota_perm_offset) %
                                              quota_local_total);
            }
            int replica_idx = 0;
#pragma unroll
            for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
                if (j < prefetch_count) {
                    replica_idx += (quota_rank >= local_prefix[j]) ? 1 : 0;
                }
            }
            for (int j = PREFETCH_REPLICAS; j < C; ++j) {
                replica_idx += (quota_rank >= rank_quota_prefix[expert_id * max_replicas + j]) ? 1 : 0;
            }
            replica_idx = min(replica_idx, max(C - 1, 0));
            const int phys = (replica_idx < PREFETCH_REPLICAS)
                ? local_l2p[replica_idx]
                : logical_to_physical_map[expert_id * max_replicas + replica_idx];
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
__global__ __launch_bounds__(1024) void reroute_backward_gather_kernel(const scalar_t* __restrict__ grad_expanded_probs,
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

void run_dense_reroute_forward_round_robin(const bool* routing_map,
                                           const void* probs,
                                           const int32_t* logical_to_physical_map,
                                           const int32_t* logical_replica_counts,
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

    constexpr int TILE_T = kDenseRerouteTileTokens;
    constexpr int WPB = kDenseRerouteWarpsPerBlock;
    constexpr int TPB = kDenseRerouteThreadsPerBlock;

    const int num_tiles = (T + TILE_T - 1) / TILE_T;
    const int num_expert_blocks = (L + WPB - 1) / WPB;
    const auto config = make_launch_config(dim3(num_expert_blocks, num_tiles), dim3(TPB), stream);

    // Pass 1: count active tokens per (expert, tile)
    launch_kernel(reroute_forward_count_kernel<TILE_T, WPB>, config, routing_map, tile_counts, T, L, num_tiles);

    // Pass 2: prefix-sum + scatter (type-dispatched)
    launch_kernel(dense_rr_reroute_scatter_kernel<float, TILE_T, WPB>,
                  config,
                  routing_map,
                  static_cast<const float*>(probs),
                  logical_to_physical_map,
                  logical_replica_counts,
                  tile_counts,
                  expanded_routing_map,
                  static_cast<float*>(expanded_probs),
                  T,
                  L,
                  P,
                  max_replicas,
                  num_tiles);
}

void run_dense_reroute_forward_quota(const bool* routing_map,
                                     const void* probs,
                                     const int32_t* logical_to_physical_map,
                                     const int32_t* logical_replica_counts,
                                     const int32_t* rank_quota_prefix,
                                     bool* expanded_routing_map,
                                     void* expanded_probs,
                                     int32_t* tile_counts,
                                     int T,
                                     int L,
                                     int P,
                                     int max_replicas,
                                     bool interleave_by_rank_quota,
                                     cudaStream_t stream) {
    if (T == 0 || L == 0)
        return;

    constexpr int TILE_T = kDenseRerouteTileTokens;
    constexpr int WPB = kDenseRerouteWarpsPerBlock;
    constexpr int TPB = kDenseRerouteThreadsPerBlock;

    const int num_tiles = (T + TILE_T - 1) / TILE_T;
    const int num_expert_blocks = (L + WPB - 1) / WPB;
    const auto config = make_launch_config(dim3(num_expert_blocks, num_tiles), dim3(TPB), stream);

    launch_kernel(reroute_forward_count_kernel<TILE_T, WPB>, config, routing_map, tile_counts, T, L, num_tiles);

    launch_kernel(dense_quota_reroute_scatter_kernel<float, TILE_T, WPB>,
                  config,
                  routing_map,
                  static_cast<const float*>(probs),
                  logical_to_physical_map,
                  logical_replica_counts,
                  rank_quota_prefix,
                  tile_counts,
                  expanded_routing_map,
                  static_cast<float*>(expanded_probs),
                  T,
                  L,
                  P,
                  max_replicas,
                  num_tiles,
                  interleave_by_rank_quota);
}

void run_dense_reroute_backward(const void* grad_expanded_probs,
                                const bool* routing_map,
                                const bool* expanded_routing_map,
                                const int32_t* logical_to_physical_map,
                                const int32_t* logical_replica_counts,
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
    const int block_y = min(kDenseRerouteBackwardRowsPerBlock, 1024 / block_x);
    const auto config = make_launch_config(
        dim3((T + block_y - 1) / block_y, (L + block_x - 1) / block_x), dim3(block_x, block_y), stream);

    launch_kernel(reroute_backward_gather_kernel<float>,
                  config,
                  static_cast<const float*>(grad_expanded_probs),
                  routing_map,
                  expanded_routing_map,
                  logical_to_physical_map,
                  logical_replica_counts,
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
// using the current placement. The per-expert local ordinal comes from a
// single atomicAdd counter:
//   - non-quota mode uses round-robin across replicas
//   - quota mode uses rank_quota_prefix to select the replica
// Each thread handles one topk entry.  An atomicAdd on a per-expert counter
// determines the local ordinal with no extra preprocessing or synchronization.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int sparse_rr_replica_index(const int local_ordinal, const int replica_count) {
    return local_ordinal % replica_count;
}

__device__ __forceinline__ int sparse_quota_replica_index(const int local_ordinal,
                                                          const int logical_id,
                                                          const int replica_count,
                                                          const int max_replicas,
                                                          const int32_t* __restrict__ rank_quota_prefix) {
    constexpr int PREFETCH_REPLICAS = 8;
    const int row_offset = logical_id * max_replicas;
    const int prefetch_count = min(replica_count, PREFETCH_REPLICAS);
    int replica_idx = 0;
    int local_prefix[PREFETCH_REPLICAS];

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        local_prefix[j] = (j < prefetch_count) ? rank_quota_prefix[row_offset + j] : 0;
    }

#pragma unroll
    for (int j = 0; j < PREFETCH_REPLICAS; ++j) {
        if (j < prefetch_count) {
            replica_idx += (local_ordinal >= local_prefix[j]) ? 1 : 0;
        }
    }
    for (int j = PREFETCH_REPLICAS; j < replica_count; ++j) {
        replica_idx += (local_ordinal >= rank_quota_prefix[row_offset + j]) ? 1 : 0;
    }
    return min(replica_idx, max(replica_count - 1, 0));
}

__global__ __launch_bounds__(256) void sparse_rr_reroute_kernel(int64_t* __restrict__ topk_ids,
                                                                const int32_t* __restrict__ logical_to_physical_map,
                                                                const int32_t* __restrict__ logical_replica_counts,
                                                                int* __restrict__ counters,
                                                                const int num_entries,
                                                                const int max_replicas,
                                                                const int num_experts) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries; idx += gridDim.x * blockDim.x) {
        int64_t logical_id = topk_ids[idx];
        if (logical_id < 0 || logical_id >= num_experts)
            continue;

        int C = logical_replica_counts[logical_id];
        if (C <= 0)
            continue;

        int local_ordinal = atomicAdd(&counters[logical_id], 1);
        int replica_idx = sparse_rr_replica_index(local_ordinal, C);
        int32_t physical_id = logical_to_physical_map[logical_id * max_replicas + replica_idx];
        topk_ids[idx] = static_cast<int64_t>(physical_id);
    }
}

__global__ __launch_bounds__(256) void sparse_quota_reroute_kernel(int64_t* __restrict__ topk_ids,
                                                                   const int32_t* __restrict__ logical_to_physical_map,
                                                                   const int32_t* __restrict__ logical_replica_counts,
                                                                   const int32_t* __restrict__ rank_quota_prefix,
                                                                   int* __restrict__ counters,
                                                                   const int num_entries,
                                                                   const int max_replicas,
                                                                   const int num_experts) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries; idx += gridDim.x * blockDim.x) {
        int64_t logical_id = topk_ids[idx];
        if (logical_id < 0 || logical_id >= num_experts)
            continue;

        int C = logical_replica_counts[logical_id];
        if (C <= 0)
            continue;

        int local_ordinal = atomicAdd(&counters[logical_id], 1);
        int replica_idx =
            sparse_quota_replica_index(local_ordinal, static_cast<int>(logical_id), C, max_replicas, rank_quota_prefix);
        int32_t physical_id = logical_to_physical_map[logical_id * max_replicas + replica_idx];
        topk_ids[idx] = static_cast<int64_t>(physical_id);
    }
}

void run_sparse_reroute_round_robin(int64_t* topk_ids_ptr,
                                    const int32_t* logical_to_physical_map,
                                    const int32_t* logical_replica_counts,
                                    int* counters,
                                    const int num_tokens,
                                    const int top_k,
                                    const int num_global_logical_experts,
                                    const int max_replicas,
                                    cudaStream_t stream) {
    int num_entries = num_tokens * top_k;
    int L = num_global_logical_experts;

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(counters, 0, L * sizeof(int), stream));

    if (num_entries > 0) {
        constexpr int BLOCK = 256;
        int num_blocks = min(256, (num_entries + BLOCK - 1) / BLOCK);
        const auto config = make_launch_config(dim3(num_blocks), dim3(BLOCK), stream);
        launch_kernel(sparse_rr_reroute_kernel,
                      config,
                      topk_ids_ptr,
                      logical_to_physical_map,
                      logical_replica_counts,
                      counters,
                      num_entries,
                      max_replicas,
                      L);
    }
}

void run_sparse_reroute_quota(int64_t* topk_ids_ptr,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int* counters,
                              const int num_tokens,
                              const int top_k,
                              const int num_global_logical_experts,
                              const int max_replicas,
                              cudaStream_t stream) {
    int num_entries = num_tokens * top_k;
    int L = num_global_logical_experts;

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(counters, 0, L * sizeof(int), stream));

    if (num_entries > 0) {
        constexpr int BLOCK = 256;
        int num_blocks = min(256, (num_entries + BLOCK - 1) / BLOCK);
        const auto config = make_launch_config(dim3(num_blocks), dim3(BLOCK), stream);
        launch_kernel(sparse_quota_reroute_kernel,
                      config,
                      topk_ids_ptr,
                      logical_to_physical_map,
                      logical_replica_counts,
                      rank_quota_prefix,
                      counters,
                      num_entries,
                      max_replicas,
                      L);
    }
}

}  // namespace ultra_ep::kernels
