/**
 * placement_gpu.cu — GPU-based expert placement solver V3 (PlacementSolverGPU).
 *
 * V3 optimizations (NCU data-driven):
 *   P0 — 2-Warp cooperative kernel (64 threads = 2 warps per block)
 *         Hides shuffle latency (Stall Wait 45%) and smem latency (Short Scoreboard 22%)
 *   P1 — Phase C 2-warp parallel argmin (each warp handles 32 GPUs)
 *         Reduces shuffle count per round from 11 to 6
 *   P2 — Template specialization on EPL and COMPACT_EOR
 *         Eliminates runtime branches (Branch Resolving 14%)
 *         Union EOR saves ~5KB shared memory
 *   External memset — p2l/l2p/lcnts initialization moved to cudaMemsetAsync
 *
 * Kernel launch: one block per NVL domain, 2 warps (64 threads) per block for
 * COMPACT_EOR=true (G <= 64), 1 warp (32 threads) for G > 64 fallback.
 *
 * Determinism: same expert_loads → identical p2l/l2p/lcnts on every rank.
 */

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "../kernels/launch.cuh"
#include "../utils/exception.cuh"
#include "api.hpp"

namespace ultra_ep::solver {

// ============================================================================
// Shared memory sizes (compile-time upper bounds)
// ============================================================================

static constexpr int MAX_EXPERTS_PER_NVL = 512;
static constexpr int MAX_GPUS_PER_NVL = 72;
static constexpr int MAX_REPLICAS_PER_NVL = 512;
static constexpr float kInfLoad = 1e30f;

struct ReplicaEntry {
    int logical_id;
    float load_per_replica;
};

// ============================================================================
// Warp-level utilities
// ============================================================================

__device__ __forceinline__ int warp_reduce_sum(int val) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, s);
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_max(int val) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, s));
    }
    return val;
}

__device__ __forceinline__ int warp_exclusive_sum(int val) {
    const int lane = threadIdx.x & 31;
    int inclusive = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int peer = __shfl_up_sync(0xFFFFFFFF, inclusive, offset);
        if (lane >= offset) {
            inclusive += peer;
        }
    }
    return inclusive - val;
}

// Warp-reduce to find the minimum float. Masked-out lanes are treated as +INF.
__device__ __forceinline__ float warp_reduce_argmin(float val, bool mask, int& out_lane) {
    float v = mask ? val : kInfLoad;
    int lane = threadIdx.x & 31;

#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float peer_v = __shfl_xor_sync(0xFFFFFFFF, v, s);
        int peer_lane = __shfl_xor_sync(0xFFFFFFFF, lane, s);
        if (peer_v < v || (peer_v == v && peer_lane < lane)) {
            v = peer_v;
            lane = peer_lane;
        }
    }
    out_lane = lane;
    return v;
}

__device__ __forceinline__ float ceil_div_f(int numerator, int denominator) {
    return static_cast<float>((numerator + denominator - 1) / denominator);
}

__device__ __forceinline__ bool ratio_greater(int load_a, int denom_a, int idx_a, int load_b, int denom_b, int idx_b) {
    long long lhs = static_cast<long long>(load_a) * denom_b;
    long long rhs = static_cast<long long>(load_b) * denom_a;
    if (lhs != rhs) {
        return lhs > rhs;
    }
    return idx_a < idx_b;
}

// Reverse-greedy fixup: remove the smallest last-replica score first.
// On exact ties, remove the larger expert index first so lower indices keep precedence.
__device__ __forceinline__ bool ratio_less_for_remove(
    int load_a, int denom_a, int idx_a, int load_b, int denom_b, int idx_b) {
    long long lhs = static_cast<long long>(load_a) * denom_b;
    long long rhs = static_cast<long long>(load_b) * denom_a;
    if (lhs != rhs) {
        return lhs < rhs;
    }
    return idx_a > idx_b;
}

__device__ __forceinline__ bool eor_is_set(const uint32_t* bitmap, int E, int g, int l_local) {
    int bit_idx = g * E + l_local;
    return (bitmap[bit_idx / 32] >> (bit_idx % 32)) & 1u;
}

__device__ __forceinline__ void eor_mark(uint32_t* bitmap, int E, int g, int l_local) {
    int bit_idx = g * E + l_local;
    bitmap[bit_idx / 32] |= (1u << (bit_idx % 32));
}

// ============================================================================
// Bitonic sort on shared-memory ReplicaEntry array (descending by load_per_replica,
// ascending logical_id on ties). n_padded must be a power of 2.
// ============================================================================

__device__ __forceinline__ bool replica_gt(const ReplicaEntry& a, const ReplicaEntry& b) {
    if (a.load_per_replica != b.load_per_replica) {
        return a.load_per_replica > b.load_per_replica;
    }
    return a.logical_id < b.logical_id;
}

// Bitonic sort using N_THREADS threads (32 or 64).
template <int N_THREADS>
__device__ void bitonic_sort_replicas(ReplicaEntry* arr, int n_padded) {
    const int tid = threadIdx.x;
    for (int k = 2; k <= n_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n_padded; i += N_THREADS) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    ReplicaEntry ai = arr[i];
                    ReplicaEntry aj = arr[ixj];
                    bool should_swap = ascending ? !replica_gt(ai, aj) : replica_gt(ai, aj);
                    if (should_swap) {
                        arr[i] = aj;
                        arr[ixj] = ai;
                    }
                }
            }
            if constexpr (N_THREADS <= 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// V3 kernel: COMPACT_EOR=true path — 2 warps (64 threads), G <= 64
//
// Template parameters:
//   EPL          — experts per lane = ceil(E / 32), compile-time constant
//   COMPACT_EOR  — true: G <= 64, uses uint64 compact EOR + 2-warp cooperative argmin
//                  false: G > 64, uses packed uint32 EOR bitmap + 1-warp smem path
// ============================================================================

template <int EPL, bool COMPACT_EOR>
__global__ __launch_bounds__(64) void placement_solve_kernel_v3(const int32_t* __restrict__ expert_loads,
                                                                     int32_t* __restrict__ p2l_map,
                                                                     int32_t* __restrict__ l2p_map,
                                                                     int32_t* __restrict__ lcnts,
                                                                     int num_nvl_ranks,
                                                                     int num_local_master,
                                                                     int num_local_redundant,
                                                                     int num_local_physical,
                                                                     int max_replicas_dim,
                                                                     int num_logical_per_nvl,
                                                                     int num_redundant_per_nvl,
                                                                     int num_global_physical,
                                                                     float balance_threshold) {
    const int domain = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;

    const int domain_start_rank = domain * num_nvl_ranks;
    const int domain_start_log = domain_start_rank * num_local_master;

    const int E = num_logical_per_nvl;
    const int G = num_nvl_ranks;
    const int B = num_redundant_per_nvl;

    // Number of threads in this kernel
    constexpr int N_THREADS = COMPACT_EOR ? 64 : 32;

    // ---- Shared memory ----
    __shared__ int32_t smem_loads[MAX_EXPERTS_PER_NVL];
    __shared__ int32_t smem_c[MAX_EXPERTS_PER_NVL];
    __shared__ ReplicaEntry smem_replicas[MAX_REPLICAS_PER_NVL];
    __shared__ int smem_l2p_slot[MAX_EXPERTS_PER_NVL];
    __shared__ int smem_replica_count;

    // Union: compact EOR (G <= 64) and packed EOR (G > 64) are mutually exclusive
    union EorUnion {
        uint64_t compact[MAX_EXPERTS_PER_NVL];                                  // COMPACT_EOR=true
        uint32_t packed[((MAX_GPUS_PER_NVL * MAX_EXPERTS_PER_NVL) + 31) / 32];  // COMPACT_EOR=false
    };
    __shared__ EorUnion smem_eor_union;

    // For non-compact path: gpu load/slots in smem (reuse union for load, separate for slots)
    __shared__ float smem_gpu_load_nc[COMPACT_EOR ? 1 : MAX_GPUS_PER_NVL];
    __shared__ int smem_gpu_slots_nc[COMPACT_EOR ? 1 : MAX_GPUS_PER_NVL];

    // For 2-warp compact path: warp result exchange
    struct WarpResult {
        float min_val;
        int winner_gpu;
        int winner_slot;
    };
    __shared__ WarpResult smem_warp_result[COMPACT_EOR ? 2 : 1];

    // =========================================================================
    // Phase 0: Place masters + initialize smem
    // External cudaMemsetAsync already initialized p2l/l2p/lcnts global memory,
    // so we only need to write master placements and init smem structures.
    // =========================================================================

    int reg_load[EPL];
    int reg_c[EPL];
    int reg_extra_hi[EPL];
    int local_max = 0;

    if constexpr (COMPACT_EOR) {
        // 2-warp cooperative Phase 0
        if (warp_id == 0) {
            // Warp 0: load expert_loads → registers, write master placements
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                int i = lane + k * 32;
                int load = 0;
                if (i < E) {
                    int l = domain_start_log + i;
                    load = expert_loads[l];
                    smem_loads[i] = load;
                    local_max = max(local_max, load);

                    // Master placement (global writes)
                    int global_rank = l / num_local_master;
                    int local_idx = l % num_local_master;
                    int phys_idx = global_rank * num_local_physical + local_idx;
                    p2l_map[phys_idx] = l;
                    l2p_map[l * max_replicas_dim + 0] = phys_idx;
                    lcnts[l] = 1;
                    smem_l2p_slot[i] = 1;
                }
                reg_load[k] = load;
                reg_c[k] = 1;
                reg_extra_hi[k] = 0;
            }
        } else {
            // Warp 1: initialize smem data structures (smem_l2p_slot done by warp 0)
            for (int i = lane; i < E; i += 32) {
                smem_c[i] = 1;
            }
            // Initialize reg arrays (will reload from smem_loads after sync)
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                reg_load[k] = 0;
                reg_c[k] = 1;
                reg_extra_hi[k] = 0;
            }
        }
        __syncthreads();

        // Warp 1 loads expert loads from smem (warp 0 already wrote them)
        if (warp_id == 1) {
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                int i = lane + k * 32;
                int load = (i < E) ? smem_loads[i] : 0;
                reg_load[k] = load;
                local_max = max(local_max, load);
            }
        }
    } else {
        // Single-warp Phase 0 (G > 64 fallback)
        // smem_c init
        for (int i = lane; i < E; i += 32) {
            smem_c[i] = 1;
        }
        __syncwarp();

#pragma unroll
        for (int k = 0; k < EPL; ++k) {
            int i = lane + k * 32;
            int load = 0;
            if (i < E) {
                int l = domain_start_log + i;
                load = expert_loads[l];
                smem_loads[i] = load;
                local_max = max(local_max, load);

                int global_rank = l / num_local_master;
                int local_idx = l % num_local_master;
                int phys_idx = global_rank * num_local_physical + local_idx;
                p2l_map[phys_idx] = l;
                l2p_map[l * max_replicas_dim + 0] = phys_idx;
                lcnts[l] = 1;
                smem_l2p_slot[i] = 1;
            }
            reg_load[k] = load;
            reg_c[k] = 1;
            reg_extra_hi[k] = 0;
        }
        __syncwarp();
    }

    if (num_local_redundant == 0 || num_nvl_ranks <= 1) {
        return;
    }

    // =========================================================================
    // Phase A: binary search on integer threshold T
    // Both warps execute redundantly (same input → same result).
    // Only warp 0 writes to smem_c in the fixup.
    // =========================================================================

    int max_load = warp_reduce_max(local_max);
    if (max_load == 0) {
        if (lane == 0 && warp_id == 0) {
            int remaining = B;
            for (int i = 0; i < E && remaining > 0; ++i) {
                int add = min(remaining, G - 1);
                smem_c[i] = 1 + add;
                remaining -= add;
            }
        }
        if constexpr (COMPACT_EOR) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    } else {
        // Early-stop: compute effective_B based on balance_threshold
        int effective_B = B;
        if (balance_threshold > 1.0f) {
            int total_load_local = 0;
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                int i = lane + k * 32;
                if (i < E)
                    total_load_local += reg_load[k];
            }
            int total_load = warp_reduce_sum(total_load_local);
            float avg_per_slot = __int2float_rn(total_load) / (G * num_local_master);
            float target_score = avg_per_slot * balance_threshold;

            if (avg_per_slot > 0.0f) {
                // Compute minimum replicas needed so each expert's LPR <= target_score
                int needed_replicas_local = 0;
#pragma unroll
                for (int k = 0; k < EPL; ++k) {
                    int i = lane + k * 32;
                    if (i < E && reg_load[k] > 0) {
                        int needed_c = max(1, __float2int_ru(__int2float_rn(reg_load[k]) / target_score));
                        needed_c = min(needed_c, G);
                        needed_replicas_local += needed_c - 1;
                    }
                }
                int needed_total = warp_reduce_sum(needed_replicas_local);
                effective_B = min(B, needed_total);
            }
        }

        int lo = 1;
        int hi = max_load + 1;  // f(hi) == 0

        while (lo + 1 < hi) {
            int mid = lo + ((hi - lo) >> 1);
            int local_sum = 0;
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                int i = lane + k * 32;
                if (i < E) {
                    local_sum += min(reg_load[k] / mid, G - 1);
                }
            }
            int total = warp_reduce_sum(local_sum);
            if (total >= effective_B) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        int local_total = 0;
        int local_boundary = 0;
#pragma unroll
        for (int k = 0; k < EPL; ++k) {
            int i = lane + k * 32;
            if (i < E) {
                int extra_lo = min(reg_load[k] / lo, G - 1);
                int extra_hi = min(reg_load[k] / hi, G - 1);
                reg_c[k] = extra_lo + 1;
                reg_extra_hi[k] = extra_hi;
                local_total += extra_lo;
                local_boundary += (extra_lo > extra_hi) ? 1 : 0;
            } else {
                reg_c[k] = 1;
                reg_extra_hi[k] = 0;
            }
        }

        int total_replicas = warp_reduce_sum(local_total);
        int surplus = total_replicas - effective_B;
        if (surplus > 0) {
            int boundary_total = warp_reduce_sum(local_boundary);
            int quick_remove = min(surplus, boundary_total);
            int removed_prefix = 0;

#pragma unroll
            for (int k = 0; k < EPL && removed_prefix < quick_remove; ++k) {
                int i = lane + k * 32;
                int removable = (i < E && (reg_c[k] - 1) > reg_extra_hi[k]) ? 1 : 0;
                int rank = warp_exclusive_sum(removable);
                int count = warp_reduce_sum(removable);
                if (removable && removed_prefix + rank < quick_remove) {
                    reg_c[k]--;
                }
                removed_prefix += count;
            }
            total_replicas -= quick_remove;
        }

        // Only warp 0 writes smem_c (both warps computed same result)
        if constexpr (COMPACT_EOR) {
            if (warp_id == 0) {
#pragma unroll
                for (int k = 0; k < EPL; ++k) {
                    int i = lane + k * 32;
                    if (i < E) {
                        smem_c[i] = reg_c[k];
                    }
                }
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int k = 0; k < EPL; ++k) {
                int i = lane + k * 32;
                if (i < E) {
                    smem_c[i] = reg_c[k];
                }
            }
            __syncwarp();
        }

        // Serial fixup: only thread 0 of warp 0
        if (lane == 0 && warp_id == 0) {
            if (total_replicas < effective_B) {
                int deficit = effective_B - total_replicas;
                while (deficit > 0) {
                    int best_idx = -1;
                    for (int i = 0; i < E; ++i) {
                        if (smem_c[i] >= G) {
                            continue;
                        }
                        if (best_idx < 0 ||
                            ratio_greater(
                                smem_loads[i], smem_c[i], i, smem_loads[best_idx], smem_c[best_idx], best_idx)) {
                            best_idx = i;
                        }
                    }
                    if (best_idx < 0) {
                        break;
                    }
                    smem_c[best_idx]++;
                    deficit--;
                }
            } else if (total_replicas > effective_B) {
                int remove_surplus = total_replicas - effective_B;
                while (remove_surplus > 0) {
                    int best_idx = -1;
                    for (int i = 0; i < E; ++i) {
                        if (smem_c[i] <= 1) {
                            continue;
                        }
                        if (best_idx < 0 ||
                            ratio_less_for_remove(smem_loads[i],
                                                  smem_c[i] - 1,
                                                  i,
                                                  smem_loads[best_idx],
                                                  smem_c[best_idx] - 1,
                                                  best_idx)) {
                            best_idx = i;
                        }
                    }
                    if (best_idx < 0) {
                        break;
                    }
                    smem_c[best_idx]--;
                    remove_surplus--;
                }
            }
        }
        if constexpr (COMPACT_EOR) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    // Count replicas (all threads participate; for 2-warp, only warp 0's result is used)
    int local_extra = 0;
    for (int i = lane; i < E; i += 32) {
        local_extra += smem_c[i] - 1;
    }
    int replica_count = warp_reduce_sum(local_extra);
    if (warp_id == 0 && lane == 0) {
        smem_replica_count = replica_count;
    }
    if constexpr (COMPACT_EOR) {
        __syncthreads();
    } else {
        __syncwarp();
    }

    replica_count = smem_replica_count;
    if (replica_count == 0) {
        if constexpr (COMPACT_EOR) {
            for (int i = tid; i < E; i += 64) {
                lcnts[domain_start_log + i] = smem_c[i];
            }
        } else {
            for (int i = lane; i < E; i += 32) {
                lcnts[domain_start_log + i] = smem_c[i];
            }
        }
        return;
    }

    // =========================================================================
    // Phase B: Build replica list + bitonic sort
    // For 2-warp: only warp 0 builds (uses warp_exclusive_sum), then all sort.
    // =========================================================================

    if constexpr (COMPACT_EOR) {
        // Warp 0 builds the replica list
        if (warp_id == 0) {
            int lane_offset = warp_exclusive_sum(local_extra);
            int write_idx = lane_offset;
            for (int i = lane; i < E; i += 32) {
                int num_extra = smem_c[i] - 1;
                if (num_extra <= 0)
                    continue;
                float lpr = ceil_div_f(smem_loads[i], smem_c[i]);
                for (int j = 0; j < num_extra; ++j) {
                    smem_replicas[write_idx++] = {i, lpr};
                }
            }
        }
        __syncthreads();
    } else {
        int lane_offset = warp_exclusive_sum(local_extra);
        int write_idx = lane_offset;
        for (int i = lane; i < E; i += 32) {
            int num_extra = smem_c[i] - 1;
            if (num_extra <= 0)
                continue;
            float lpr = ceil_div_f(smem_loads[i], smem_c[i]);
            for (int j = 0; j < num_extra; ++j) {
                smem_replicas[write_idx++] = {i, lpr};
            }
        }
        __syncwarp();
    }

    int n_padded = 1;
    while (n_padded < replica_count) {
        n_padded <<= 1;
    }
    // Pad with sentinel entries (all threads help)
    for (int i = replica_count + tid; i < n_padded; i += N_THREADS) {
        smem_replicas[i] = {INT_MAX, -kInfLoad};
    }
    if constexpr (COMPACT_EOR) {
        __syncthreads();
    } else {
        __syncwarp();
    }

    // Sort using all available threads
    bitonic_sort_replicas<N_THREADS>(smem_replicas, n_padded);
    if constexpr (COMPACT_EOR) {
        __syncthreads();
    } else {
        __syncwarp();
    }

    // =========================================================================
    // Phase C: Greedy bin-packing
    // =========================================================================

    if constexpr (COMPACT_EOR) {
        // ---- 2-Warp cooperative Phase C (G <= 64) ----
        // Warp 0 manages GPU[0..31], Warp 1 manages GPU[32..63]

        // Initialize compact EOR masks
        for (int i = tid; i < E; i += 64) {
            int g = i / num_local_master;
            smem_eor_union.compact[i] = (1ULL << g);
        }
        __syncthreads();

        // Each warp initializes its 32 GPU registers
        float my_gpu_load = kInfLoad;
        int my_gpu_slots = num_local_redundant;  // saturated = no more slots

        int my_gpu = warp_id * 32 + lane;
        if (my_gpu < G) {
            float load = 0.0f;
            for (int m = 0; m < num_local_master; ++m) {
                int l_local = my_gpu * num_local_master + m;
                if (l_local < E) {
                    load += ceil_div_f(smem_loads[l_local], smem_c[l_local]);
                }
            }
            my_gpu_load = load;
            my_gpu_slots = 0;
        }
        __syncthreads();

        for (int r = 0; r < replica_count; ++r) {
            int l_local = smem_replicas[r].logical_id;
            float lpr = smem_replicas[r].load_per_replica;

            uint64_t expert_mask = smem_eor_union.compact[l_local];

            // Each warp independently does argmin over its 32 GPUs
            int my_gpu_idx = warp_id * 32 + lane;
            bool valid =
                (my_gpu_idx < G) && (my_gpu_slots < num_local_redundant) && ((expert_mask & (1ULL << my_gpu_idx)) == 0);

            int winner = -1;
            float min_val = warp_reduce_argmin(my_gpu_load, valid, winner);
            int winner_slot = __shfl_sync(0xFFFFFFFF, my_gpu_slots, winner);

            // Lane 0 of each warp writes result to smem
            if (lane == 0) {
                bool has_valid = (min_val < kInfLoad * 0.5f);
                smem_warp_result[warp_id].min_val = has_valid ? min_val : kInfLoad;
                smem_warp_result[warp_id].winner_gpu = has_valid ? (warp_id * 32 + winner) : -1;
                smem_warp_result[warp_id].winner_slot = has_valid ? winner_slot : -1;
            }
            __syncthreads();

            // All threads read both warp results and select the best
            float lo_val = smem_warp_result[0].min_val;
            int lo_gpu = smem_warp_result[0].winner_gpu;
            int lo_slot = smem_warp_result[0].winner_slot;
            float hi_val = smem_warp_result[1].min_val;
            int hi_gpu = smem_warp_result[1].winner_gpu;
            int hi_slot = smem_warp_result[1].winner_slot;

            int best_gpu, best_slot;
            bool use_lo = (lo_gpu >= 0) && (hi_gpu < 0 || lo_val < hi_val || (lo_val == hi_val && lo_gpu < hi_gpu));
            if (use_lo) {
                best_gpu = lo_gpu;
                best_slot = lo_slot;
            } else if (hi_gpu >= 0) {
                best_gpu = hi_gpu;
                best_slot = hi_slot;
            } else {
                best_gpu = -1;
                best_slot = -1;
            }

            // Thread 0 (warp 0, lane 0) does global + smem writes
            if (tid == 0) {
                if (best_gpu >= 0) {
                    int global_rank = domain_start_rank + best_gpu;
                    int phys_idx = global_rank * num_local_physical + num_local_master + best_slot;
                    int l_global = domain_start_log + l_local;

                    p2l_map[phys_idx] = l_global;
                    int l2p_slot = smem_l2p_slot[l_local];
                    l2p_map[l_global * max_replicas_dim + l2p_slot] = phys_idx;
                    smem_l2p_slot[l_local] = l2p_slot + 1;
                    smem_eor_union.compact[l_local] |= (1ULL << best_gpu);
                } else {
                    smem_c[l_local]--;
                }
            }

            // Winner GPU's owning warp+lane updates registers
            if (best_gpu >= 0) {
                int best_warp = best_gpu / 32;
                int best_lane = best_gpu % 32;
                if (warp_id == best_warp && lane == best_lane) {
                    my_gpu_load += lpr;
                    my_gpu_slots++;
                }
            }
            __syncthreads();
        }
    } else {
        // ---- Single-warp Phase C (G > 64 fallback) ----
        // Uses smem for GPU load/slots, packed uint32 EOR bitmap

        for (int g = lane; g < G; g += 32) {
            float load = 0.0f;
            for (int m = 0; m < num_local_master; ++m) {
                int l_local = g * num_local_master + m;
                if (l_local < E) {
                    load += ceil_div_f(smem_loads[l_local], smem_c[l_local]);
                }
            }
            smem_gpu_load_nc[g] = load;
            smem_gpu_slots_nc[g] = 0;
        }

        int eor_words = (G * E + 31) / 32;
        for (int w = lane; w < eor_words; w += 32) {
            smem_eor_union.packed[w] = 0;
        }
        __syncwarp();

        for (int i = lane; i < E; i += 32) {
            int g = i / num_local_master;
            int bit_idx = g * E + i;
            atomicOr(&smem_eor_union.packed[bit_idx / 32], (1u << (bit_idx % 32)));
        }
        __syncwarp();

        for (int r = 0; r < replica_count; ++r) {
            int l_local = smem_replicas[r].logical_id;
            float lpr = smem_replicas[r].load_per_replica;
            int best_gpu = -1;
            float best_load = kInfLoad;

            for (int base = 0; base < G; base += 32) {
                int g = base + lane;
                bool valid = (g < G) && (smem_gpu_slots_nc[g] < num_local_redundant) &&
                    !eor_is_set(smem_eor_union.packed, E, g, l_local);
                float load_val = (g < G) ? smem_gpu_load_nc[g] : kInfLoad;

                int argmin_lane = -1;
                float min_load = warp_reduce_argmin(load_val, valid, argmin_lane);
                int candidate_gpu = (min_load < kInfLoad * 0.5f) ? (base + argmin_lane) : -1;
                if (candidate_gpu >= 0 &&
                    (min_load < best_load || (min_load == best_load && (best_gpu < 0 || candidate_gpu < best_gpu)))) {
                    best_load = min_load;
                    best_gpu = candidate_gpu;
                }
            }

            if (lane == 0) {
                if (best_gpu >= 0 && best_gpu < G) {
                    int global_rank = domain_start_rank + best_gpu;
                    int slot = smem_gpu_slots_nc[best_gpu];
                    int phys_idx = global_rank * num_local_physical + num_local_master + slot;
                    int l_global = domain_start_log + l_local;

                    p2l_map[phys_idx] = l_global;
                    int l2p_slot = smem_l2p_slot[l_local];
                    l2p_map[l_global * max_replicas_dim + l2p_slot] = phys_idx;
                    smem_l2p_slot[l_local] = l2p_slot + 1;

                    smem_gpu_load_nc[best_gpu] += lpr;
                    smem_gpu_slots_nc[best_gpu]++;
                    eor_mark(smem_eor_union.packed, E, best_gpu, l_local);
                } else {
                    smem_c[l_local]--;
                }
            }
            __syncwarp();
        }
    }

    // Write final lcnts
    if constexpr (COMPACT_EOR) {
        for (int i = tid; i < E; i += 64) {
            lcnts[domain_start_log + i] = smem_c[i];
        }
    } else {
        for (int i = lane; i < E; i += 32) {
            lcnts[domain_start_log + i] = smem_c[i];
        }
    }
}

// ============================================================================
// PlacementSolverGPU implementation
// ============================================================================

PlacementSolverGPU::PlacementSolverGPU(int num_global_logical_experts,
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
      num_redundant_per_nvl_(num_local_redundant_experts * num_nvl_ranks) {
    EP_HOST_ASSERT(num_nvl_ranks > 0 && num_ranks % num_nvl_ranks == 0);
    EP_HOST_ASSERT(num_local_master_experts > 0);
    EP_HOST_ASSERT(num_local_redundant_experts >= 0);
    EP_HOST_ASSERT(max_replicas_dim >= 1);
    EP_HOST_ASSERT(num_logical_per_nvl_ <= MAX_EXPERTS_PER_NVL);
    EP_HOST_ASSERT(num_nvl_ranks_ <= MAX_GPUS_PER_NVL);
    EP_HOST_ASSERT(num_redundant_per_nvl_ <= MAX_REPLICAS_PER_NVL);
}

void PlacementSolverGPU::solve(const int32_t* expert_loads_gpu,
                               int32_t* p2l_gpu,
                               int32_t* l2p_gpu,
                               int32_t* lcnts_gpu,
                               cudaStream_t stream,
                               float balance_threshold) const {
    if (num_nvl_domains_ == 0) {
        return;
    }

    // External memset: initialize output arrays before kernel launch
    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(p2l_gpu, 0xFF, static_cast<size_t>(num_global_physical_) * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        l2p_gpu, 0xFF, static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(lcnts_gpu, 0, static_cast<size_t>(num_global_logical_experts_) * sizeof(int32_t), stream));

    dim3 grid(num_nvl_domains_);

    const int epl = (num_logical_per_nvl_ + 31) / 32;
    const bool compact = (num_nvl_ranks_ <= 64);

    // Template dispatch: EPL × COMPACT_EOR
    // COMPACT_EOR=true uses 64 threads (2 warps), false uses 32 threads (1 warp)
#define LAUNCH_V3(EPL_VAL, COMPACT_VAL)                                                                     \
    do {                                                                                                    \
        constexpr int NTHREADS = (COMPACT_VAL) ? 64 : 32;                                                   \
        const auto config = ultra_ep::kernels::make_launch_config(grid, dim3(NTHREADS), stream);   \
        ultra_ep::kernels::launch_kernel(placement_solve_kernel_v3<EPL_VAL, COMPACT_VAL>,          \
                                                 config,                                                    \
                                                 expert_loads_gpu,                                          \
                                                 p2l_gpu,                                                   \
                                                 l2p_gpu,                                                   \
                                                 lcnts_gpu,                                                 \
                                                 num_nvl_ranks_,                                            \
                                                 num_local_master_,                                         \
                                                 num_local_redundant_,                                      \
                                                 num_local_physical_,                                       \
                                                 max_replicas_dim_,                                         \
                                                 num_logical_per_nvl_,                                      \
                                                 num_redundant_per_nvl_,                                    \
                                                 num_global_physical_,                                      \
                                                 balance_threshold);                                        \
    } while (0)

    if (compact) {
        switch (epl) {
            case 1:
                LAUNCH_V3(1, true);
                break;
            case 2:
                LAUNCH_V3(2, true);
                break;
            case 3:
                LAUNCH_V3(3, true);
                break;
            case 4:
                LAUNCH_V3(4, true);
                break;
            case 5:
                LAUNCH_V3(5, true);
                break;
            case 6:
                LAUNCH_V3(6, true);
                break;
            case 7:
                LAUNCH_V3(7, true);
                break;
            case 8:
                LAUNCH_V3(8, true);
                break;
            default:
                LAUNCH_V3(16, true);
                break;
        }
    } else {
        switch (epl) {
            case 1:
                LAUNCH_V3(1, false);
                break;
            case 2:
                LAUNCH_V3(2, false);
                break;
            case 3:
                LAUNCH_V3(3, false);
                break;
            case 4:
                LAUNCH_V3(4, false);
                break;
            case 5:
                LAUNCH_V3(5, false);
                break;
            case 6:
                LAUNCH_V3(6, false);
                break;
            case 7:
                LAUNCH_V3(7, false);
                break;
            case 8:
                LAUNCH_V3(8, false);
                break;
            default:
                LAUNCH_V3(16, false);
                break;
        }
    }
#undef LAUNCH_V3

    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace ultra_ep::solver
