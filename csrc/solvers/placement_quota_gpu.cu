#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "../kernels/ptx.cuh"
#include "../utils/exception.cuh"
#include "api.hpp"

// Forward declarations avoid pulling NVSHMEM device headers into this TU.
namespace ultra_ep::runtime {
extern bool is_runtime_initialized;
extern int rank_idx;
}  // namespace ultra_ep::runtime

namespace ultra_ep::solver {

namespace ptx = ultra_ep::kernels::ptx;

namespace {

static constexpr int MAX_EXPERTS_PER_NVL = 512;
static constexpr int MAX_GPUS_PER_NVL = 72;
static constexpr int MAX_REPLICAS_PER_NVL = 512;
static constexpr int QUOTA_SOLVER_THREADS = 128;
static constexpr int QUOTA_SOLVER_WARPS = QUOTA_SOLVER_THREADS / 32;
static constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;

static_assert(QUOTA_SOLVER_THREADS % 32 == 0);

struct ExportPlanEntry {
    int expert_local;
    int target_rank_local;
    int quota;
};

__device__ __forceinline__ bool occ_has(uint64_t low, uint64_t high, int rank_local) {
    if (rank_local < 64) {
        return ((low >> rank_local) & 1ULL) != 0ULL;
    }
    return ((high >> (rank_local - 64)) & 1ULL) != 0ULL;
}

__device__ __forceinline__ void occ_set(uint64_t& low, uint64_t& high, int rank_local) {
    if (rank_local < 64) {
        low |= (1ULL << rank_local);
    } else {
        high |= (1ULL << (rank_local - 64));
    }
}

__device__ void warp_sort_source_ranks(int* source_order, const int32_t* excess, int G, int lane_id) {
    for (int i = lane_id; i < G; i += 32) {
        const int my_excess = excess[i];
        int pos = 0;
        for (int j = 0; j < G; ++j) {
            const int other_excess = excess[j];
            pos += ((other_excess > my_excess) || (other_excess == my_excess && j < i)) ? 1 : 0;
        }
        source_order[pos] = i;
    }
}

__device__ void warp_init_oracle_state_parallel(int32_t* export_sum,
                                                uint64_t* occ_lo,
                                                uint64_t* occ_hi,
                                                int32_t* excess,
                                                int32_t* slack,
                                                int32_t* slots_used,
                                                const int32_t* rank_load,
                                                int threshold,
                                                int E,
                                                int G,
                                                int num_local_master,
                                                int lane_id) {
    for (int l = lane_id; l < E; l += 32) {
        export_sum[l] = 0;
        occ_lo[l] = 0ULL;
        occ_hi[l] = 0ULL;
        occ_set(occ_lo[l], occ_hi[l], l / num_local_master);
    }
    for (int r = lane_id; r < G; r += 32) {
        excess[r] = max(rank_load[r] - threshold, 0);
        slack[r] = max(threshold - rank_load[r], 0);
        slots_used[r] = 0;
    }
}

__device__ int warp_find_best_target(const int32_t* slack,
                                     const int32_t* slots_used,
                                     const uint64_t* occ_lo,
                                     const uint64_t* occ_hi,
                                     int expert_local,
                                     int need,
                                     int available,
                                     int min_tokens_per_replica,
                                     int num_local_redundant,
                                     int G,
                                     int lane_id) {
    int my_best_target = -1;
    int my_best_cap = -1;

    for (int target_rank_local = lane_id; target_rank_local < G; target_rank_local += 32) {
        if (slack[target_rank_local] <= 0 || slots_used[target_rank_local] >= num_local_redundant ||
            occ_has(occ_lo[expert_local], occ_hi[expert_local], target_rank_local)) {
            continue;
        }

        const int q = min(min(need, slack[target_rank_local]), available);
        if (q < min_tokens_per_replica) {
            continue;
        }

        if (slack[target_rank_local] > my_best_cap ||
            (slack[target_rank_local] == my_best_cap &&
             (my_best_target < 0 || target_rank_local < my_best_target))) {
            my_best_target = target_rank_local;
            my_best_cap = slack[target_rank_local];
        }
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int other_target = __shfl_xor_sync(FULL_WARP_MASK, my_best_target, delta);
        const int other_cap = __shfl_xor_sync(FULL_WARP_MASK, my_best_cap, delta);
        const bool other_better =
            (other_target >= 0) &&
            ((my_best_target < 0) || (other_cap > my_best_cap) ||
             (other_cap == my_best_cap && other_target < my_best_target));
        if (other_better) {
            my_best_target = other_target;
            my_best_cap = other_cap;
        }
    }

    return my_best_target;
}

template <bool STORE_PLAN>
__device__ bool warp_build_export_plan(const int32_t* loads,
                                       const int* sorted_experts,
                                       int32_t* export_sum,
                                       int32_t* excess,
                                       int32_t* slack,
                                       int32_t* slots_used,
                                       int* source_order,
                                       uint64_t* occ_lo,
                                       uint64_t* occ_hi,
                                       ExportPlanEntry* export_plan,
                                       int& num_exports,
                                       int G,
                                       int num_local_master,
                                       int num_local_redundant,
                                       int min_tokens_per_replica,
                                       bool allow_zero_master_quota,
                                       int lane_id) {
    warp_sort_source_ranks(source_order, excess, G, lane_id);
    __syncwarp();

    if (lane_id == 0) {
        num_exports = 0;
    }
    __syncwarp();

    for (int ord = 0; ord < G; ++ord) {
        const int source_rank_local = source_order[ord];
        int need = excess[source_rank_local];
        if (need <= 0) {
            continue;
        }

        for (int pos = 0; pos < num_local_master; ++pos) {
            const int expert_local = sorted_experts[source_rank_local * num_local_master + pos];
            const int keep_on_master = (!allow_zero_master_quota && loads[expert_local] > 0) ? 1 : 0;
            int available = max(loads[expert_local] - keep_on_master - export_sum[expert_local], 0);

            while (need > 0 && available > 0) {
                const int best_target = warp_find_best_target(slack,
                                                              slots_used,
                                                              occ_lo,
                                                              occ_hi,
                                                              expert_local,
                                                              need,
                                                              available,
                                                              min_tokens_per_replica,
                                                              num_local_redundant,
                                                              G,
                                                              lane_id);
                if (best_target < 0) {
                    break;
                }

                if constexpr (STORE_PLAN) {
                    int can_store = 1;
                    if (lane_id == 0 && num_exports >= MAX_REPLICAS_PER_NVL) {
                        can_store = 0;
                    }
                    can_store = __shfl_sync(FULL_WARP_MASK, can_store, 0);
                    if (!can_store) {
                        return false;
                    }
                }

                const int q = min(min(need, slack[best_target]), available);
                if (lane_id == 0) {
                    if constexpr (STORE_PLAN) {
                        export_plan[num_exports++] = {expert_local, best_target, q};
                    }
                    export_sum[expert_local] += q;
                    slack[best_target] -= q;
                    slots_used[best_target] += 1;
                    occ_set(occ_lo[expert_local], occ_hi[expert_local], best_target);
                }
                __syncwarp();

                need -= q;
                available -= q;
            }

            if (need == 0) {
                break;
            }
        }

        if (need > 0) {
            return false;
        }
    }

    return true;
}

__global__ void quota_placement_solve_kernel(const int32_t* __restrict__ expert_loads,
                                             const int32_t* __restrict__ expert_loads_per_rank,
                                             int32_t* __restrict__ p2l_map,
                                             int32_t* __restrict__ l2p_map,
                                             int32_t* __restrict__ lcnts,
                                             int32_t* __restrict__ quota,
                                             int32_t* __restrict__ quota_prefix,
                                             int32_t* __restrict__ rank_quota_prefix,
                                             int num_ranks,
                                             int num_nvl_ranks,
                                             int num_local_master,
                                             int num_local_redundant,
                                             int num_local_physical,
                                             int max_replicas_dim,
                                             int num_global_logical_experts,
                                             int num_logical_per_nvl,
                                             float balance_threshold,
                                             int32_t min_tokens_per_replica,
                                             bool allow_zero_master_quota,
                                             bool locality_aware,
                                             int my_rank) {
    (void)num_ranks;

    extern __shared__ char smem_dynamic_raw[];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int domain = blockIdx.x;

    const int E = num_logical_per_nvl;
    const int G = num_nvl_ranks;
    const int stride_elems = ((E + 3) / 4) * 4;

    // Layout dynamic shared memory: domain_loads | occ_lo | occ_hi
    size_t dyn_off = 0;
    int32_t* smem_domain_loads = reinterpret_cast<int32_t*>(smem_dynamic_raw + dyn_off);
    dyn_off += static_cast<size_t>(G) * stride_elems * sizeof(int32_t);
    dyn_off = (dyn_off + 7u) & ~size_t(7);  // align to 8 for uint64_t
    uint64_t* smem_warp_occ_lo_base = reinterpret_cast<uint64_t*>(smem_dynamic_raw + dyn_off);
    dyn_off += static_cast<size_t>(QUOTA_SOLVER_WARPS) * E * sizeof(uint64_t);
    uint64_t* smem_warp_occ_hi_base = reinterpret_cast<uint64_t*>(smem_dynamic_raw + dyn_off);
    const int domain_start_rank = domain * num_nvl_ranks;
    const int domain_start_log = domain_start_rank * num_local_master;

    __shared__ int32_t smem_loads[MAX_EXPERTS_PER_NVL];
    __shared__ int32_t smem_c[MAX_EXPERTS_PER_NVL];
    __shared__ int32_t smem_rank_load[MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_my_loads[MAX_EXPERTS_PER_NVL];
    __shared__ int smem_sorted_experts[MAX_EXPERTS_PER_NVL];
    __shared__ ExportPlanEntry smem_export_plan[MAX_REPLICAS_PER_NVL];
    __shared__ int smem_num_exports;
    __shared__ ptx::arrival_phase smem_tma_phase;

    __shared__ int smem_bs_lo;
    __shared__ int smem_bs_hi;
    __shared__ bool smem_precheck_done;
    __shared__ int smem_probes[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_valid[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_feasible[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_small_range;

    __shared__ int32_t smem_warp_export_sum[QUOTA_SOLVER_WARPS][MAX_EXPERTS_PER_NVL];
    __shared__ int32_t smem_warp_excess[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_warp_slack[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_warp_slots_used[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];
    __shared__ int smem_warp_source_order[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];

    for (int i = tid; i < E; i += blockDim.x) {
        const int l_global = domain_start_log + i;
        const int load = expert_loads[l_global];
        smem_loads[i] = load;
        smem_c[i] = 1;

        const int global_rank = domain_start_rank + i / num_local_master;
        const int local_idx = i % num_local_master;
        const int phys_idx = global_rank * num_local_physical + local_idx;
        p2l_map[phys_idx] = l_global;
        l2p_map[l_global * max_replicas_dim] = phys_idx;
    }
    __syncthreads();

    for (int r = tid; r < G; r += blockDim.x) {
        int sum = 0;
        for (int m = 0; m < num_local_master; ++m) {
            sum += smem_loads[r * num_local_master + m];
        }
        smem_rank_load[r] = sum;
    }
    __syncthreads();

    for (int r = tid; r < G; r += blockDim.x) {
        int* row = smem_sorted_experts + r * num_local_master;
        for (int m = 0; m < num_local_master; ++m) {
            row[m] = r * num_local_master + m;
        }
        for (int i = 1; i < num_local_master; ++i) {
            const int key = row[i];
            int j = i - 1;
            while (j >= 0) {
                const int cur = row[j];
                const bool cur_better = (smem_loads[cur] > smem_loads[key]) ||
                                        (smem_loads[cur] == smem_loads[key] && cur < key);
                if (cur_better) {
                    break;
                }
                row[j + 1] = cur;
                --j;
            }
            row[j + 1] = key;
        }
    }

    for (int i = tid; i < E; i += blockDim.x) {
        smem_my_loads[i] = expert_loads_per_rank[my_rank * num_global_logical_experts + domain_start_log + i];
    }

    const bool can_use_tma =
        (domain_start_log % 4 == 0) && (E % 4 == 0) && (num_global_logical_experts % 4 == 0);
    ptx::mbarrier* mbar = nullptr;
    if (can_use_tma) {
        mbar = ptx::create_mbarrier();
        if (tid == 0) {
            ptx::mbarrier_init(mbar, 1);
            smem_tma_phase = 0;
            const int total_bytes = G * E * static_cast<int>(sizeof(int32_t));
            ptx::mbarrier_arrive_and_set_tx(mbar, total_bytes);
            for (int r = 0; r < G; ++r) {
                ptx::tma_load_1d(smem_domain_loads + r * stride_elems,
                                 expert_loads_per_rank +
                                     (domain_start_rank + r) * num_global_logical_experts + domain_start_log,
                                 mbar,
                                 E * static_cast<int>(sizeof(int32_t)),
                                 ptx::TMACacheHint::kEvictFirst);
            }
        }
    } else {
        for (int idx = tid; idx < G * E; idx += blockDim.x) {
            const int r = idx / E;
            const int e = idx % E;
            smem_domain_loads[r * stride_elems + e] =
                expert_loads_per_rank[(domain_start_rank + r) * num_global_logical_experts + domain_start_log + e];
        }
    }
    __syncthreads();

    if (tid == 0) {
        smem_num_exports = 0;
    }
    __syncthreads();

    const bool do_binary_search = (num_local_redundant > 0 && G > 1);
    if (do_binary_search) {
        if (tid == 0) {
            int total_rank_load = 0;
            int max_rank_load = 0;
            for (int r = 0; r < G; ++r) {
                total_rank_load += smem_rank_load[r];
                max_rank_load = max(max_rank_load, static_cast<int>(smem_rank_load[r]));
            }
            const float bt = fmaxf(balance_threshold, 1.0f);
            smem_bs_lo = static_cast<int>(ceilf((static_cast<float>(total_rank_load) / G) * bt));
            smem_bs_hi = max(max_rank_load, smem_bs_lo);
            smem_precheck_done = false;
        }
        __syncthreads();

        if (warp_id == 0) {
            warp_init_oracle_state_parallel(smem_warp_export_sum[0],
                                            smem_warp_occ_lo_base,
                                            smem_warp_occ_hi_base,
                                            smem_warp_excess[0],
                                            smem_warp_slack[0],
                                            smem_warp_slots_used[0],
                                            smem_rank_load,
                                            smem_bs_lo,
                                            E,
                                            G,
                                            num_local_master,
                                            lane_id);
            __syncwarp();

            int precheck_exports = 0;
            const bool precheck_feasible = warp_build_export_plan<true>(smem_loads,
                                                                        smem_sorted_experts,
                                                                        smem_warp_export_sum[0],
                                                                        smem_warp_excess[0],
                                                                        smem_warp_slack[0],
                                                                        smem_warp_slots_used[0],
                                                                        smem_warp_source_order[0],
                                                                        smem_warp_occ_lo_base,
                                                                        smem_warp_occ_hi_base,
                                                                        smem_export_plan,
                                                                        precheck_exports,
                                                                        G,
                                                                        num_local_master,
                                                                        num_local_redundant,
                                                                        min_tokens_per_replica,
                                                                        allow_zero_master_quota,
                                                                        lane_id);
            if (lane_id == 0) {
                smem_precheck_done = precheck_feasible;
                if (precheck_feasible) {
                    smem_num_exports = precheck_exports;
                } else {
                    smem_bs_lo = min(smem_bs_lo + 1, smem_bs_hi);
                }
            }
        }
        __syncthreads();

        while (true) {
            if (smem_precheck_done || smem_bs_lo >= smem_bs_hi) {
                break;
            }

            if (tid == 0) {
                const int range = smem_bs_hi - smem_bs_lo;
                smem_probe_small_range = (range <= QUOTA_SOLVER_WARPS) ? 1 : 0;
                if (smem_probe_small_range) {
                    for (int w = 0; w < QUOTA_SOLVER_WARPS; ++w) {
                        smem_probes[w] = smem_bs_lo + w;
                        smem_probe_valid[w] = (smem_probes[w] < smem_bs_hi) ? 1 : 0;
                        smem_probe_feasible[w] = 0;
                    }
                } else {
                    for (int w = 0; w < QUOTA_SOLVER_WARPS; ++w) {
                        smem_probes[w] =
                            smem_bs_lo + static_cast<int>((static_cast<int64_t>(range) * (w + 1)) /
                                                          (QUOTA_SOLVER_WARPS + 1));
                        smem_probe_valid[w] = 1;
                        smem_probe_feasible[w] = 0;
                    }
                }
            }
            __syncthreads();

            bool feasible = false;
            if (smem_probe_valid[warp_id]) {
                warp_init_oracle_state_parallel(smem_warp_export_sum[warp_id],
                                                smem_warp_occ_lo_base + warp_id * E,
                                                smem_warp_occ_hi_base + warp_id * E,
                                                smem_warp_excess[warp_id],
                                                smem_warp_slack[warp_id],
                                                smem_warp_slots_used[warp_id],
                                                smem_rank_load,
                                                smem_probes[warp_id],
                                                E,
                                                G,
                                                num_local_master,
                                                lane_id);
                __syncwarp();

                int dummy_exports = 0;
                feasible = warp_build_export_plan<false>(smem_loads,
                                                         smem_sorted_experts,
                                                         smem_warp_export_sum[warp_id],
                                                         smem_warp_excess[warp_id],
                                                         smem_warp_slack[warp_id],
                                                         smem_warp_slots_used[warp_id],
                                                         smem_warp_source_order[warp_id],
                                                         smem_warp_occ_lo_base + warp_id * E,
                                                         smem_warp_occ_hi_base + warp_id * E,
                                                         nullptr,
                                                         dummy_exports,
                                                         G,
                                                         num_local_master,
                                                         num_local_redundant,
                                                         min_tokens_per_replica,
                                                         allow_zero_master_quota,
                                                         lane_id);
            }
            if (lane_id == 0) {
                smem_probe_feasible[warp_id] = feasible ? 1 : 0;
            }
            __syncthreads();

            if (tid == 0) {
                if (smem_probe_small_range) {
                    int best_probe = -1;
                    for (int w = 0; w < QUOTA_SOLVER_WARPS; ++w) {
                        if (smem_probe_valid[w] && smem_probe_feasible[w]) {
                            best_probe = smem_probes[w];
                            break;
                        }
                    }
                    if (best_probe >= 0) {
                        smem_bs_lo = best_probe;
                        smem_bs_hi = best_probe;
                    } else {
                        smem_bs_lo = smem_bs_hi;
                    }
                } else {
                    int first_feasible = -1;
                    for (int w = 0; w < QUOTA_SOLVER_WARPS; ++w) {
                        if (smem_probe_feasible[w]) {
                            first_feasible = w;
                            break;
                        }
                    }
                    if (first_feasible >= 0) {
                        smem_bs_hi = smem_probes[first_feasible];
                        if (first_feasible > 0) {
                            smem_bs_lo = smem_probes[first_feasible - 1] + 1;
                        }
                    } else {
                        smem_bs_lo = smem_probes[QUOTA_SOLVER_WARPS - 1] + 1;
                    }
                }
            }
            __syncthreads();
        }

        if (!smem_precheck_done) {
            if (warp_id == 0) {
                warp_init_oracle_state_parallel(smem_warp_export_sum[0],
                                                smem_warp_occ_lo_base,
                                                smem_warp_occ_hi_base,
                                                smem_warp_excess[0],
                                                smem_warp_slack[0],
                                                smem_warp_slots_used[0],
                                                smem_rank_load,
                                                smem_bs_lo,
                                                E,
                                                G,
                                                num_local_master,
                                                lane_id);
                __syncwarp();

                int final_exports = 0;
                const bool final_feasible = warp_build_export_plan<true>(smem_loads,
                                                                         smem_sorted_experts,
                                                                         smem_warp_export_sum[0],
                                                                         smem_warp_excess[0],
                                                                         smem_warp_slack[0],
                                                                         smem_warp_slots_used[0],
                                                                         smem_warp_source_order[0],
                                                                         smem_warp_occ_lo_base,
                                                                         smem_warp_occ_hi_base,
                                                                         smem_export_plan,
                                                                         final_exports,
                                                                         G,
                                                                         num_local_master,
                                                                         num_local_redundant,
                                                                         min_tokens_per_replica,
                                                                         allow_zero_master_quota,
                                                                         lane_id);
                if (lane_id == 0) {
                    EP_DEVICE_ASSERT(final_feasible);
                    smem_num_exports = final_exports;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        int next_slot[MAX_GPUS_PER_NVL] = {0};
        int expert_slot[MAX_EXPERTS_PER_NVL];
        int expert_prefix[MAX_EXPERTS_PER_NVL];
        const int32_t* final_export_sum = smem_warp_export_sum[0];

        for (int expert_local = 0; expert_local < E; ++expert_local) {
            const int l_global = domain_start_log + expert_local;
            const int row_offset = l_global * max_replicas_dim;
            const int master_quota =
                do_binary_search ? (smem_loads[expert_local] - final_export_sum[expert_local]) : smem_loads[expert_local];
            quota[row_offset] = master_quota;
            quota_prefix[row_offset] = master_quota;
            expert_slot[expert_local] = 1;
            expert_prefix[expert_local] = master_quota;
        }

        for (int plan_idx = 0; plan_idx < smem_num_exports; ++plan_idx) {
            const ExportPlanEntry entry = smem_export_plan[plan_idx];
            const int expert_local = entry.expert_local;
            const int l_global = domain_start_log + expert_local;
            const int row_offset = l_global * max_replicas_dim;
            const int slot = expert_slot[expert_local]++;

            const int target_global_rank = domain_start_rank + entry.target_rank_local;
            const int phys_idx = target_global_rank * num_local_physical + num_local_master +
                                 next_slot[entry.target_rank_local]++;
            p2l_map[phys_idx] = l_global;
            l2p_map[row_offset + slot] = phys_idx;
            quota[row_offset + slot] = entry.quota;
            expert_prefix[expert_local] += entry.quota;
            quota_prefix[row_offset + slot] = expert_prefix[expert_local];
        }

        for (int expert_local = 0; expert_local < E; ++expert_local) {
            const int l_global = domain_start_log + expert_local;
            lcnts[l_global] = expert_slot[expert_local];
            smem_c[expert_local] = expert_slot[expert_local];
        }
    }
    __syncthreads();

    if (can_use_tma) {
        if (tid == 0) {
            ptx::mbarrier_wait_and_flip_phase(mbar, smem_tma_phase);
            ptx::mbarrier_invalidate(mbar);
        }
        __syncthreads();
    }

    for (int expert_local = tid; expert_local < E; expert_local += blockDim.x) {
        const int l_global = domain_start_log + expert_local;
        const int row_offset = l_global * max_replicas_dim;
        const int C = smem_c[expert_local];

        int host_rank[MAX_GPUS_PER_NVL];
        int my_alloc[MAX_GPUS_PER_NVL];
        int64_t remainders[MAX_GPUS_PER_NVL];

        for (int j = 0; j < C; ++j) {
            host_rank[j] = l2p_map[row_offset + j] / num_local_physical;
            my_alloc[j] = 0;
            remainders[j] = -1;
        }

        if (!locality_aware) {
            const int rem_my = smem_my_loads[expert_local];
            int assigned = 0;
            int total_quota = 0;
            for (int j = 0; j < C; ++j) {
                total_quota += quota[row_offset + j];
            }
            if (rem_my > 0 && total_quota > 0) {
                for (int j = 0; j < C; ++j) {
                    const int64_t scaled = static_cast<int64_t>(rem_my) * quota[row_offset + j];
                    const int share = static_cast<int>(scaled / total_quota);
                    my_alloc[j] = share;
                    remainders[j] = scaled % total_quota;
                    assigned += share;
                }
                int remaining = rem_my - assigned;
                while (remaining > 0) {
                    int best_j = -1;
                    for (int j = 0; j < C; ++j) {
                        if (best_j < 0 || remainders[j] > remainders[best_j] ||
                            (remainders[j] == remainders[best_j] && j < best_j)) {
                            best_j = j;
                        }
                    }
                    EP_DEVICE_ASSERT(best_j >= 0);
                    my_alloc[best_j] += 1;
                    remainders[best_j] = -1;
                    --remaining;
                }
            }
        } else {
            int host_remaining[MAX_GPUS_PER_NVL];
            int remote_cap[MAX_GPUS_PER_NVL];

            for (int r = 0; r < G; ++r) {
                host_remaining[r] = smem_domain_loads[r * stride_elems + expert_local];
            }

            int rem_my = smem_my_loads[expert_local];
            int total_remote_cap = 0;
            for (int j = 0; j < C; ++j) {
                const int q = quota[row_offset + j];
                const int host_local = host_rank[j] - domain_start_rank;
                EP_DEVICE_ASSERT(host_local >= 0 && host_local < G);
                const int local_fill = min(host_remaining[host_local], q);
                host_remaining[host_local] -= local_fill;
                remote_cap[j] = q - local_fill;
                total_remote_cap += remote_cap[j];
                if (host_rank[j] == my_rank) {
                    my_alloc[j] += local_fill;
                    rem_my -= local_fill;
                }
            }
            rem_my = max(rem_my, 0);

            if (rem_my > 0 && total_remote_cap > 0) {
                int assigned = 0;
                for (int j = 0; j < C; ++j) {
                    if (host_rank[j] == my_rank || remote_cap[j] <= 0) {
                        continue;
                    }
                    const int64_t scaled = static_cast<int64_t>(rem_my) * remote_cap[j];
                    const int share = static_cast<int>(scaled / total_remote_cap);
                    my_alloc[j] += share;
                    remainders[j] = scaled % total_remote_cap;
                    assigned += share;
                }
                int remaining = rem_my - assigned;
                while (remaining > 0) {
                    int best_j = -1;
                    for (int j = 0; j < C; ++j) {
                        if (host_rank[j] == my_rank || remote_cap[j] <= 0) {
                            continue;
                        }
                        if (best_j < 0 || remainders[j] > remainders[best_j] ||
                            (remainders[j] == remainders[best_j] &&
                             (host_rank[j] < host_rank[best_j] ||
                              (host_rank[j] == host_rank[best_j] && j < best_j)))) {
                            best_j = j;
                        }
                    }
                    EP_DEVICE_ASSERT(best_j >= 0);
                    my_alloc[best_j] += 1;
                    remainders[best_j] = -1;
                    --remaining;
                }
            }
        }

        int prefix = 0;
        for (int j = 0; j < C; ++j) {
            prefix += my_alloc[j];
            rank_quota_prefix[row_offset + j] = prefix;
        }
    }
}

}  // namespace

PlacementSolverQuota::PlacementSolverQuota(int num_global_logical_experts,
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
    EP_HOST_ASSERT(num_nvl_ranks_ > 0 && num_ranks_ % num_nvl_ranks_ == 0);
    EP_HOST_ASSERT(num_local_master_ > 0);
    EP_HOST_ASSERT(num_local_redundant_ >= 0);
    EP_HOST_ASSERT(max_replicas_dim_ >= 1);
    EP_HOST_ASSERT(num_logical_per_nvl_ <= MAX_EXPERTS_PER_NVL);
    EP_HOST_ASSERT(num_nvl_ranks_ <= MAX_GPUS_PER_NVL);
    EP_HOST_ASSERT(num_redundant_per_nvl_ <= MAX_REPLICAS_PER_NVL);
}

PlacementSolverQuota::~PlacementSolverQuota() = default;

void PlacementSolverQuota::solve(const int32_t* expert_loads_gpu,
                                 const int32_t* expert_loads_per_rank_gpu,
                                 int32_t* p2l_gpu,
                                 int32_t* l2p_gpu,
                                 int32_t* lcnts_gpu,
                                 int32_t* quota_gpu,
                                 int32_t* quota_prefix_gpu,
                                 int32_t* rank_quota_prefix_gpu,
                                 cudaStream_t stream,
                                 float balance_threshold,
                                 int32_t min_tokens_per_replica,
                                 bool allow_zero_master_quota,
                                 bool locality_aware) const {
    if (num_nvl_domains_ == 0) {
        return;
    }

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        p2l_gpu, 0xFF,
        static_cast<size_t>(num_global_physical_) * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        l2p_gpu, 0xFF,
        static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        lcnts_gpu, 0,
        static_cast<size_t>(num_global_logical_experts_) * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        quota_gpu, 0,
        static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        quota_prefix_gpu, 0,
        static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        rank_quota_prefix_gpu, 0,
        static_cast<size_t>(num_global_logical_experts_) * max_replicas_dim_ * sizeof(int32_t), stream));

    const int stride_elems = ((num_logical_per_nvl_ + 3) / 4) * 4;
    const size_t domain_loads_bytes = static_cast<size_t>(num_nvl_ranks_) * stride_elems * sizeof(int32_t);
    const size_t occ_offset = (domain_loads_bytes + 7u) & ~size_t(7);
    const size_t occ_bytes = 2 * static_cast<size_t>(QUOTA_SOLVER_WARPS) * num_logical_per_nvl_ * sizeof(uint64_t);
    const size_t smem_size = occ_offset + occ_bytes;
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(quota_placement_solve_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem_size)));

    const int my_rank = runtime::is_runtime_initialized ? runtime::rank_idx : 0;
    dim3 grid(num_nvl_domains_);
    dim3 block(QUOTA_SOLVER_THREADS);
    quota_placement_solve_kernel<<<grid, block, smem_size, stream>>>(expert_loads_gpu,
                                                                     expert_loads_per_rank_gpu,
                                                                     p2l_gpu,
                                                                     l2p_gpu,
                                                                     lcnts_gpu,
                                                                     quota_gpu,
                                                                     quota_prefix_gpu,
                                                                     rank_quota_prefix_gpu,
                                                                     num_ranks_,
                                                                     num_nvl_ranks_,
                                                                     num_local_master_,
                                                                     num_local_redundant_,
                                                                     num_local_physical_,
                                                                     max_replicas_dim_,
                                                                     num_global_logical_experts_,
                                                                     num_logical_per_nvl_,
                                                                     balance_threshold,
                                                                     min_tokens_per_replica,
                                                                     allow_zero_master_quota,
                                                                     locality_aware,
                                                                     my_rank);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace ultra_ep::solver
