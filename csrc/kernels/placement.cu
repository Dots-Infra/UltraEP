#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "launch.cuh"
#include "api.cuh"
#include "ptx.cuh"
#include "../utils/exception.cuh"

// Forward declarations avoid pulling NVSHMEM device headers into this TU.
namespace ultra_ep::runtime {
extern bool is_runtime_initialized;
extern int rank_idx;
}  // namespace ultra_ep::runtime

namespace ultra_ep::kernels {

namespace {

static constexpr int MAX_EXPERTS_PER_NVL = 512;
static constexpr int MAX_GPUS_PER_NVL = 72;
static constexpr int MAX_REPLICAS_PER_NVL = 512;
static constexpr int QUOTA_SOLVER_THREADS = 128;
static constexpr int QUOTA_SOLVER_WARPS = QUOTA_SOLVER_THREADS / 32;
static constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;
static constexpr int QUOTA_FAST_REPLICA_LIMIT = 8;
static constexpr int QUOTA_ORACLE_MAX_BATCH_K = 8;
static constexpr int QUOTA_ORACLE_FAST_MAX_RETRY = 2;

static_assert(QUOTA_SOLVER_THREADS % 32 == 0);
static_assert(QUOTA_FAST_REPLICA_LIMIT <= MAX_GPUS_PER_NVL);

struct ReplicaEntry {
    int logical_id;
    float load_per_replica;
};

struct ExportPlanEntry {
    int expert_local;
    int target_rank_local;
    int quota;
};

struct V4OracleTask {
    int source_rank_local;
    int expert_local;
    int expert_pos;
    int need;
    int available;
    int min_tokens;
    int target_1;
    int quota_1;
    int target_2;
    int quota_2;
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

__device__ __forceinline__ int warp_reduce_sum(int value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_xor_sync(FULL_WARP_MASK, value, delta);
    }
    return value;
}

__device__ void warp_sort_source_ranks_by_load(int* source_order, const int32_t* rank_load, int G, int lane_id) {
    for (int i = lane_id; i < G; i += 32) {
        const int my_load = rank_load[i];
        int pos = 0;
        for (int j = 0; j < G; ++j) {
            const int other_load = rank_load[j];
            pos += ((other_load > my_load) || (other_load == my_load && j < i)) ? 1 : 0;
        }
        source_order[pos] = i;
    }
}

__device__ bool warp_capacity_feasible(const int32_t* rank_load, int threshold, int G, int lane_id) {
    int local_excess = 0;
    int local_slack = 0;
    for (int r = lane_id; r < G; r += 32) {
        local_excess += max(rank_load[r] - threshold, 0);
        local_slack += max(threshold - rank_load[r], 0);
    }
    return warp_reduce_sum(local_excess) <= warp_reduce_sum(local_slack);
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
            (slack[target_rank_local] == my_best_cap && (my_best_target < 0 || target_rank_local < my_best_target))) {
            my_best_target = target_rank_local;
            my_best_cap = slack[target_rank_local];
        }
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int other_target = __shfl_xor_sync(FULL_WARP_MASK, my_best_target, delta);
        const int other_cap = __shfl_xor_sync(FULL_WARP_MASK, my_best_cap, delta);
        const bool other_better = (other_target >= 0) &&
            ((my_best_target < 0) || (other_cap > my_best_cap) ||
             (other_cap == my_best_cap && other_target < my_best_target));
        if (other_better) {
            my_best_target = other_target;
            my_best_cap = other_cap;
        }
    }

    return my_best_target;
}

__device__ int warp_find_best_target_reg(int reg_slack_0,
                                         int reg_slack_1,
                                         int reg_slots_0,
                                         int reg_slots_1,
                                         uint64_t occ_lo,
                                         uint64_t occ_hi,
                                         int need,
                                         int available,
                                         int min_tokens_per_replica,
                                         int num_local_redundant,
                                         int G,
                                         int lane_id) {
    int my_best_target = -1;
    int my_best_cap = -1;

    const int target_0 = lane_id;
    if (target_0 < G && reg_slack_0 > 0 && reg_slots_0 < num_local_redundant && !occ_has(occ_lo, occ_hi, target_0)) {
        const int q0 = min(min(need, reg_slack_0), available);
        if (q0 >= min_tokens_per_replica) {
            my_best_target = target_0;
            my_best_cap = reg_slack_0;
        }
    }

    const int target_1 = lane_id + 32;
    if (target_1 < G && reg_slack_1 > 0 && reg_slots_1 < num_local_redundant && !occ_has(occ_lo, occ_hi, target_1)) {
        const int q1 = min(min(need, reg_slack_1), available);
        if (q1 >= min_tokens_per_replica &&
            (reg_slack_1 > my_best_cap ||
             (reg_slack_1 == my_best_cap && (my_best_target < 0 || target_1 < my_best_target)))) {
            my_best_target = target_1;
            my_best_cap = reg_slack_1;
        }
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        const int other_target = __shfl_xor_sync(FULL_WARP_MASK, my_best_target, delta);
        const int other_cap = __shfl_xor_sync(FULL_WARP_MASK, my_best_cap, delta);
        const bool other_better = (other_target >= 0) &&
            ((my_best_target < 0) || (other_cap > my_best_cap) ||
             (other_cap == my_best_cap && other_target < my_best_target));
        if (other_better) {
            my_best_target = other_target;
            my_best_cap = other_cap;
        }
    }

    return my_best_target;
}

__device__ int warp_collect_topk_targets_reg(int reg_slack_0,
                                             int reg_slack_1,
                                             int reg_slots_0,
                                             int reg_slots_1,
                                             uint64_t occ_lo,
                                             uint64_t occ_hi,
                                             int need,
                                             int available,
                                             int min_tokens_per_replica,
                                             int num_local_redundant,
                                             int G,
                                             int lane_id,
                                             int batch_k,
                                             int* out_targets,
                                             int* out_quota) {
    int cand_rank_0 = -1;
    int cand_cap_0 = -1;
    int cand_rank_1 = -1;
    int cand_cap_1 = -1;

    const int rank_0 = lane_id;
    if (rank_0 < G && reg_slack_0 > 0 && reg_slots_0 < num_local_redundant && !occ_has(occ_lo, occ_hi, rank_0)) {
        const int cap = min(min(need, reg_slack_0), available);
        if (cap >= min_tokens_per_replica) {
            cand_rank_0 = rank_0;
            cand_cap_0 = cap;
        }
    }

    const int rank_1 = lane_id + 32;
    if (rank_1 < G && reg_slack_1 > 0 && reg_slots_1 < num_local_redundant && !occ_has(occ_lo, occ_hi, rank_1)) {
        const int cap = min(min(need, reg_slack_1), available);
        if (cap >= min_tokens_per_replica) {
            cand_rank_1 = rank_1;
            cand_cap_1 = cap;
        }
    }

    int selected = 0;
    int sel_rank[QUOTA_ORACLE_MAX_BATCH_K];
    int sel_quota[QUOTA_ORACLE_MAX_BATCH_K];
    int top_rank[QUOTA_ORACLE_MAX_BATCH_K];
    int top_cap[QUOTA_ORACLE_MAX_BATCH_K];
    int top_n = 0;
    const int picks = min(batch_k, QUOTA_ORACLE_MAX_BATCH_K);
    for (int src = 0; src < 32; ++src) {
        const int r0 = __shfl_sync(FULL_WARP_MASK, cand_rank_0, src);
        const int c0 = __shfl_sync(FULL_WARP_MASK, cand_cap_0, src);
        if (lane_id == 0 && r0 >= 0 && c0 > 0 && picks > 0) {
            int idx = top_n;
            if (top_n < picks) {
                top_rank[idx] = r0;
                top_cap[idx] = c0;
                ++top_n;
            } else if (c0 > top_cap[picks - 1] || (c0 == top_cap[picks - 1] && r0 < top_rank[picks - 1])) {
                idx = picks - 1;
                top_rank[idx] = r0;
                top_cap[idx] = c0;
            } else {
                idx = -1;
            }
            while (idx > 0 &&
                   (top_cap[idx] > top_cap[idx - 1] ||
                    (top_cap[idx] == top_cap[idx - 1] && top_rank[idx] < top_rank[idx - 1]))) {
                const int tmp_rank = top_rank[idx - 1];
                const int tmp_cap = top_cap[idx - 1];
                top_rank[idx - 1] = top_rank[idx];
                top_cap[idx - 1] = top_cap[idx];
                top_rank[idx] = tmp_rank;
                top_cap[idx] = tmp_cap;
                --idx;
            }
        }
        const int r1 = __shfl_sync(FULL_WARP_MASK, cand_rank_1, src);
        const int c1 = __shfl_sync(FULL_WARP_MASK, cand_cap_1, src);
        if (lane_id == 0 && r1 >= 0 && c1 > 0 && picks > 0) {
            int idx = top_n;
            if (top_n < picks) {
                top_rank[idx] = r1;
                top_cap[idx] = c1;
                ++top_n;
            } else if (c1 > top_cap[picks - 1] || (c1 == top_cap[picks - 1] && r1 < top_rank[picks - 1])) {
                idx = picks - 1;
                top_rank[idx] = r1;
                top_cap[idx] = c1;
            } else {
                idx = -1;
            }
            while (idx > 0 &&
                   (top_cap[idx] > top_cap[idx - 1] ||
                    (top_cap[idx] == top_cap[idx - 1] && top_rank[idx] < top_rank[idx - 1]))) {
                const int tmp_rank = top_rank[idx - 1];
                const int tmp_cap = top_cap[idx - 1];
                top_rank[idx - 1] = top_rank[idx];
                top_cap[idx - 1] = top_cap[idx];
                top_rank[idx] = tmp_rank;
                top_cap[idx] = tmp_cap;
                --idx;
            }
        }
    }

    if (lane_id == 0) {
        int need_left = need;
        int available_left = available;
        for (int pick = 0; pick < top_n && need_left > 0 && available_left > 0; ++pick) {
            const int q = min(min(need_left, top_cap[pick]), available_left);
            if (q < min_tokens_per_replica) {
                break;
            }
            sel_rank[selected] = top_rank[pick];
            sel_quota[selected] = q;
            ++selected;

            need_left -= q;
            available_left -= q;
        }
    }

    selected = __shfl_sync(FULL_WARP_MASK, selected, 0);
    for (int i = 0; i < selected; ++i) {
        out_targets[i] = __shfl_sync(FULL_WARP_MASK, sel_rank[i], 0);
        out_quota[i] = __shfl_sync(FULL_WARP_MASK, sel_quota[i], 0);
    }
    return selected;
}

template <bool STORE_PLAN>
__device__ bool warp_build_export_plan(const int32_t* loads,
                                       const int* sorted_experts,
                                       int32_t* export_sum,
                                       int32_t* excess,
                                       int32_t* slack,
                                       int32_t* slots_used,
                                       const int32_t* rank_load,
                                       const int* source_order,
                                       uint64_t* occ_lo,
                                       uint64_t* occ_hi,
                                       ExportPlanEntry* export_plan,
                                       int& num_exports,
                                       int threshold,
                                       int G,
                                       int num_local_master,
                                       int num_local_redundant,
                                       int min_tokens_per_replica,
                                       bool allow_zero_master_quota,
                                       int lane_id,
                                       bool use_dynamic_q_floor = false,
                                       int batch_k = 1) {
    if (lane_id == 0) {
        num_exports = 0;
    }
    __syncwarp();

    int local_excess = 0;
    int local_slack = 0;
    for (int r = lane_id; r < G; r += 32) {
        local_excess += excess[r];
        local_slack += slack[r];
    }
    const int total_excess = warp_reduce_sum(local_excess);
    const int total_slack = warp_reduce_sum(local_slack);
    if (total_excess == 0) {
        return true;
    }
    if (total_excess > total_slack) {
        return false;
    }

    const bool use_register_state = (G <= 64);
    const int my_rank_0 = lane_id;
    const int my_rank_1 = lane_id + 32;
    int reg_slack_0 = 0;
    int reg_slack_1 = 0;
    int reg_slots_0 = 0;
    int reg_slots_1 = 0;
    if (use_register_state) {
        if (my_rank_0 < G) {
            reg_slack_0 = max(threshold - rank_load[my_rank_0], 0);
        }
        if (my_rank_1 < G) {
            reg_slack_1 = max(threshold - rank_load[my_rank_1], 0);
        }
    }

    int running_excess = total_excess;
    int running_slack = total_slack;
    for (int ord = 0; ord < G; ++ord) {
        const int source_rank_local = source_order[ord];
        int need = excess[source_rank_local];
        if (need <= 0) {
            continue;
        }
        int source_impossible = 0;
        if (lane_id == 0 && need > running_slack) {
            source_impossible = 1;
        }
        source_impossible = __shfl_sync(FULL_WARP_MASK, source_impossible, 0);
        if (source_impossible) {
            return false;
        }

        for (int pos = 0; pos < num_local_master; ++pos) {
            const int expert_local = sorted_experts[source_rank_local * num_local_master + pos];
            const int keep_on_master = (!allow_zero_master_quota && loads[expert_local] > 0) ? 1 : 0;
            if (use_register_state) {
                uint64_t local_occ_lo = occ_lo[expert_local];
                uint64_t local_occ_hi = occ_hi[expert_local];
                int local_export_sum = export_sum[expert_local];
                int available = max(loads[expert_local] - keep_on_master - local_export_sum, 0);
                const int batch_k_local = min(max(batch_k, 1), QUOTA_ORACLE_MAX_BATCH_K);

                while (need > 0 && available > 0) {
                    int effective_min_tokens = min_tokens_per_replica;
                    if (use_dynamic_q_floor) {
                        int local_feasible = 0;
                        const int rank0 = lane_id;
                        if (rank0 < G && reg_slack_0 > 0 && reg_slots_0 < num_local_redundant &&
                            !occ_has(local_occ_lo, local_occ_hi, rank0) && min(min(need, reg_slack_0), available) > 0) {
                            local_feasible += 1;
                        }
                        const int rank1 = lane_id + 32;
                        if (rank1 < G && reg_slack_1 > 0 && reg_slots_1 < num_local_redundant &&
                            !occ_has(local_occ_lo, local_occ_hi, rank1) && min(min(need, reg_slack_1), available) > 0) {
                            local_feasible += 1;
                        }
                        const int feasible_slots = warp_reduce_sum(local_feasible);
                        if (feasible_slots <= 0) {
                            break;
                        }
                        effective_min_tokens = max(1, (need + feasible_slots - 1) / feasible_slots);
                    }

                    if (batch_k_local > 1) {
                        int selected_targets[QUOTA_ORACLE_MAX_BATCH_K];
                        int selected_quota[QUOTA_ORACLE_MAX_BATCH_K];
                        const int selected = warp_collect_topk_targets_reg(reg_slack_0,
                                                                           reg_slack_1,
                                                                           reg_slots_0,
                                                                           reg_slots_1,
                                                                           local_occ_lo,
                                                                           local_occ_hi,
                                                                           need,
                                                                           available,
                                                                           effective_min_tokens,
                                                                           num_local_redundant,
                                                                           G,
                                                                           lane_id,
                                                                           batch_k_local,
                                                                           selected_targets,
                                                                           selected_quota);
                        if (selected <= 0) {
                            break;
                        }

                        if constexpr (STORE_PLAN) {
                            int can_store = 1;
                            if (lane_id == 0 && num_exports + selected > MAX_REPLICAS_PER_NVL) {
                                can_store = 0;
                            }
                            can_store = __shfl_sync(FULL_WARP_MASK, can_store, 0);
                            if (!can_store) {
                                return false;
                            }
                        }

                        for (int idx = 0; idx < selected; ++idx) {
                            const int target = selected_targets[idx];
                            const int q = selected_quota[idx];

                            if constexpr (STORE_PLAN) {
                                if (lane_id == 0) {
                                    export_plan[num_exports++] = {expert_local, target, q};
                                }
                            }
                            if (my_rank_0 == target) {
                                reg_slack_0 -= q;
                                reg_slots_0 += 1;
                            }
                            if (my_rank_1 == target) {
                                reg_slack_1 -= q;
                                reg_slots_1 += 1;
                            }
                            occ_set(local_occ_lo, local_occ_hi, target);
                            if (lane_id == 0) {
                                local_export_sum += q;
                                running_excess -= q;
                                running_slack -= q;
                            }
                            need -= q;
                            available -= q;
                        }
                    } else {
                        const int best_target = warp_find_best_target_reg(reg_slack_0,
                                                                          reg_slack_1,
                                                                          reg_slots_0,
                                                                          reg_slots_1,
                                                                          local_occ_lo,
                                                                          local_occ_hi,
                                                                          need,
                                                                          available,
                                                                          effective_min_tokens,
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

                        int my_best_slack = 0;
                        if (my_rank_0 == best_target) {
                            my_best_slack = reg_slack_0;
                        }
                        if (my_rank_1 == best_target) {
                            my_best_slack = reg_slack_1;
                        }
                        const int owner_lane = best_target & 31;
                        const int best_slack = __shfl_sync(FULL_WARP_MASK, my_best_slack, owner_lane);
                        int q = 0;
                        if (lane_id == 0) {
                            q = min(min(need, best_slack), available);
                        }
                        q = __shfl_sync(FULL_WARP_MASK, q, 0);
                        if (q < effective_min_tokens) {
                            break;
                        }

                        if constexpr (STORE_PLAN) {
                            if (lane_id == 0) {
                                export_plan[num_exports++] = {expert_local, best_target, q};
                            }
                        }
                        if (my_rank_0 == best_target) {
                            reg_slack_0 -= q;
                            reg_slots_0 += 1;
                        }
                        if (my_rank_1 == best_target) {
                            reg_slack_1 -= q;
                            reg_slots_1 += 1;
                        }
                        occ_set(local_occ_lo, local_occ_hi, best_target);
                        if (lane_id == 0) {
                            local_export_sum += q;
                            running_excess -= q;
                            running_slack -= q;
                        }

                        need -= q;
                        available -= q;
                    }
                }

                if (lane_id == 0) {
                    occ_lo[expert_local] = local_occ_lo;
                    occ_hi[expert_local] = local_occ_hi;
                    export_sum[expert_local] = local_export_sum;
                }
                __syncwarp();
            } else {
                int available = max(loads[expert_local] - keep_on_master - export_sum[expert_local], 0);

                while (need > 0 && available > 0) {
                    int effective_min_tokens = min_tokens_per_replica;
                    if (use_dynamic_q_floor) {
                        int local_feasible = 0;
                        for (int target_rank_local = lane_id; target_rank_local < G; target_rank_local += 32) {
                            if (slack[target_rank_local] > 0 && slots_used[target_rank_local] < num_local_redundant &&
                                !occ_has(occ_lo[expert_local], occ_hi[expert_local], target_rank_local) &&
                                min(min(need, slack[target_rank_local]), available) > 0) {
                                local_feasible += 1;
                            }
                        }
                        const int feasible_slots = warp_reduce_sum(local_feasible);
                        if (feasible_slots <= 0) {
                            break;
                        }
                        effective_min_tokens = max(1, (need + feasible_slots - 1) / feasible_slots);
                    }

                    const int best_target = warp_find_best_target(slack,
                                                                  slots_used,
                                                                  occ_lo,
                                                                  occ_hi,
                                                                  expert_local,
                                                                  need,
                                                                  available,
                                                                  effective_min_tokens,
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
                    if (q < effective_min_tokens) {
                        break;
                    }
                    if (lane_id == 0) {
                        if constexpr (STORE_PLAN) {
                            export_plan[num_exports++] = {expert_local, best_target, q};
                        }
                        export_sum[expert_local] += q;
                        slack[best_target] -= q;
                        slots_used[best_target] += 1;
                        occ_set(occ_lo[expert_local], occ_hi[expert_local], best_target);
                        running_excess -= q;
                        running_slack -= q;
                    }
                    __syncwarp();

                    need -= q;
                    available -= q;
                }
            }

            if (need == 0) {
                break;
            }
        }

        if (need > 0) {
            return false;
        }
        int early_exit = 0;
        if (lane_id == 0 && running_excess > running_slack) {
            early_exit = 1;
        }
        early_exit = __shfl_sync(FULL_WARP_MASK, early_exit, 0);
        if (early_exit) {
            return false;
        }
    }

    return true;
}

template <bool STORE_PLAN>
__device__ bool block_build_export_plan_parallel_v4c(const int32_t* loads,
                                                     const int* sorted_experts,
                                                     int32_t* export_sum,
                                                     int32_t* excess,
                                                     int32_t* slack,
                                                     int32_t* slots_used,
                                                     const int* source_order,
                                                     uint64_t* occ_lo,
                                                     uint64_t* occ_hi,
                                                     ExportPlanEntry* export_plan,
                                                     int& num_exports,
                                                     int G,
                                                     int num_local_master,
                                                     int num_local_redundant,
                                                     int min_tokens_per_replica,
                                                     bool allow_zero_master_quota,
                                                     bool use_dynamic_q_floor,
                                                     int* source_pos,
                                                     V4OracleTask* tasks,
                                                     int* num_tasks_ptr,
                                                     int* running_excess_ptr,
                                                     int* running_slack_ptr,
                                                     int* fail_flag_ptr) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (tid == 0) {
        num_exports = 0;
        *num_tasks_ptr = 0;
        *running_excess_ptr = 0;
        *running_slack_ptr = 0;
        *fail_flag_ptr = 0;
    }
    for (int r = tid; r < G; r += blockDim.x) {
        source_pos[r] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        int total_excess = 0;
        int total_slack = 0;
        for (int r = 0; r < G; ++r) {
            total_excess += excess[r];
            total_slack += slack[r];
        }
        *running_excess_ptr = total_excess;
        *running_slack_ptr = total_slack;
        if (total_excess > total_slack) {
            *fail_flag_ptr = 1;
        }
    }
    __syncthreads();

    if (*fail_flag_ptr) {
        return false;
    }
    if (*running_excess_ptr == 0) {
        return true;
    }

    const int max_rounds = MAX_REPLICAS_PER_NVL + MAX_EXPERTS_PER_NVL;
    for (int round = 0; round < max_rounds; ++round) {
        if (tid == 0) {
            *num_tasks_ptr = 0;
            if (*running_excess_ptr <= 0) {
                *num_tasks_ptr = -1;
            } else if (*running_excess_ptr > *running_slack_ptr) {
                *fail_flag_ptr = 1;
            } else {
                int task_count = 0;
                for (int ord = 0; ord < G && task_count < QUOTA_SOLVER_WARPS; ++ord) {
                    const int source_rank_local = source_order[ord];
                    const int need = excess[source_rank_local];
                    if (need <= 0) {
                        continue;
                    }

                    int pos = source_pos[source_rank_local];
                    int expert_local = -1;
                    int available = 0;
                    while (pos < num_local_master) {
                        const int candidate = sorted_experts[source_rank_local * num_local_master + pos];
                        const int keep_on_master = (!allow_zero_master_quota && loads[candidate] > 0) ? 1 : 0;
                        const int avail = max(loads[candidate] - keep_on_master - export_sum[candidate], 0);
                        if (avail > 0) {
                            expert_local = candidate;
                            available = avail;
                            break;
                        }
                        ++pos;
                    }
                    source_pos[source_rank_local] = pos;
                    if (expert_local < 0) {
                        *fail_flag_ptr = 1;
                        break;
                    }

                    V4OracleTask& task = tasks[task_count];
                    task.source_rank_local = source_rank_local;
                    task.expert_local = expert_local;
                    task.expert_pos = pos;
                    task.need = need;
                    task.available = available;
                    task.min_tokens = min_tokens_per_replica;
                    task.target_1 = -1;
                    task.quota_1 = 0;
                    task.target_2 = -1;
                    task.quota_2 = 0;
                    ++task_count;
                }

                if (!*fail_flag_ptr) {
                    *num_tasks_ptr = task_count;
                    if (task_count == 0) {
                        *fail_flag_ptr = 1;
                    }
                }
            }
        }
        __syncthreads();

        if (*fail_flag_ptr) {
            return false;
        }
        if (*num_tasks_ptr == -1) {
            return true;
        }
        const int num_tasks = *num_tasks_ptr;
        if (num_tasks <= 0) {
            return false;
        }

        if (warp_id < num_tasks) {
            const V4OracleTask task = tasks[warp_id];
            const int need = task.need;
            const int available = task.available;
            const int expert_local = task.expert_local;
            const uint64_t local_occ_lo = occ_lo[expert_local];
            const uint64_t local_occ_hi = occ_hi[expert_local];

            int reg_slack_0 = 0;
            int reg_slack_1 = 0;
            int reg_slots_0 = num_local_redundant;
            int reg_slots_1 = num_local_redundant;
            const int rank_0 = lane_id;
            const int rank_1 = lane_id + 32;
            if (rank_0 < G) {
                reg_slack_0 = slack[rank_0];
                reg_slots_0 = slots_used[rank_0];
            }
            if (rank_1 < G) {
                reg_slack_1 = slack[rank_1];
                reg_slots_1 = slots_used[rank_1];
            }

            int effective_min_tokens = min_tokens_per_replica;
            if (use_dynamic_q_floor) {
                int local_feasible = 0;
                if (rank_0 < G && reg_slack_0 > 0 && reg_slots_0 < num_local_redundant &&
                    !occ_has(local_occ_lo, local_occ_hi, rank_0) && min(min(need, reg_slack_0), available) > 0) {
                    local_feasible += 1;
                }
                if (rank_1 < G && reg_slack_1 > 0 && reg_slots_1 < num_local_redundant &&
                    !occ_has(local_occ_lo, local_occ_hi, rank_1) && min(min(need, reg_slack_1), available) > 0) {
                    local_feasible += 1;
                }
                const int feasible_slots = warp_reduce_sum(local_feasible);
                if (feasible_slots <= 0) {
                    effective_min_tokens = INT_MAX;
                } else {
                    effective_min_tokens = max(1, (need + feasible_slots - 1) / feasible_slots);
                }
            }

            int selected_targets[QUOTA_ORACLE_MAX_BATCH_K];
            int selected_quota[QUOTA_ORACLE_MAX_BATCH_K];
            int selected = 0;
            if (effective_min_tokens != INT_MAX) {
                selected = warp_collect_topk_targets_reg(reg_slack_0,
                                                         reg_slack_1,
                                                         reg_slots_0,
                                                         reg_slots_1,
                                                         local_occ_lo,
                                                         local_occ_hi,
                                                         need,
                                                         available,
                                                         effective_min_tokens,
                                                         num_local_redundant,
                                                         G,
                                                         lane_id,
                                                         2,
                                                         selected_targets,
                                                         selected_quota);
            }

            if (lane_id == 0) {
                tasks[warp_id].min_tokens =
                    (effective_min_tokens == INT_MAX) ? min_tokens_per_replica : effective_min_tokens;
                tasks[warp_id].target_1 = (selected > 0) ? selected_targets[0] : -1;
                tasks[warp_id].quota_1 = (selected > 0) ? selected_quota[0] : 0;
                tasks[warp_id].target_2 = (selected > 1) ? selected_targets[1] : -1;
                tasks[warp_id].quota_2 = (selected > 1) ? selected_quota[1] : 0;
            }
        }
        __syncthreads();

        if (tid == 0) {
            int progress = 0;

            for (int w = 0; w < num_tasks; ++w) {
                const V4OracleTask task = tasks[w];
                const int source_rank_local = task.source_rank_local;
                const int expert_local = task.expert_local;

                int need = excess[source_rank_local];
                if (need <= 0) {
                    continue;
                }

                const int keep_on_master = (!allow_zero_master_quota && loads[expert_local] > 0) ? 1 : 0;
                const int available = max(loads[expert_local] - keep_on_master - export_sum[expert_local], 0);
                if (available <= 0) {
                    source_pos[source_rank_local] = max(source_pos[source_rank_local], task.expert_pos + 1);
                    continue;
                }

                const int min_tokens = max(task.min_tokens, 1);
                int chosen_target = -1;
                int chosen_quota = 0;
                for (int candidate_idx = 0; candidate_idx < 2; ++candidate_idx) {
                    const int target = (candidate_idx == 0) ? task.target_1 : task.target_2;
                    if (target < 0 || target >= G) {
                        continue;
                    }
                    if (slack[target] <= 0 || slots_used[target] >= num_local_redundant) {
                        continue;
                    }
                    if (occ_has(occ_lo[expert_local], occ_hi[expert_local], target)) {
                        continue;
                    }

                    const int q = min(min(need, slack[target]), available);
                    if (q < min_tokens) {
                        continue;
                    }
                    chosen_target = target;
                    chosen_quota = q;
                    break;
                }

                if (chosen_target < 0) {
                    int best_cap = -1;
                    for (int target = 0; target < G; ++target) {
                        if (slack[target] <= 0 || slots_used[target] >= num_local_redundant) {
                            continue;
                        }
                        if (occ_has(occ_lo[expert_local], occ_hi[expert_local], target)) {
                            continue;
                        }
                        const int q = min(min(need, slack[target]), available);
                        if (q < min_tokens) {
                            continue;
                        }
                        if (q > best_cap || (q == best_cap && (chosen_target < 0 || target < chosen_target))) {
                            chosen_target = target;
                            chosen_quota = q;
                            best_cap = q;
                        }
                    }
                }

                if (chosen_target >= 0) {
                    if constexpr (STORE_PLAN) {
                        if (num_exports >= MAX_REPLICAS_PER_NVL) {
                            *fail_flag_ptr = 1;
                            break;
                        }
                        export_plan[num_exports++] = {expert_local, chosen_target, chosen_quota};
                    }
                    export_sum[expert_local] += chosen_quota;
                    excess[source_rank_local] -= chosen_quota;
                    slack[chosen_target] -= chosen_quota;
                    slots_used[chosen_target] += 1;
                    occ_set(occ_lo[expert_local], occ_hi[expert_local], chosen_target);

                    *running_excess_ptr -= chosen_quota;
                    *running_slack_ptr -= chosen_quota;
                    progress = 1;

                    const int remain_available =
                        max(loads[expert_local] - keep_on_master - export_sum[expert_local], 0);
                    if (remain_available <= 0) {
                        source_pos[source_rank_local] = max(source_pos[source_rank_local], task.expert_pos + 1);
                    }
                } else {
                    source_pos[source_rank_local] = max(source_pos[source_rank_local], task.expert_pos + 1);
                }
            }

            if (*running_excess_ptr > *running_slack_ptr) {
                *fail_flag_ptr = 1;
            } else if (!progress && *running_excess_ptr > 0) {
                bool any_candidate = false;
                for (int ord = 0; ord < G && !any_candidate; ++ord) {
                    const int source_rank_local = source_order[ord];
                    if (excess[source_rank_local] <= 0) {
                        continue;
                    }
                    int pos = source_pos[source_rank_local];
                    while (pos < num_local_master) {
                        const int candidate = sorted_experts[source_rank_local * num_local_master + pos];
                        const int keep_on_master = (!allow_zero_master_quota && loads[candidate] > 0) ? 1 : 0;
                        const int available = max(loads[candidate] - keep_on_master - export_sum[candidate], 0);
                        if (available > 0) {
                            any_candidate = true;
                            break;
                        }
                        ++pos;
                    }
                }
                if (!any_candidate) {
                    *fail_flag_ptr = 1;
                }
            }
        }
        __syncthreads();
        if (*fail_flag_ptr) {
            return false;
        }
    }

    return (*running_excess_ptr == 0);
}

__device__ __noinline__ void build_rank_quota_prefix_slow(int expert_local,
                                                          int row_offset,
                                                          int C,
                                                          const int32_t* smem_my_loads,
                                                          const int32_t* smem_domain_loads,
                                                          int stride_elems,
                                                          int domain_start_rank,
                                                          int num_local_physical,
                                                          int my_rank,
                                                          int G,
                                                          bool locality_aware,
                                                          const int32_t* quota,
                                                          const int32_t* l2p_map,
                                                          int32_t* rank_quota_prefix) {
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
                         (host_rank[j] < host_rank[best_j] || (host_rank[j] == host_rank[best_j] && j < best_j)))) {
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


__global__ __launch_bounds__(QUOTA_SOLVER_THREADS) void quota_placement_solve_kernel(
    const int32_t* __restrict__ expert_loads,
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
    float oracle_eps,
    int kernel_stage,
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
    const int clamped_kernel_stage = max(min(kernel_stage, 1), 0);
    const bool enable_v4a = (clamped_kernel_stage >= 1);
    const bool use_fast_t_oracle = true;
    const bool use_dynamic_q_floor = true;
    const int oracle_batch_k = 1;

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
    __shared__ int32_t smem_rank_phys_base[MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_my_loads[MAX_EXPERTS_PER_NVL];
    __shared__ int smem_sorted_experts[MAX_EXPERTS_PER_NVL];
    __shared__ int smem_presorted_source[MAX_GPUS_PER_NVL];
    __shared__ ExportPlanEntry smem_export_plan[MAX_REPLICAS_PER_NVL];
    __shared__ int smem_num_exports;
    __shared__ volatile int smem_tma_wait_done;
    __shared__ ptx::arrival_phase smem_tma_phase;
    __shared__ int smem_next_slot[MAX_GPUS_PER_NVL];
    __shared__ int smem_expert_slot[MAX_EXPERTS_PER_NVL];

    __shared__ int smem_bs_lo;
    __shared__ int smem_bs_hi;
    __shared__ bool smem_precheck_done;
    __shared__ int smem_fast_plan_done;
    __shared__ int smem_probes[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_valid[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_feasible[QUOTA_SOLVER_WARPS];
    __shared__ int smem_probe_small_range;

    __shared__ int32_t smem_warp_export_sum[QUOTA_SOLVER_WARPS][MAX_EXPERTS_PER_NVL];
    __shared__ int32_t smem_warp_excess[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_warp_slack[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];
    __shared__ int32_t smem_warp_slots_used[QUOTA_SOLVER_WARPS][MAX_GPUS_PER_NVL];

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

    if (warp_id == 0) {
        warp_sort_source_ranks_by_load(smem_presorted_source, smem_rank_load, G, lane_id);
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
                const bool cur_better =
                    (smem_loads[cur] > smem_loads[key]) || (smem_loads[cur] == smem_loads[key] && cur < key);
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
    for (int r = tid; r < G; r += blockDim.x) {
        smem_rank_phys_base[r] = (domain_start_rank + r) * num_local_physical + num_local_master;
    }
    if (tid == 0) {
        smem_tma_wait_done = 0;
    }

    const bool can_use_tma = (domain_start_log % 4 == 0) && (E % 4 == 0) && (num_global_logical_experts % 4 == 0);
    ptx::mbarrier* mbar = nullptr;
    if (can_use_tma) {
        mbar = ptx::create_mbarrier();
        if (tid == 0) {
            ptx::mbarrier_init(mbar, 1);
            smem_tma_phase = 0;
            const int total_bytes = G * E * static_cast<int>(sizeof(int32_t));
            ptx::mbarrier_arrive_and_set_tx(mbar, total_bytes);
            for (int r = 0; r < G; ++r) {
                ptx::tma_load_1d(
                    smem_domain_loads + r * stride_elems,
                    expert_loads_per_rank + (domain_start_rank + r) * num_global_logical_experts + domain_start_log,
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
    const bool do_binary_search = (num_local_redundant > 0 && G > 1);
    if (enable_v4a) {
        if (tid == 0) {
            smem_num_exports = 0;
            if (do_binary_search) {
                smem_fast_plan_done = 0;
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
        }
        __syncthreads();
    } else {
        __syncthreads();
        if (tid == 0) {
            smem_num_exports = 0;
        }
        __syncthreads();
    }

    if (do_binary_search) {
        if (!enable_v4a) {
            if (tid == 0) {
                smem_fast_plan_done = 0;
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
        }

        if (use_fast_t_oracle) {
            int fast_threshold = smem_bs_lo;
            if (warp_id == 0) {
                if (lane_id == 0) {
                    const float eps = fmaxf(oracle_eps, 0.0f);
                    fast_threshold = static_cast<int>(ceilf(static_cast<float>(smem_bs_lo) * (1.0f + eps)));
                    fast_threshold = max(fast_threshold, smem_bs_lo);
                }
                fast_threshold = __shfl_sync(FULL_WARP_MASK, fast_threshold, 0);
            }
            if (warp_id == 0) {
                const int fast_step = max(1, (smem_bs_hi + 99) / 100);
                bool feasible_fast = false;
                for (int retry = 0; retry <= QUOTA_ORACLE_FAST_MAX_RETRY && !feasible_fast; ++retry) {
                    warp_init_oracle_state_parallel(smem_warp_export_sum[0],
                                                    smem_warp_occ_lo_base,
                                                    smem_warp_occ_hi_base,
                                                    smem_warp_excess[0],
                                                    smem_warp_slack[0],
                                                    smem_warp_slots_used[0],
                                                    smem_rank_load,
                                                    fast_threshold,
                                                    E,
                                                    G,
                                                    num_local_master,
                                                    lane_id);
                    __syncwarp();

                    int fast_exports = 0;
                    feasible_fast = warp_build_export_plan<true>(smem_loads,
                                                                 smem_sorted_experts,
                                                                 smem_warp_export_sum[0],
                                                                 smem_warp_excess[0],
                                                                 smem_warp_slack[0],
                                                                 smem_warp_slots_used[0],
                                                                 smem_rank_load,
                                                                 smem_presorted_source,
                                                                 smem_warp_occ_lo_base,
                                                                 smem_warp_occ_hi_base,
                                                                 smem_export_plan,
                                                                 fast_exports,
                                                                 fast_threshold,
                                                                 G,
                                                                 num_local_master,
                                                                 num_local_redundant,
                                                                 min_tokens_per_replica,
                                                                 allow_zero_master_quota,
                                                                 lane_id,
                                                                 use_dynamic_q_floor,
                                                                 oracle_batch_k);
                    if (lane_id == 0) {
                        if (feasible_fast) {
                            smem_num_exports = fast_exports;
                            smem_fast_plan_done = 1;
                            smem_bs_lo = fast_threshold;
                        } else {
                            fast_threshold = min(smem_bs_hi, fast_threshold + fast_step);
                        }
                    }
                    fast_threshold = __shfl_sync(FULL_WARP_MASK, fast_threshold, 0);
                }
            }
            __syncthreads();
        }
        if (!smem_fast_plan_done) {
            if (warp_id == 0) {
                int coarse_lo = smem_bs_lo;
                int coarse_hi = smem_bs_hi;
                while (coarse_lo < coarse_hi) {
                    const int probe = coarse_lo + ((coarse_hi - coarse_lo) >> 1);
                    const bool feasible = warp_capacity_feasible(smem_rank_load, probe, G, lane_id);
                    if (lane_id == 0) {
                        if (feasible) {
                            coarse_hi = probe;
                        } else {
                            coarse_lo = probe + 1;
                        }
                    }
                    coarse_lo = __shfl_sync(FULL_WARP_MASK, coarse_lo, 0);
                    coarse_hi = __shfl_sync(FULL_WARP_MASK, coarse_hi, 0);
                }
                if (lane_id == 0) {
                    smem_bs_lo = coarse_lo;
                }
                __syncwarp();

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
                                                                            smem_rank_load,
                                                                            smem_presorted_source,
                                                                            smem_warp_occ_lo_base,
                                                                            smem_warp_occ_hi_base,
                                                                            smem_export_plan,
                                                                            precheck_exports,
                                                                            smem_bs_lo,
                                                                            G,
                                                                            num_local_master,
                                                                            num_local_redundant,
                                                                            min_tokens_per_replica,
                                                                            allow_zero_master_quota,
                                                                            lane_id,
                                                                            false,
                                                                            1);
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
                            smem_probes[w] = smem_bs_lo +
                                static_cast<int>((static_cast<int64_t>(range) * (w + 1)) / (QUOTA_SOLVER_WARPS + 1));
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
                                                             smem_rank_load,
                                                             smem_presorted_source,
                                                             smem_warp_occ_lo_base + warp_id * E,
                                                             smem_warp_occ_hi_base + warp_id * E,
                                                             nullptr,
                                                             dummy_exports,
                                                             smem_probes[warp_id],
                                                             G,
                                                             num_local_master,
                                                             num_local_redundant,
                                                             min_tokens_per_replica,
                                                             allow_zero_master_quota,
                                                             lane_id,
                                                             false,
                                                             1);
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
                                                                             smem_rank_load,
                                                                             smem_presorted_source,
                                                                             smem_warp_occ_lo_base,
                                                                             smem_warp_occ_hi_base,
                                                                             smem_export_plan,
                                                                             final_exports,
                                                                             smem_bs_lo,
                                                                             G,
                                                                             num_local_master,
                                                                             num_local_redundant,
                                                                             min_tokens_per_replica,
                                                                             allow_zero_master_quota,
                                                                             lane_id,
                                                                             false,
                                                                             1);
                    if (lane_id == 0) {
                        EP_DEVICE_ASSERT(final_feasible);
                        smem_num_exports = final_exports;
                    }
                }
                __syncthreads();
            }
        }
    }

    const int32_t* final_export_sum = smem_warp_export_sum[0];
    for (int r = tid; r < G; r += blockDim.x) {
        smem_next_slot[r] = 0;
    }
    for (int expert_local = tid; expert_local < E; expert_local += blockDim.x) {
        const int l_global = domain_start_log + expert_local;
        const int row_offset = l_global * max_replicas_dim;
        const int master_quota =
            do_binary_search ? (smem_loads[expert_local] - final_export_sum[expert_local]) : smem_loads[expert_local];
        quota[row_offset] = master_quota;
        smem_expert_slot[expert_local] = 1;
    }
    __syncthreads();

    if (tid == 0) {
        for (int plan_idx = 0; plan_idx < smem_num_exports; ++plan_idx) {
            const ExportPlanEntry entry = smem_export_plan[plan_idx];
            const int expert_local = entry.expert_local;
            const int l_global = domain_start_log + expert_local;
            const int row_offset = l_global * max_replicas_dim;
            const int slot = smem_expert_slot[expert_local]++;
            const int target_local = entry.target_rank_local;
            const int phys_base = enable_v4a
                                      ? smem_rank_phys_base[target_local]
                                      : ((domain_start_rank + target_local) * num_local_physical + num_local_master);
            const int phys_idx = phys_base + smem_next_slot[target_local]++;
            p2l_map[phys_idx] = l_global;
            l2p_map[row_offset + slot] = phys_idx;
            quota[row_offset + slot] = entry.quota;
        }
    }
    __syncthreads();

    for (int expert_local = tid; expert_local < E; expert_local += blockDim.x) {
        const int l_global = domain_start_log + expert_local;
        const int row_offset = l_global * max_replicas_dim;
        const int C = smem_expert_slot[expert_local];
        smem_c[expert_local] = C;
        lcnts[l_global] = C;

        int prefix = 0;
        for (int j = 0; j < C; ++j) {
            prefix += quota[row_offset + j];
            quota_prefix[row_offset + j] = prefix;
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
        const int row_offset = (domain_start_log + expert_local) * max_replicas_dim;
        const int C = smem_c[expert_local];

        if (C <= QUOTA_FAST_REPLICA_LIMIT) {
            int host_rank[QUOTA_FAST_REPLICA_LIMIT];
            int my_alloc[QUOTA_FAST_REPLICA_LIMIT];
            int64_t remainders[QUOTA_FAST_REPLICA_LIMIT];
            for (int j = 0; j < C; ++j) {
                const int phys_idx = l2p_map[row_offset + j];
                host_rank[j] = phys_idx / num_local_physical;
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
                int remote_cap[QUOTA_FAST_REPLICA_LIMIT];
                int host_local_ids[QUOTA_FAST_REPLICA_LIMIT];
                int host_remaining[QUOTA_FAST_REPLICA_LIMIT];
                int num_hosts = 0;

                int rem_my = smem_my_loads[expert_local];
                int total_remote_cap = 0;
                for (int j = 0; j < C; ++j) {
                    const int q = quota[row_offset + j];
                    const int host_local = host_rank[j] - domain_start_rank;
                    EP_DEVICE_ASSERT(host_local >= 0 && host_local < G);

                    int host_slot = -1;
                    for (int u = 0; u < num_hosts; ++u) {
                        if (host_local_ids[u] == host_local) {
                            host_slot = u;
                            break;
                        }
                    }
                    if (host_slot < 0) {
                        EP_DEVICE_ASSERT(num_hosts < QUOTA_FAST_REPLICA_LIMIT);
                        host_slot = num_hosts;
                        host_local_ids[num_hosts] = host_local;
                        host_remaining[num_hosts] = smem_domain_loads[host_local * stride_elems + expert_local];
                        ++num_hosts;
                    }

                    const int local_fill = min(host_remaining[host_slot], q);
                    host_remaining[host_slot] -= local_fill;
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
        } else {
            build_rank_quota_prefix_slow(expert_local,
                                         row_offset,
                                         C,
                                         smem_my_loads,
                                         smem_domain_loads,
                                         stride_elems,
                                         domain_start_rank,
                                         num_local_physical,
                                         my_rank,
                                         G,
                                         locality_aware,
                                         quota,
                                         l2p_map,
                                         rank_quota_prefix);
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
    const int num_local_physical_experts = num_local_master_experts + num_local_redundant_experts;
    const int num_global_physical_experts = num_local_physical_experts * num_ranks;
    const int num_nvl_domains = num_ranks / num_nvl_ranks;
    const int num_logical_per_nvl = num_local_master_experts * num_nvl_ranks;
    const int num_redundant_per_nvl = num_local_redundant_experts * num_nvl_ranks;

    EP_HOST_ASSERT(num_nvl_ranks > 0 && num_ranks % num_nvl_ranks == 0);
    EP_HOST_ASSERT(num_local_master_experts > 0);
    EP_HOST_ASSERT(num_local_redundant_experts >= 0);
    EP_HOST_ASSERT(max_replicas_dim >= 1);
    EP_HOST_ASSERT(num_logical_per_nvl <= MAX_EXPERTS_PER_NVL);
    EP_HOST_ASSERT(num_nvl_ranks <= MAX_GPUS_PER_NVL);
    EP_HOST_ASSERT(num_redundant_per_nvl <= MAX_REPLICAS_PER_NVL);
    EP_HOST_ASSERT((kernel_stage == 0 || kernel_stage == 1) &&
                   "quota kernel_stage supports only {0,1}; stage 2/3 has been removed");
    if (num_nvl_domains == 0) {
        return;
    }

    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(physical_to_logical_map, 0xFF, static_cast<size_t>(num_global_physical_experts) * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        logical_to_physical_map,
        0xFF,
        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t),
        stream));
    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(logical_replica_counts, 0, static_cast<size_t>(num_global_logical_experts) * sizeof(int32_t), stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        logical_instance_quota,
        0,
        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t),
        stream));
    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(logical_instance_quota_prefix,
                        0,
                        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t),
                        stream));
    CUDA_RUNTIME_CHECK(
        cudaMemsetAsync(rank_quota_prefix,
                        0,
                        static_cast<size_t>(num_global_logical_experts) * max_replicas_dim * sizeof(int32_t),
                        stream));

    const int stride_elems = ((num_logical_per_nvl + 3) / 4) * 4;
    const size_t domain_loads_bytes = static_cast<size_t>(num_nvl_ranks) * stride_elems * sizeof(int32_t);
    const size_t occ_offset = (domain_loads_bytes + 7u) & ~size_t(7);
    const int my_rank = runtime::is_runtime_initialized ? runtime::rank_idx : 0;
    const dim3 grid(num_nvl_domains);

    const size_t occ_bytes = 2 * static_cast<size_t>(QUOTA_SOLVER_WARPS) * num_logical_per_nvl * sizeof(uint64_t);
    const size_t shared_memory_bytes = occ_offset + occ_bytes;
    const auto config = make_launch_config(grid, dim3(QUOTA_SOLVER_THREADS), stream, shared_memory_bytes);
    launch_kernel(quota_placement_solve_kernel,
                  config,
                  expert_loads,
                  expert_loads_per_rank,
                  physical_to_logical_map,
                  logical_to_physical_map,
                  logical_replica_counts,
                  logical_instance_quota,
                  logical_instance_quota_prefix,
                  rank_quota_prefix,
                  num_ranks,
                  num_nvl_ranks,
                  num_local_master_experts,
                  num_local_redundant_experts,
                  num_local_physical_experts,
                  max_replicas_dim,
                  num_global_logical_experts,
                  num_logical_per_nvl,
                  balance_threshold,
                  min_tokens_per_replica,
                  allow_zero_master_quota,
                  locality_aware,
                  oracle_eps,
                  kernel_stage,
                  my_rank);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace ultra_ep::kernels
