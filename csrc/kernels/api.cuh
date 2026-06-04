#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "config.cuh"

namespace ultra_ep::kernels {

struct GradReduceTask {
    float* master_local_addr;
    float* replica_remote_addr;
    size_t numel;
};

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
                     float balance_threshold = 1.0f,
                     int32_t min_tokens_per_replica = 1,
                     bool allow_zero_master_quota = true,
                     bool locality_aware = true,
                     float oracle_eps = 0.01f,
                     int kernel_stage = 1,
                     int rank_quota_source_rank = -1);

void init_master_placement(int32_t* physical_to_logical_map,
                           int32_t* logical_to_physical_map,
                           int32_t* logical_replica_counts,
                           int32_t* logical_instance_quota,
                           int32_t* logical_instance_quota_prefix,
                           int32_t* rank_quota_prefix,
                           cudaStream_t stream,
                           int num_global_physical_experts,
                           int num_global_logical_experts,
                           int num_ranks,
                           int num_local_master_experts,
                           int num_local_redundant_experts,
                           int max_replicas_dim);

namespace legacy {

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
                     float balance_threshold = 1.0f,
                     int32_t min_tokens_per_replica = 1,
                     bool allow_zero_master_quota = true,
                     bool locality_aware = true,
                     float oracle_eps = 0.01f,
                     int kernel_stage = 1);

}  // namespace legacy

// Run gradient reduce using the device-resident task list produced by the task-build kernel.
void run_grad_reduce(GradReduceTask* tasks,
                     int* task_tile_offsets,
                     int* task_metadata,
                     int* global_tile_counter,
                     cudaStream_t stream,
                     int num_sms,
                     bool deterministic = false);

// ============================================================================
// Weight Sync: Broadcast master weights to replicas
// ============================================================================

enum class WeightSyncPlanMode : int32_t {
    kDirect = 0,
    kAdaptive = 1,
    kForceRelay = 2,
};

// A broadcast task from one master to multiple replicas
// This structure enables loading SMEM once and TMA storing to multiple destinations
struct WeightSyncTask {
    __nv_bfloat16* master_local_addr;                            // Source: local master weight
    __nv_bfloat16* replica_remote_addrs[kMaxNvlDomainSize - 1];  // Destinations: replica addresses
    int num_replicas;                                            // Number of replicas (1 to kMaxNvlDomainSize-1)
    size_t numel;                                                // Number of elements
    int wait_ready_slot;                                         // Local relay-ready flag slot, -1 if no wait
    int num_ready_signals;                                       // Number of remote relay-ready flags to set
    int ready_signal_slots[kMaxNvlDomainSize - 1];               // Symmetric ready-flag slot per relay
    int ready_signal_nvl_ranks[kMaxNvlDomainSize - 1];           // Target NVL-domain rank slot per relay-ready signal
};

// Run weight sync using the device-resident task list produced by the task-build kernel.
void run_weight_sync(WeightSyncTask* tasks,
                     int* task_tile_offsets,
                     int* task_metadata,
                     int* global_tile_counter,
                     int* task_remaining_tiles,
                     uint64_t* local_ready_flags,
                     uint64_t* const* remote_ready_flag_ptrs,
                     uint64_t current_epoch,
                     cudaStream_t stream,
                     int num_device_sms,
                     int num_nvl_ranks,
                     int max_possible_tiles,
                     int cta_multiplier = 2);

// ============================================================================
// GPU Task Build: Build weight_sync/grad_reduce tasks entirely on GPU
// ============================================================================

// Immutable config for device task-build kernels (copied once in Manager ctor)
struct TaskBuildConfig {
    int rank_idx;
    int nvl_rank_idx;
    int num_nvl_ranks;
    int num_local_master_experts;
    int num_local_physical_experts;
    int num_local_redundant_experts;
    int64_t expert_fc1_numel;
    int64_t expert_fc2_numel;
    int64_t expert_total_numel;
    int max_replicas_dim;
    int weight_sync_plan_mode;
    int weight_sync_relay_min_replicas;
    int weight_sync_relay_max_relays;
    int weight_sync_relay_min_fanout_gain;
};

// Build stage-1 and stage-2 weight sync task arrays on GPU from placement data.
// Stage 1 seeds relays (or directly fans out); stage 2 waits on per-relay ready
// flags and forwards chunked relay traffic to downstream replicas.
void build_weight_sync_task_lists(const TaskBuildConfig* config,
                                  const int32_t* physical_to_logical_map,
                                  const int32_t* logical_to_physical_map,
                                  const int32_t* logical_replica_counts,
                                  void* const* remote_weight_ptrs,
                                  const int64_t* local_master_fc1_ptrs,
                                  const int64_t* local_master_fc2_ptrs,
                                  __nv_bfloat16* local_replica_weight_buffer,
                                  WeightSyncTask* stage1_tasks,
                                  int* stage1_task_tile_offsets,
                                  int* stage1_task_metadata,
                                  int* stage1_task_remaining_tiles,
                                  int* stage1_global_tile_counter,
                                  WeightSyncTask* stage2_tasks,
                                  int* stage2_task_tile_offsets,
                                  int* stage2_task_metadata,
                                  int* stage2_global_tile_counter,
                                  cudaStream_t stream);

// Build grad reduce task array on GPU from placement data.
void build_grad_reduce_tasks(const TaskBuildConfig* config,
                             const int32_t* physical_to_logical_map,
                             const int32_t* logical_to_physical_map,
                             const int32_t* logical_replica_counts,
                             void* const* remote_grad_ptrs,
                             const int64_t* local_master_fc1_ptrs,
                             const int64_t* local_master_fc2_ptrs,
                             GradReduceTask* tasks,
                             int* task_tile_offsets,
                             int* task_metadata,
                             int* global_task_or_tile_counter,
                             cudaStream_t stream);

// ============================================================================
// Reroute: Expand logical routing map to physical routing map (CUDA path)
// ============================================================================

// Forward (two-pass): scatter probs from [T,L] logical to [T,P] physical space.
// Pass 1 counts active tokens per (expert, tile), pass 2 uses prefix sums for
// deterministic round-robin scatter.  Grid: ceil(L/8) × ceil(T/128) blocks.
//
// Parameters:
//   routing_map:           [T, L] bool, device
//   probs:                 [T, L] scalar_t (float/bf16), device
//   l2p_map:               [L, max_replicas] int32, device (this layer's slice)
//   lcnts:                 [L] int32, device (this layer's slice)
//   expanded_routing_map:  [T, P] bool, device, output (must be zero-initialized)
//   expanded_probs:        [T, P] scalar_t, device, output (must be zero-initialized)
//   tile_counts:           [L, num_tiles] int32, device, scratch buffer
//   T, L, P, max_replicas: dimensions
//   stream: CUDA stream
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
                                           cudaStream_t stream);

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
                                     cudaStream_t stream);

// Backward (row-parallel gather): gather gradients from [T,P] physical to [T,L] logical.
// Each thread handles one (t, l) pair.  For active pairs, searches the forward's
// expanded_routing_map to find the assigned physical replica, then gathers the gradient.
// Eliminates the serial round-robin recomputation from the old backward kernel.
//
// Parameters:
//   grad_expanded_probs:   [T, P] scalar_t, device
//   routing_map:           [T, L] bool, device (saved from forward)
//   expanded_routing_map:  [T, P] bool, device (produced by forward, same layer)
//   l2p_map:               [L, max_replicas] int32, device
//   lcnts:                 [L] int32, device
//   grad_probs:            [T, L] scalar_t, device, output (must be zero-initialized)
//   T, L, P, max_replicas: dimensions
//   stream: CUDA stream
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
                                cudaStream_t stream);

void rmap_local_sum(int num_tokens,                  // T
                    int num_global_logical_experts,  // L
                    const bool* routing_map_ptr,     // [T, L] bool
                    int32_t* expert_loads_ptr,       // [L] int32, alloc by nvshmem
                    cudaStream_t stream);

// ============================================================================
// Sparse topk format support (for frameworks using topk_ids instead of routing_map)
// ============================================================================

// Compute per-expert token counts from sparse topk_ids
// Replaces rmap_local_sum for sparse topk format.
//   topk_ids_ptr: [T, K] int64, device — each entry is a logical expert ID
//   expert_loads_ptr: [L] int32, allocated by nvshmem — output (global loads)
void topk_local_sum(const int64_t* topk_ids_ptr,
                    const int num_tokens,
                    const int top_k,
                    const int num_global_logical_experts,
                    int32_t* expert_loads_ptr,
                    cudaStream_t stream);

void reduce_per_rank_loads(const int32_t* loads_per_rank, int32_t* global_loads, int G, int L, cudaStream_t stream);

// In-place remap topk_ids from logical to physical expert IDs using current
// placement. The sparse path uses a per-expert local ordinal produced by
// atomicAdd on `counters`, then:
//   - non-quota mode: local_ordinal % replica_count
//   - quota mode:     upper_bound(rank_quota_prefix, local_ordinal)
//   topk_ids_ptr: [T, K] int64, device — modified in place
//   logical_to_physical_map: [L, max_replicas] int32, device — logical-to-physical map
//   logical_replica_counts: [L] int32, device — replica counts per logical expert
//   counters: [L] int32, device scratch — zeroed internally each call
void run_sparse_reroute_round_robin(int64_t* topk_ids_ptr,
                                    const int32_t* logical_to_physical_map,
                                    const int32_t* logical_replica_counts,
                                    int* counters,
                                    const int num_tokens,
                                    const int top_k,
                                    const int num_global_logical_experts,
                                    const int max_replicas,
                                    cudaStream_t stream);

void run_sparse_reroute_quota(int64_t* topk_ids_ptr,
                              const int32_t* logical_to_physical_map,
                              const int32_t* logical_replica_counts,
                              const int32_t* rank_quota_prefix,
                              int* counters,
                              const int num_tokens,
                              const int top_k,
                              const int num_global_logical_experts,
                              const int max_replicas,
                              cudaStream_t stream);

// Helper functions
static __host__ __device__ __forceinline__ int ceil_div(const int a, const int b) {
    return (a + b - 1) / b;
}
static __host__ __device__ __forceinline__ int64_t ceil_div(const int64_t a, const int64_t b) {
    return static_cast<int>((a + b - 1) / b);
}
static __host__ __device__ __forceinline__ int weight_sync_num_tiles(const size_t numel) {
    return static_cast<int>((numel + kWeightSyncTileElements - 1) / kWeightSyncTileElements);
}
static __host__ __device__ __forceinline__ int weight_sync_num_chunks(const size_t numel) {
    return ceil_div(weight_sync_num_tiles(numel), kWeightSyncRelayChunkTiles);
}

}  // namespace ultra_ep::kernels
