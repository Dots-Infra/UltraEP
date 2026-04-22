#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

#include "../config.hpp"

namespace ultra_ep::kernels {

struct GradReduceTask {
    float* master_local_addr;
    float* replica_remote_addr;
    size_t numel;
};

// Run gradient reduce operation across all tasks (CPU task-build path)
// task_metadata_gpu: device buffer [2], written by this function as {total_tasks, total_tiles}
void run_grad_reduce_low_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                            GradReduceTask* grad_reduce_tasks_gpu,
                            int* global_task_counter_gpu,
                            int* task_metadata_gpu,
                            const int total_tasks,
                            cudaStream_t stream,
                            const int num_device_sms);
void run_grad_reduce_high_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                             GradReduceTask* grad_reduce_tasks_gpu,
                             int* global_tile_counter_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             const int total_tasks,
                             cudaStream_t stream,
                             const int num_device_sms,
                             int num_ctas_override = 0);

// Run gradient reduce using GPU-resident tasks (no H2D copy)
// task_metadata_gpu must already contain {total_tasks, total_tiles} (written by build kernel)
void run_grad_reduce_low_sm_from_gpu(GradReduceTask* grad_reduce_tasks_gpu,
                                     int* global_task_counter_gpu,
                                     int* task_metadata_gpu,
                                     cudaStream_t stream,
                                     int num_device_sms,
                                     int max_possible_tasks);
void run_grad_reduce_high_sm_from_gpu(GradReduceTask* grad_reduce_tasks_gpu,
                                      int* task_tile_offsets_gpu,
                                      int* task_metadata_gpu,
                                      int* global_tile_counter_gpu,
                                      cudaStream_t stream,
                                      int num_device_sms,
                                      int max_possible_tiles,
                                      int num_ctas_override = 0);

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
    __nv_bfloat16* master_local_addr;                              // Source: local master weight
    __nv_bfloat16* replica_remote_addrs[MAX_NVL_DOMAIN_SIZE - 1];  // Destinations: replica addresses
    int num_replicas;                                              // Number of replicas (1 to MAX_NVL_DOMAIN_SIZE-1)
    size_t numel;                                                  // Number of elements
    int wait_ready_slot;                                           // Local relay-ready flag slot, -1 if no wait
    int num_ready_signals;                                         // Number of remote relay-ready flags to set
    int ready_signal_slots[MAX_NVL_DOMAIN_SIZE - 1];               // Symmetric ready-flag slot per relay
    int ready_signal_nvl_ranks[MAX_NVL_DOMAIN_SIZE - 1];           // Target NVL-domain rank slot per relay-ready signal
};

// Run weight sync operation — CPU task-build path
// task_metadata_gpu: device buffer [2], written by this function as {total_tasks, total_tiles}
void run_weight_sync(const WeightSyncTask* weight_sync_tasks_cpu,
                     WeightSyncTask* weight_sync_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     int* task_tile_offsets_cpu,
                     int* task_metadata_gpu,
                     int* task_remaining_tiles_gpu,
                     uint64_t* local_ready_flags,
                     uint64_t* const* remote_ready_flag_ptrs_gpu,
                     uint64_t current_epoch,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms,
                     const int cta_multiplier = 2);

// Run weight sync using GPU-resident tasks (no H2D copy)
void run_weight_sync_from_gpu(WeightSyncTask* tasks_gpu,
                              int* task_tile_offsets_gpu,
                              int* task_metadata_gpu,
                              int* global_tile_counter_gpu,
                              int* task_remaining_tiles_gpu,
                              uint64_t* local_ready_flags,
                              uint64_t* const* remote_ready_flag_ptrs_gpu,
                              uint64_t current_epoch,
                              cudaStream_t stream,
                              int num_device_sms,
                              int max_possible_tiles,
                              int cta_multiplier = 2);

// ============================================================================
// GPU Task Build: Build weight_sync/grad_reduce tasks entirely on GPU
// ============================================================================

// Immutable config for GPU task build kernels (copied to GPU once in Manager ctor)
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

static __host__ __device__ __forceinline__ int ceil_div_int(const int a, const int b) {
    return (a + b - 1) / b;
}

static __host__ __device__ __forceinline__ int weight_sync_num_tiles(const size_t numel) {
    return static_cast<int>((numel + WEIGHT_SYNC_TILE_ELEMENTS - 1) / WEIGHT_SYNC_TILE_ELEMENTS);
}

static __host__ __device__ __forceinline__ size_t weight_sync_chunk_offset_elements(const int chunk_idx) {
    return static_cast<size_t>(chunk_idx) * WEIGHT_SYNC_RELAY_CHUNK_TILES * WEIGHT_SYNC_TILE_ELEMENTS;
}

static __host__ __device__ __forceinline__ size_t weight_sync_chunk_numel(const size_t total_numel,
                                                                          const int chunk_idx) {
    const size_t chunk_offset = weight_sync_chunk_offset_elements(chunk_idx);
    if (chunk_offset >= total_numel) {
        return 0;
    }

    const size_t chunk_capacity = static_cast<size_t>(WEIGHT_SYNC_RELAY_CHUNK_TILES) * WEIGHT_SYNC_TILE_ELEMENTS;
    const size_t remaining = total_numel - chunk_offset;
    return remaining < chunk_capacity ? remaining : chunk_capacity;
}

static __host__ __device__ __forceinline__ int floor_sqrt_int(const int x) {
    int root = 0;
    while ((root + 1) * (root + 1) <= x) {
        ++root;
    }
    return root;
}

static __host__ __device__ __forceinline__ int choose_weight_sync_relay_count(const int num_replicas,
                                                                              const TaskBuildConfig& config) {
    if (num_replicas <= 1) {
        return 0;
    }

    int relay_count = floor_sqrt_int(num_replicas);
    if (relay_count < 1) {
        relay_count = 1;
    }
    if (config.weight_sync_relay_max_relays > 0 && relay_count > config.weight_sync_relay_max_relays) {
        relay_count = config.weight_sync_relay_max_relays;
    }
    if (relay_count >= num_replicas) {
        relay_count = num_replicas - 1;
    }
    return relay_count;
}

static __host__ __device__ __forceinline__ int weight_sync_num_chunks(const size_t numel) {
    return ceil_div_int(weight_sync_num_tiles(numel), WEIGHT_SYNC_RELAY_CHUNK_TILES);
}

static __host__ __device__ __forceinline__ int max_weight_sync_relay_chunks_per_shard(const TaskBuildConfig& config) {
    const size_t max_numel = static_cast<size_t>(
        config.expert_fc1_numel > config.expert_fc2_numel ? config.expert_fc1_numel : config.expert_fc2_numel);
    return weight_sync_num_chunks(max_numel);
}

static __host__ __device__ __forceinline__ int weight_sync_ready_flag_slot(const TaskBuildConfig& config,
                                                                           const int local_replica_offset,
                                                                           const int shard_idx,
                                                                           const int chunk_idx) {
    return ((local_replica_offset * 2 + shard_idx) * max_weight_sync_relay_chunks_per_shard(config)) + chunk_idx;
}

static __host__ __device__ __forceinline__ int relay_stage_child_count(const int num_replicas,
                                                                       const int relay_count,
                                                                       const int relay_idx) {
    const int leaf_count = num_replicas - relay_count;
    if (leaf_count <= 0 || relay_idx < 0 || relay_idx >= relay_count || relay_idx >= leaf_count) {
        return 0;
    }
    return ceil_div_int(leaf_count - relay_idx, relay_count);
}

static __host__ __device__ __forceinline__ int relay_stage_leaf_owner(const int leaf_idx, const int relay_count) {
    return leaf_idx % relay_count;
}

static __host__ __device__ __forceinline__ bool should_use_weight_sync_relay(const int num_replicas,
                                                                             const TaskBuildConfig& config) {
    if (config.weight_sync_plan_mode == static_cast<int>(WeightSyncPlanMode::kDirect)) {
        return false;
    }

    const int relay_count = choose_weight_sync_relay_count(num_replicas, config);
    if (relay_count <= 0) {
        return false;
    }

    if (config.weight_sync_plan_mode == static_cast<int>(WeightSyncPlanMode::kForceRelay)) {
        return true;
    }

    if (num_replicas < config.weight_sync_relay_min_replicas) {
        return false;
    }

    const int relay_sender_fanout = relay_count;
    const int relay_child_fanout = ceil_div_int(num_replicas - relay_count, relay_count);
    const int relay_critical_fanout =
        relay_sender_fanout > relay_child_fanout ? relay_sender_fanout : relay_child_fanout;
    return (num_replicas - relay_critical_fanout) >= config.weight_sync_relay_min_fanout_gain;
}

// Build stage-1 and stage-2 weight sync task arrays on GPU from placement data.
// Stage 1 seeds relays (or directly fans out); stage 2 waits on per-relay ready
// flags and forwards chunked relay traffic to downstream replicas.
void build_weight_sync_task_lists(const TaskBuildConfig* config_gpu,
                                  const int32_t* p2l_gpu,
                                  const int32_t* l2p_gpu,
                                  const int32_t* lcnts_gpu,
                                  void* const* remote_weight_ptrs_gpu,
                                  const int64_t* local_master_fc1_ptrs_gpu,
                                  const int64_t* local_master_fc2_ptrs_gpu,
                                  __nv_bfloat16* local_replica_weight_buffer,
                                  WeightSyncTask* stage1_tasks_gpu,
                                  int* stage1_task_tile_offsets_gpu,
                                  int* stage1_task_metadata_gpu,
                                  int* stage1_task_remaining_tiles_gpu,
                                  int* stage1_global_tile_counter_gpu,
                                  WeightSyncTask* stage2_tasks_gpu,
                                  int* stage2_task_tile_offsets_gpu,
                                  int* stage2_task_metadata_gpu,
                                  int* stage2_global_tile_counter_gpu,
                                  cudaStream_t stream);

// Build grad reduce task array on GPU from placement data.
void build_grad_reduce_tasks(const TaskBuildConfig* config_gpu,
                             const int32_t* p2l_gpu,
                             const int32_t* l2p_gpu,
                             const int32_t* lcnts_gpu,
                             void* const* remote_grad_ptrs_gpu,
                             const int64_t* local_master_fc1_ptrs_gpu,
                             const int64_t* local_master_fc2_ptrs_gpu,
                             GradReduceTask* tasks_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             int* global_task_or_tile_counter_gpu,
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
                         cudaStream_t stream);

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
// atomicAdd on `counters_gpu`, then:
//   - non-quota mode: local_ordinal % replica_count
//   - quota mode:     upper_bound(rank_quota_prefix, local_ordinal)
//   topk_ids_ptr: [T, K] int64, device — modified in place
//   l2p_map_gpu: [L, max_replicas] int32, device — logical-to-physical map
//   lcnts_gpu: [L] int32, device — replica counts per logical expert
//   counters_gpu: [L] int32, device scratch — zeroed internally each call
void run_reroute_sparse(int64_t* topk_ids_ptr,
                        const int32_t* l2p_map_gpu,
                        const int32_t* lcnts_gpu,
                        int* counters_gpu,
                        const int num_tokens,
                        const int top_k,
                        const int num_global_logical_experts,
                        const int max_replicas,
                        cudaStream_t stream);

void run_reroute_sparse_quota(int64_t* topk_ids_ptr,
                              const int32_t* l2p_map_gpu,
                              const int32_t* lcnts_gpu,
                              const int32_t* rank_quota_prefix_gpu,
                              int* counters_gpu,
                              const int num_tokens,
                              const int top_k,
                              const int num_global_logical_experts,
                              const int max_replicas,
                              cudaStream_t stream);

}  // namespace ultra_ep::kernels
