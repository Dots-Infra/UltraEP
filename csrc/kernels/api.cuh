#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
                             const int num_device_sms);

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
                                      int max_possible_tiles);

// ============================================================================
// Weight Sync: Broadcast master weights to replicas
// ============================================================================

// A broadcast task from one master to multiple replicas
// This structure enables loading SMEM once and TMA storing to multiple destinations
struct WeightSyncTask {
    __nv_bfloat16* master_local_addr;                              // Source: local master weight
    __nv_bfloat16* replica_remote_addrs[MAX_NVL_DOMAIN_SIZE - 1];  // Destinations: replica addresses
    int num_replicas;                                              // Number of replicas (1 to MAX_NVL_DOMAIN_SIZE-1)
    size_t numel;                                                  // Number of elements
};

// Run weight sync operation — CPU task-build path
// task_metadata_gpu: device buffer [2], written by this function as {total_tasks, total_tiles}
void run_weight_sync(const WeightSyncTask* weight_sync_tasks_cpu,
                     WeightSyncTask* weight_sync_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     int* task_tile_offsets_cpu,
                     int* task_metadata_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms);

// Run weight sync using GPU-resident tasks (no H2D copy)
void run_weight_sync_from_gpu(WeightSyncTask* tasks_gpu,
                              int* task_tile_offsets_gpu,
                              int* task_metadata_gpu,
                              int* global_tile_counter_gpu,
                              cudaStream_t stream,
                              int num_device_sms,
                              int max_possible_tiles);

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
};

// Build weight sync task array on GPU from placement data.
// Writes tasks, tile_offsets, and task_metadata ({total_tasks, total_tiles}).
// Also resets global_tile_counter to 0.
void build_weight_sync_tasks(const TaskBuildConfig* config_gpu,
                             const int32_t* p2l_gpu,
                             const int32_t* l2p_gpu,
                             const int32_t* lcnts_gpu,
                             void* const* remote_weight_ptrs_gpu,
                             const int64_t* local_master_fc1_ptrs_gpu,
                             const int64_t* local_master_fc2_ptrs_gpu,
                             WeightSyncTask* tasks_gpu,
                             int* task_tile_offsets_gpu,
                             int* task_metadata_gpu,
                             int* global_tile_counter_gpu,
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

// In-place remap topk_ids from logical to physical expert IDs using round-robin.
//   topk_ids_ptr: [T, K] int64, device — modified in place
//   l2p_map_gpu: [L, max_replicas] int32, device — logical-to-physical map
//   lcnts_gpu: [L] int32, device — replica counts per logical expert
//   counters_gpu: [L] int32, device scratch — round-robin counters (zeroed internally)
void run_reroute_sparse(int64_t* topk_ids_ptr,
                        const int32_t* l2p_map_gpu,
                        const int32_t* lcnts_gpu,
                        int* counters_gpu,
                        const int num_tokens,
                        const int top_k,
                        const int num_global_logical_experts,
                        const int max_replicas,
                        cudaStream_t stream);

}  // namespace ultra_ep::kernels