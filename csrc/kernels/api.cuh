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

// Run gradient reduce operation across all tasks
// Parameters:
//   grad_reduce_tasks_cpu: Host-side array of tasks (will be copied to GPU)
//   grad_reduce_tasks_gpu: Device-side buffer for tasks
//   global_tile_counter_gpu: Device-side atomic counter for tile distribution
//   task_tile_offsets_gpu: Device-side buffer for prefix sum of tile counts (size: total_tasks + 1)
//   total_tasks: Number of tasks
//   stream: CUDA stream for async execution
//   num_device_sms: Number of SMs on the device
void run_grad_reduce_low_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                            GradReduceTask* grad_reduce_tasks_gpu,
                            int* global_task_counter_gpu,
                            const int total_tasks,
                            cudaStream_t stream,
                            const int num_device_sms);
void run_grad_reduce_high_sm(const GradReduceTask* grad_reduce_tasks_cpu,
                             GradReduceTask* grad_reduce_tasks_gpu,
                             int* global_tile_counter_gpu,
                             int* task_tile_offsets_gpu,
                             const int total_tasks,
                             cudaStream_t stream,
                             const int num_device_sms);

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

// Run weight sync operation (master broadcasts to all replicas)
// Parameters:
//   weight_sync_tasks_cpu: Host-side array of broadcast tasks
//   weight_sync_tasks_gpu: Device-side buffer for tasks
//   global_tile_counter_gpu: Device-side atomic counter for tile distribution
//   task_tile_offsets_gpu: Device-side buffer for prefix sum of tile counts
//   task_tile_offsets_cpu: Host-side buffer for prefix sum of tile counts
//   total_tasks: Number of broadcast tasks
//   stream: CUDA stream for async execution
//   num_device_sms: Number of SMs on the device
void run_weight_sync(const WeightSyncTask* weight_sync_tasks_cpu,
                     WeightSyncTask* weight_sync_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     int* task_tile_offsets_cpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms);

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