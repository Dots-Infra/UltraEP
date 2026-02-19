#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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
//   total_tasks: Number of broadcast tasks
//   stream: CUDA stream for async execution
//   num_device_sms: Number of SMs on the device
void run_weight_sync(const WeightSyncTask* weight_sync_tasks_cpu,
                     WeightSyncTask* weight_sync_tasks_gpu,
                     int* global_tile_counter_gpu,
                     int* task_tile_offsets_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms);

// ============================================================================
// Reroute: Expand logical routing map to physical routing map (CUDA path)
// ============================================================================

// Forward: scatter probs from [T,L] logical to [T,P] physical space,
//          and construct expanded_routing_map [T,P] bool.
// Uses deterministic round-robin: for each expert l with C_l replicas,
// the k-th token (in ascending token order) maps to l2p[l, k % C_l].
//
// Parameters:
//   routing_map:           [T, L] bool, device
//   probs:                 [T, L] scalar_t (float/bf16), device
//   l2p_map:               [L, max_replicas] int32, device (this layer's slice)
//   lcnts:                 [L] int32, device (this layer's slice)
//   expanded_routing_map:  [T, P] bool, device, output (must be zero-initialized)
//   expanded_probs:        [T, P] scalar_t, device, output (must be zero-initialized)
//   T, L, P, max_replicas: dimensions
//   stream: CUDA stream
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
                         cudaStream_t stream);

// Backward: gather gradients from [T,P] physical back to [T,L] logical space.
// Recomputes the same round-robin mapping as forward, then:
//   grad_probs[t, l] = grad_expanded_probs[t, phys]
//
// Parameters:
//   grad_expanded_probs:  [T, P] scalar_t, device
//   routing_map:          [T, L] bool, device (saved from forward)
//   l2p_map:              [L, max_replicas] int32, device
//   lcnts:                [L] int32, device
//   grad_probs:           [T, L] scalar_t, device, output (must be zero-initialized)
//   T, L, P, max_replicas: dimensions
//   stream: CUDA stream
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
                          cudaStream_t stream);

}  // namespace ultra_ep::kernels