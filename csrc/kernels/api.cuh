#pragma once

#include <cuda_runtime.h>

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

}  // namespace ultra_ep::kernels