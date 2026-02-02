#pragma once

namespace ultra_ep::kernels {

struct GradReduceTask {
    float* master_local_addr;
    float* replica_remote_addr;
    size_t numel;
};

void run_grad_reduce(const GradReduceTask* grad_reduce_tasks_cpu,
                     GradReduceTask* grad_reduce_tasks_gpu,
                     int* global_task_counter_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms);

}  // namespace ultra_ep::kernels