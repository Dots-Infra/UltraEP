#include "ultra_ep.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace ultra_ep {

Manager::Manager(const int& num_local_master_experts,
                 const int& num_local_redundant_experts,
                 const int64_t& expert_fc1_numel,
                 const int64_t& expert_fc2_numel,
                 const bool& explicitly_destroy)
    : num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true))

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    num_global_physical_experts = num_local_physical_experts * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;

    // Allocate global placement tensors on CPU
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.physical_to_logical_map =
        torch::full({num_global_physical_experts}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    placement.logical_to_physical_map = torch::full(
        {num_global_logical_experts, num_ranks}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    placement.logical_replica_counts =
        torch::zeros({num_global_logical_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    placement.p2l_ptr = placement.physical_to_logical_map.data<int32_t>();
    placement.l2p_ptr = placement.logical_to_physical_map.data<int32_t>();
    placement.lcnts_ptr = placement.logical_replica_counts.data<int32_t>();

    // Allocate local replica weight and grad buffers via NVSHMEM symmetric heap
    // This enables automatic cross-GPU access within NVL domain
    int64_t local_replica_weight_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * WEIGHT_ELEMENT_SIZE;
    int64_t local_replica_grad_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * GRAD_ELEMENT_SIZE;

    // Allocate via NVSHMEM for symmetric heap (accessible from all PEs)
    local_replica_weight_buffer = nvshmem::alloc(local_replica_weight_bytes, NVSHMEM_ALIGNMENT);
    local_replica_grad_buffer = nvshmem::alloc(local_replica_grad_bytes, NVSHMEM_ALIGNMENT);
    EP_HOST_ASSERT(local_replica_weight_buffer != nullptr && "Failed to allocate NVSHMEM weight buffer");
    EP_HOST_ASSERT(local_replica_grad_buffer != nullptr && "Failed to allocate NVSHMEM grad buffer");

    // Initialize local replica weight and grad buffer tensors
    local_replica_weight_buffer_tensor = make_tensor_from_buffer(local_replica_weight_buffer,
                                                                 {num_local_redundant_experts, expert_total_numel},
                                                                 torch::kBFloat16,
                                                                 torch::Device(torch::kCUDA, device_id));
    local_replica_grad_buffer_tensor = make_tensor_from_buffer(local_replica_grad_buffer,
                                                               {num_local_redundant_experts, expert_total_numel},
                                                               torch::kFloat32,
                                                               torch::Device(torch::kCUDA, device_id));

    // Synchronize all PEs to ensure buffers are allocated on all ranks
    nvshmem::barrier(true);

    // Obtain remote pointers via nvshmem_ptr() for all NVL ranks
    // nvshmem_ptr() returns a pointer that can be used to directly access
    // the symmetric memory on the specified PE from the local PE
    int num_nvl_ranks = runtime::num_nvl_ranks;
    int rdma_rank_idx = runtime::rdma_rank_idx;
    for (int i = 0; i < num_nvl_ranks; ++i) {
        int target_rank = rdma_rank_idx * num_nvl_ranks + i;
        global_replica_weight_buffer_ptrs[i] = nvshmem::ptr(local_replica_weight_buffer, target_rank);
        global_replica_grad_buffer_ptrs[i] = nvshmem::ptr(local_replica_grad_buffer, target_rank);
        EP_HOST_ASSERT(global_replica_weight_buffer_ptrs[i] != nullptr &&
                       "nvshmem_ptr failed for weight buffer - target PE may not be in same NVL domain");
        EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[i] != nullptr &&
                       "nvshmem_ptr failed for grad buffer - target PE may not be in same NVL domain");
    }

    // Allocate intermediate buffers for grad reduce tasks (regular CUDA memory)
    CUDA_RUNTIME_CHECK(cudaMallocHost((void**)&_grad_reduce_tasks_cpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_grad_reduce_tasks_gpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_global_task_counter_gpu, sizeof(int)));

    // Ready to use (no IPC handle exchange needed with NVSHMEM)
    _available = true;
}

Manager::~Manager() noexcept(false) {
    if (!explicitly_destroy) {
        if (_available) {
            destroy();
        }
    } else if (_available) {
        printf("WARNING: destroy() was not called before UltraEP manager destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

void Manager::destroy() {
    EP_HOST_ASSERT(is_available());

    // Synchronize all PEs before cleanup
    nvshmem::barrier(true);

    // Free NVSHMEM symmetric heap buffers
    // No need to close remote handles - nvshmem_ptr pointers are automatically invalidated
    nvshmem::free(local_replica_weight_buffer);
    nvshmem::free(local_replica_grad_buffer);
    local_replica_weight_buffer = nullptr;
    local_replica_grad_buffer = nullptr;

    // Clear remote pointers
    for (int i = 0; i < runtime::num_nvl_ranks; ++i) {
        global_replica_weight_buffer_ptrs[i] = nullptr;
        global_replica_grad_buffer_ptrs[i] = nullptr;
    }

    // Free intermediate CUDA buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(_grad_reduce_tasks_cpu));
    CUDA_RUNTIME_CHECK(cudaFree(_grad_reduce_tasks_gpu));
    CUDA_RUNTIME_CHECK(cudaFree(_global_task_counter_gpu));
    _grad_reduce_tasks_cpu = nullptr;
    _grad_reduce_tasks_gpu = nullptr;
    _global_task_counter_gpu = nullptr;

    // Free NVSHMEM runtime
    runtime::destroy();

    // Ready to destroy
    _available = false;
}

void Manager::grad_reduce(torch::Tensor local_master_fc1_grad_ptr_tensor, torch::Tensor local_master_fc2_grad_ptr_tensor) {
    EP_HOST_ASSERT(is_available());

    void** local_master_fc1_grad_ptrs = reinterpret_cast<void**>(local_master_fc1_grad_ptr_tensor.data<int64_t>());
    void** local_master_fc2_grad_ptrs = reinterpret_cast<void**>(local_master_fc2_grad_ptr_tensor.data<int64_t>());

    // Flatten task list (host-side)
    int num_tasks = 0;
    for (int i = 0; i < num_local_master_experts; ++i) {
        int master_global_phy_idx = runtime::rank_idx * num_local_physical_experts + i;
        int master_global_log_idx = placement.p2l_ptr[master_global_phy_idx];
        int num_replicas = placement.lcnts_ptr[master_global_log_idx];
        float* local_master_fc1_grad_ptr = reinterpret_cast<float*>(local_master_fc1_grad_ptrs[i]);
        float* local_master_fc2_grad_ptr = reinterpret_cast<float*>(local_master_fc2_grad_ptrs[i]);

        for (int j = 1; j < num_replicas; ++j) {  // skip the master itself
            int replica_global_phy_idx = placement.l2p_ptr[master_global_log_idx * runtime::num_ranks + j];
            int replica_global_rank_idx = replica_global_phy_idx / num_local_physical_experts;
            EP_HOST_ASSERT(is_in_same_nvl_domain(runtime::rank_idx, replica_global_rank_idx, runtime::num_nvl_ranks) &&
                           "Replica rank is not in the same NVL domain as the master rank");
            int replica_nvl_rank_idx = replica_global_rank_idx % runtime::num_nvl_ranks;
            EP_HOST_ASSERT(replica_nvl_rank_idx != runtime::nvl_rank_idx &&
                           "Replica rank is the same as the master rank, which is not allowed");
            EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[replica_nvl_rank_idx] != nullptr);
            int replica_local_offset = replica_global_phy_idx % num_local_physical_experts - num_local_master_experts;
            EP_HOST_ASSERT(replica_local_offset >= 0 and replica_local_offset < num_local_redundant_experts);
            float* replica_remote_grad_buffer_ptr = reinterpret_cast<float*>(global_replica_grad_buffer_ptrs[replica_nvl_rank_idx]);
            float* replica_remote_fc1_grad_ptr = replica_remote_grad_buffer_ptr + replica_local_offset * expert_total_numel;
            float* replica_remote_fc2_grad_ptr = replica_remote_fc1_grad_ptr + expert_fc1_numel;
            _grad_reduce_tasks_cpu[num_tasks++] = {
                local_master_fc1_grad_ptr, replica_remote_fc1_grad_ptr, static_cast<size_t>(expert_fc1_numel)};
            _grad_reduce_tasks_cpu[num_tasks++] = {
                local_master_fc2_grad_ptr, replica_remote_fc2_grad_ptr, static_cast<size_t>(expert_fc2_numel)};
        }
    }
    if (num_tasks == 0) {
        return;
    }

    // Call device-side kernels
    kernels::run_grad_reduce(
        _grad_reduce_tasks_cpu, _grad_reduce_tasks_gpu, _global_task_counter_gpu, num_tasks, comm_stream, runtime::num_device_sms);
}

}  // namespace ultra_ep