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
      comm_stream(at::cuda::getStreamFromPool(true)),
      mem_allocator()

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

    // Allocate local replica weight and grad buffers, and ptr buffers on GPU
    // then set local IPC handles
    int64_t local_replica_weight_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * WEIGHT_ELEMENT_SIZE;
    int64_t local_replica_grad_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * GRAD_ELEMENT_SIZE;
    int64_t global_replica_buffer_ptrs_bytes = (int64_t)runtime::num_nvl_ranks * num_local_redundant_experts * sizeof(void*);

    mem_allocator.malloc(&local_replica_weight_buffer, local_replica_weight_bytes);
    mem_allocator.malloc(&local_replica_grad_buffer, local_replica_grad_bytes);
    mem_allocator.get_handle(&weight_ipc_handles[runtime::nvl_rank_idx], local_replica_weight_buffer);
    mem_allocator.get_handle(&grad_ipc_handles[runtime::nvl_rank_idx], local_replica_grad_buffer);

    // Initialize local replica weight and grad buffer tensors
    local_replica_weight_buffer_tensor = make_tensor_from_buffer(local_replica_weight_buffer,
                                                                 {num_local_redundant_experts, expert_total_numel},
                                                                 torch::kBFloat16,
                                                                 torch::Device(torch::kCUDA, device_id));
    local_replica_grad_buffer_tensor = make_tensor_from_buffer(local_replica_grad_buffer,
                                                               {num_local_redundant_experts, expert_total_numel},
                                                               torch::kFloat32,
                                                               torch::Device(torch::kCUDA, device_id));
    mem_allocator.malloc_pinned((void**)&_grad_reduce_tasks_cpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask));
    mem_allocator.malloc((void**)&_grad_reduce_tasks_gpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask));
    mem_allocator.malloc((void**)&_global_task_counter_gpu, sizeof(int));
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

pybind11::bytes Manager::get_local_weight_ipc_handle() const {
    const ipc::MemHandle& handle = weight_ipc_handles[runtime::nvl_rank_idx];
    return pybind11::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

pybind11::bytes Manager::get_local_grad_ipc_handle() const {
    const ipc::MemHandle& handle = grad_ipc_handles[runtime::nvl_rank_idx];
    return pybind11::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

void Manager::sync_global_ipc_handles(const std::vector<std::optional<pybind11::bytes>>& all_gathered_weight_handles,
                                      const std::vector<std::optional<pybind11::bytes>>& all_gathered_grad_handles) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    int rdma_rank_idx = runtime::rdma_rank_idx;
    int num_nvl_ranks = runtime::num_nvl_ranks;
    int rank_idx = runtime::rank_idx;
    for (int i = 0, offset = rdma_rank_idx * num_nvl_ranks; i < num_nvl_ranks; ++i) {
        EP_HOST_ASSERT(all_gathered_weight_handles[offset + i].has_value());
        EP_HOST_ASSERT(all_gathered_grad_handles[offset + i].has_value());
        std::string weight_handle_str = all_gathered_weight_handles[offset + i].value();
        std::string grad_handle_str = all_gathered_grad_handles[offset + i].value();
        EP_HOST_ASSERT(weight_handle_str.size() == ipc::HANDLE_SIZE);
        EP_HOST_ASSERT(grad_handle_str.size() == ipc::HANDLE_SIZE);
        if (offset + i != rank_idx) {
            std::memcpy(&weight_ipc_handles[i], weight_handle_str.data(), ipc::HANDLE_SIZE);
            mem_allocator.open_handle(&global_replica_weight_buffer_ptrs[i], &weight_ipc_handles[i]);
            std::memcpy(&grad_ipc_handles[i], grad_handle_str.data(), ipc::HANDLE_SIZE);
            mem_allocator.open_handle(&global_replica_grad_buffer_ptrs[i], &grad_ipc_handles[i]);
        } else {
            EP_HOST_ASSERT(std::memcmp(&weight_ipc_handles[i], weight_handle_str.data(), ipc::HANDLE_SIZE) == 0);
            EP_HOST_ASSERT(std::memcmp(&grad_ipc_handles[i], grad_handle_str.data(), ipc::HANDLE_SIZE) == 0);
        }
    }

    nvshmem::barrier(true);
    // Ready to use
    _available = true;
}

void Manager::destroy() {
    EP_HOST_ASSERT(is_available());

    // Synchronize
    nvshmem::barrier(true);

    // Close remote IPC
    for (int i = 0; i < runtime::num_nvl_ranks; ++i) {
        if (i != runtime::nvl_rank_idx) {
            mem_allocator.close_handle(global_replica_weight_buffer_ptrs[i]);
            mem_allocator.close_handle(global_replica_grad_buffer_ptrs[i]);
        }
    }

    // Free local buffer and error flag
    mem_allocator.free(local_replica_weight_buffer);
    mem_allocator.free(local_replica_grad_buffer);
    mem_allocator.free_pinned(_grad_reduce_tasks_cpu);
    mem_allocator.free(_grad_reduce_tasks_gpu);
    mem_allocator.free(_global_task_counter_gpu);

    // Free NVSHMEM
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