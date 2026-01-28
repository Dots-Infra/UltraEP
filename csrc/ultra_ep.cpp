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
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      mem_allocator()

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    num_global_physical_experts = (num_local_master_experts + num_local_redundant_experts) * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;

    // Allocate global placement tensors
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.physical_to_logical_map =
        torch::full({num_global_physical_experts}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id));
    placement.logical_to_physical_map = torch::full(
        {num_global_logical_experts, num_ranks}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id));
    placement.logical_replica_counts =
        torch::zeros({num_global_logical_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id));

    // Allocate local replica weight and grad buffers, and ptr buffers on GPU
    // then set local IPC handles
    int64_t local_replica_weight_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * WEIGHT_ELEMENT_SIZE;
    int64_t local_replica_grad_bytes = (int64_t)num_local_redundant_experts * expert_total_numel * GRAD_ELEMENT_SIZE;
    int64_t global_replica_buffer_ptrs_bytes = (int64_t)runtime::num_nvl_ranks * num_local_redundant_experts * sizeof(void*);

    mem_allocator.malloc(&local_replica_weight_buffer, local_replica_weight_bytes);
    mem_allocator.malloc(&local_replica_grad_buffer, local_replica_grad_bytes);
    global_replica_weight_buffer_ptrs = (void**)std::malloc(global_replica_buffer_ptrs_bytes);
    global_replica_grad_buffer_ptrs = (void**)std::malloc(global_replica_buffer_ptrs_bytes);
    mem_allocator.get_handle(&weight_ipc_handles[runtime::nvl_rank_idx], local_replica_weight_buffer);
    mem_allocator.get_handle(&grad_ipc_handles[runtime::nvl_rank_idx], local_replica_grad_buffer);
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
    std::free(global_replica_weight_buffer_ptrs);
    std::free(global_replica_grad_buffer_ptrs);

    // Free NVSHMEM
    runtime::destroy();

    // Ready to destroy
    _available = false;
}

}  // namespace ultra_ep