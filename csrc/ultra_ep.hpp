#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "config.hpp"
#include "kernels/api.cuh"
#include "runtime.hpp"
#include "utils/mem_alloc.hpp"
#include "utils/nvshmem.hpp"
#include "utils/utils.hpp"

namespace ultra_ep {

/* Describes the placement of global experts across all ranks.

Attributes:
    physical_to_logical_map: [num_global_physical_experts]
        mapping from physical to logical expert indices
    logical_to_physical_map: [num_global_logical_experts, max_replicas]
        mapping from logical to physical expert indices, padded with -1.
        The first entry is always the master, followed by replicas.
    logical_replica_counts: [num_global_logical_experts]
        number of replicas for each logical expert (includes master)

Example:
    Suppose EP2 (2 GPUs) and 4 logical experts 0~3, each EP rank has 1 redundant expert
    - Master assignment: rank0 masters [2, 1], rank1 masters [0, 3]
    - num_local_physical_experts = 2 + 1 = 3 per rank
    - Physical layout:
        * Rank 0: physical [0,1,2] = [master(2), master(1), redundant]
        * Rank 1: physical [3,4,5] = [master(0), master(3), redundant]
    - Suppose rank0 replicates expert 3, rank1 replicates expert 1
    - physical_to_logical_map: [2, 1, 3, 0, 3, 1]
        (physical 0→2, 1→1, 2→3, 3→0, 4→3, 5→1)
    - logical_to_physical_map: [[3, -1], [1, 5], [0, -1], [4, 2]]
        (expert 0: master at phys 3; expert 1: master at phys 1, replica at phys 5;
        expert 2: master at phys 0; expert 3: master at phys 4, replica at phys 2)
    - logical_replica_counts: [1, 2, 1, 2]
*/
struct GlobalExpertPlacement {
    torch::Tensor physical_to_logical_map;
    torch::Tensor logical_to_physical_map;
    torch::Tensor logical_replica_counts;
    int32_t* p2l_ptr;
    int32_t* l2p_ptr;
    int32_t* lcnts_ptr;
};

class Manager {
    // Model and expert settings
    int num_local_master_experts;
    int num_local_redundant_experts;
    int num_local_physical_experts;
    int64_t expert_fc1_numel, expert_fc2_numel;
    int64_t expert_total_numel;
    int num_global_physical_experts;
    int num_global_logical_experts;

    // Placement (on CPU)
    GlobalExpertPlacement placement;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool _available = false;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // CUDA stream for communication
    at::cuda::CUDAStream comm_stream;

    // Device-side local replica weight (bf16)/grad (fp32) buffers, shared by layers
    // Shape (before flattened): [num_local_redundant_experts, expert_total_numel]
    void* local_replica_weight_buffer = nullptr;
    void* local_replica_grad_buffer = nullptr;
    torch::Tensor local_replica_weight_buffer_tensor;
    torch::Tensor local_replica_grad_buffer_tensor;

    // Host-side remote memory pointers with cudaIPC access of replica buffers of NVL ranks
    // Shape: [num_nvl_ranks,]
    void* global_replica_weight_buffer_ptrs[MAX_NVL_DOMAIN_SIZE] = {nullptr};
    void* global_replica_grad_buffer_ptrs[MAX_NVL_DOMAIN_SIZE] = {nullptr};

    // Host-side IPC manager for intra-NVL domain communication
    ipc::RemoteMemAllocator mem_allocator;
    ipc::MemHandle weight_ipc_handles[MAX_NVL_DOMAIN_SIZE];
    ipc::MemHandle grad_ipc_handles[MAX_NVL_DOMAIN_SIZE];

    // Intermediate buffers for grad reduce tasks
    kernels::GradReduceTask* _grad_reduce_tasks_cpu = nullptr;
    kernels::GradReduceTask* _grad_reduce_tasks_gpu = nullptr;
    int* _global_task_counter_gpu = nullptr;

public:
    Manager(const int& num_local_master_experts,
            const int& num_local_redundant_experts,
            const int64_t& expert_fc1_numel,
            const int64_t& expert_fc2_numel,
            const bool& explicitly_destroy);
    ~Manager() noexcept(false);
    void destroy();
    bool is_available() const { return _available; }

    // Aggregate grad from remote replicas to local master
    // then zero-out replica grad buffers
    // Parameters (ptr tensor of local master grad buffers, for the current layer):
    // - local_master_fc1_grad_ptr_tensor: [num_local_master_experts]
    // - local_master_fc2_grad_ptr_tensor: [num_local_master_experts]
    void grad_reduce(torch::Tensor local_master_fc1_grad_ptr_tensor, torch::Tensor local_master_fc2_grad_ptr_tensor);

    pybind11::bytes get_local_weight_ipc_handle() const;
    pybind11::bytes get_local_grad_ipc_handle() const;
    void sync_global_ipc_handles(const std::vector<std::optional<pybind11::bytes>>& all_gathered_weight_handles,
                                 const std::vector<std::optional<pybind11::bytes>>& all_gathered_grad_handles);
    torch::Tensor get_local_replica_weight_buffer_tensor() const { return local_replica_weight_buffer_tensor; }
    torch::Tensor get_local_replica_grad_buffer_tensor() const { return local_replica_grad_buffer_tensor; }
    torch::Tensor get_physical_to_logical_map_tensor() const { return placement.physical_to_logical_map; }
    torch::Tensor get_logical_to_physical_map_tensor() const { return placement.logical_to_physical_map; }
    torch::Tensor get_logical_replica_counts_tensor() const { return placement.logical_replica_counts; }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Manager>(m, "Manager")
        .def(pybind11::init<int, int, int64_t, int64_t, bool>())
        .def("destroy", &Manager::destroy)
        .def("is_available", &Manager::is_available)
        .def("grad_reduce", &Manager::grad_reduce)
        .def("get_local_weight_ipc_handle", &Manager::get_local_weight_ipc_handle)
        .def("get_local_grad_ipc_handle", &Manager::get_local_grad_ipc_handle)
        .def("sync_global_ipc_handles", &Manager::sync_global_ipc_handles)
        .def("get_local_replica_weight_buffer_tensor", &Manager::get_local_replica_weight_buffer_tensor)
        .def("get_local_replica_grad_buffer_tensor", &Manager::get_local_replica_grad_buffer_tensor)
        .def("get_physical_to_logical_map_tensor", &Manager::get_physical_to_logical_map_tensor)
        .def("get_logical_to_physical_map_tensor", &Manager::get_logical_to_physical_map_tensor)
        .def("get_logical_replica_counts_tensor", &Manager::get_logical_replica_counts_tensor);
}

}  // namespace ultra_ep