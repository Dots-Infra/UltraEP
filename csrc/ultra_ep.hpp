#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "config.hpp"
#include "kernels/api.cuh"
#include "runtime.hpp"
#include "solvers/api.hpp"
#include "utils/event.hpp"
#include "utils/exception.cuh"
#include "utils/nvshmem.cuh"
#include "utils/utils.hpp"

namespace ultra_ep {

/* Describes the placement of global experts across all ranks.

Attributes:
    physical_to_logical_map: [num_layers, num_global_physical_experts]
        mapping from physical to logical expert indices
    logical_to_physical_map: [num_layers, num_global_logical_experts, max_replicas]
        mapping from logical to physical expert indices, padded with -1.
        The first entry is always the master, followed by replicas.
    logical_replica_counts: [num_layers, num_global_logical_experts]
        number of replicas for each logical expert (includes master)

Example:
    Suppose EP2 (2 GPUs) and 4 logical experts 0~3, each EP rank has 1 redundant expert
    Omit the first index for layer for simplicity
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
};

class Manager {
    // Model and expert settings
    int num_layers;
    int num_local_master_experts;
    int num_local_redundant_experts;
    int num_local_physical_experts;
    int64_t expert_fc1_numel, expert_fc2_numel;
    int64_t expert_total_numel;
    int num_global_physical_experts;
    int num_global_logical_experts;

    // Placement (on CPU)
    GlobalExpertPlacement placement;

    // After NVSHMEM synchronization, this flag will be true
    bool _available = false;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // CUDA stream for communication
    at::cuda::CUDAStream comm_stream;

    // Device-side local replica weight (bf16)/grad (fp32) buffers, shared by layers
    // Allocated via NVSHMEM symmetric heap for cross-GPU access
    // Shape (before flattened): [num_local_redundant_experts, expert_total_numel]
    void* local_replica_weight_buffer = nullptr;
    void* local_replica_grad_buffer = nullptr;
    torch::Tensor local_replica_weight_buffer_tensor;
    torch::Tensor local_replica_grad_buffer_tensor;

    // Host-side remote memory pointers obtained via nvshmem_ptr() for NVL ranks
    // Shape: [num_nvl_ranks,]
    void* global_replica_weight_buffer_ptrs[MAX_NVL_DOMAIN_SIZE] = {nullptr};
    void* global_replica_grad_buffer_ptrs[MAX_NVL_DOMAIN_SIZE] = {nullptr};

    // Intermediate buffers for grad reduce tasks
    kernels::GradReduceTask* _grad_reduce_tasks_cpu = nullptr;
    kernels::GradReduceTask* _grad_reduce_tasks_gpu = nullptr;
    int* _global_task_or_tile_counter_gpu = nullptr;
    int* _task_tile_offsets_gpu = nullptr;

    // Intermediate buffers for weight sync tasks
    kernels::WeightSyncTask* _weight_sync_tasks_cpu = nullptr;
    kernels::WeightSyncTask* _weight_sync_tasks_gpu = nullptr;
    // Reuse _global_task_or_tile_counter_gpu and _task_tile_offsets_gpu for weight sync

    // Pre-allocated placement solver (zero-alloc on hot path)
    std::unique_ptr<solver::PlacementSolver> placement_solver_;

public:
    Manager(const int& num_layers,
            const int& num_local_master_experts,
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
    std::optional<EventHandle> grad_reduce(const int& layer_id,
                                           torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                           torch::Tensor& local_master_fc2_grad_ptr_tensor,
                                           std::string& mode,
                                           std::optional<EventHandle>& previous_event,
                                           bool async);
    // Sync replica weights with masters
    // Parameters (ptr tensor of local master weight buffers, for the current layer):
    // - local_master_fc1_weight_ptr_tensor: [num_local_master_experts]
    // - local_master_fc2_weight_ptr_tensor: [num_local_master_experts]
    std::optional<EventHandle> weight_sync(const int& layer_id,
                                           torch::Tensor& local_master_fc1_weight_ptr_tensor,
                                           torch::Tensor& local_master_fc2_weight_ptr_tensor,
                                           std::optional<EventHandle>& previous_event,
                                           bool async);

    // Update expert placement for a single layer based on real-time load statistics.
    // expert_loads: [num_global_logical_experts], int32 token counts per logical expert.
    // Runs entirely on CPU. Deterministic across all ranks.
    void update_placement(const int& layer_id, torch::Tensor& expert_loads);

    torch::Stream get_comm_stream() const { return comm_stream; }

    torch::Tensor get_local_replica_weight_buffer_tensor() const { return local_replica_weight_buffer_tensor; }
    torch::Tensor get_local_replica_grad_buffer_tensor() const { return local_replica_grad_buffer_tensor; }
    torch::Tensor get_physical_to_logical_map_tensor() const { return placement.physical_to_logical_map; }
    torch::Tensor get_logical_to_physical_map_tensor() const { return placement.logical_to_physical_map; }
    torch::Tensor get_logical_replica_counts_tensor() const { return placement.logical_replica_counts; }

private:
    std::tuple<int32_t*, int32_t*, int32_t*> get_placement_map_ptrs(const int& layer_id) const;
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Manager>(m, "Manager")
        .def(pybind11::init<int, int, int, int64_t, int64_t, bool>())
        .def("destroy", &Manager::destroy)
        .def("is_available", &Manager::is_available)
        .def("update_placement", &Manager::update_placement)
        .def("grad_reduce", &Manager::grad_reduce)
        .def("weight_sync", &Manager::weight_sync)
        .def("get_comm_stream", &Manager::get_comm_stream)
        .def("get_local_replica_weight_buffer_tensor", &Manager::get_local_replica_weight_buffer_tensor)
        .def("get_local_replica_grad_buffer_tensor", &Manager::get_local_replica_grad_buffer_tensor)
        .def("get_physical_to_logical_map_tensor", &Manager::get_physical_to_logical_map_tensor)
        .def("get_logical_to_physical_map_tensor", &Manager::get_logical_to_physical_map_tensor)
        .def("get_logical_replica_counts_tensor", &Manager::get_logical_replica_counts_tensor);
}

}  // namespace ultra_ep