#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "kernels/api.cuh"
#include "kernels/config.cuh"
#include "runtime.hpp"
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
    // CPU tensor views (strided views into cpu_buffer)
    torch::Tensor physical_to_logical_map;
    torch::Tensor logical_to_physical_map;
    torch::Tensor logical_replica_counts;
    torch::Tensor logical_instance_quota;
    torch::Tensor logical_instance_quota_prefix;

    // Device tensor views (strided views into device_buffer)
    torch::Tensor physical_to_logical_map_device;
    torch::Tensor logical_to_physical_map_device;
    torch::Tensor logical_replica_counts_device;
    torch::Tensor logical_instance_quota_device;
    torch::Tensor logical_instance_quota_prefix_device;
    // Per-rank quota prefix only needs a device copy.
    torch::Tensor rank_quota_prefix;

    // Contiguous per-layer buffer management.
    // Layout per layer:
    //   [p2l (P) | l2p (L*R) | lcnts (L) | quota (L*R) | quota_prefix (L*R)]
    // with padding to alignment.
    // Cross-layer stride is per_layer_stride_numel (may be > per_layer_data_numel for alignment).
    int32_t* cpu_buffer = nullptr;  // pinned host memory
    int32_t* device_buffer = nullptr;
    int32_t* quota_buf_cpu = nullptr;
    int32_t* quota_buf_device = nullptr;

    int num_layers_ = 0;
    int p2l_numel = 0;               // = num_global_physical_experts
    int l2p_numel = 0;               // = num_global_logical_experts * max_replicas_dim
    int lcnts_numel = 0;             // = num_global_logical_experts
    int quota_numel = 0;             // = num_global_logical_experts * max_replicas_dim
    int quota_prefix_numel = 0;      // = num_global_logical_experts * max_replicas_dim
    int rank_quota_numel = 0;        // = num_global_logical_experts * max_replicas_dim
    int per_layer_data_numel = 0;    // = p2l + l2p + lcnts + quota + quota_prefix
    int per_layer_stride_numel = 0;  // aligned stride (in int32 elements)
    int per_layer_data_bytes = 0;
    int per_layer_stride_bytes = 0;
    int total_bytes = 0;

    // Alignment for DMA transfers (256 bytes works well for PCIe/NVLink)
    static constexpr int ALIGNMENT_BYTES = 256;

    // Initialize contiguous buffers and create tensor views.
    void init(int num_layers,
              int num_global_physical_experts,
              int num_global_logical_experts,
              int max_replicas_dim,
              int device_id);

    // Free contiguous buffers. Safe to call multiple times.
    void cleanup();

    // Copy placement data from CPU to device for one layer (layer_id >= 0) or all layers (layer_id == -1).
    // Uses the given CUDA stream (or current stream if not specified).
    void to_device(const int layer_id = -1,
                   const bool async = true,
                   std::optional<at::cuda::CUDAStream> stream = std::nullopt) const;

    // Copy placement data from device to CPU for one layer (layer_id >= 0) or all layers (layer_id == -1).
    // Used to refresh the host-visible mirror on demand for CPU fallbacks and debugging.
    void to_cpu(const int layer_id = -1,
                const bool async = true,
                std::optional<at::cuda::CUDAStream> stream = std::nullopt) const;

    // Raw pointer access for a specific layer.
    std::tuple<int32_t*, int32_t*, int32_t*> get_cpu_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_device_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_quota_cpu_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_quota_ptrs(int layer_id) const;
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
    bool is_train;

    // Placement buffers. CPU views are a host-visible mirror of the GPU state.
    GlobalExpertPlacement placement;

    // After NVSHMEM synchronization, this flag will be true
    bool _available = false;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;
    at::cuda::CUDAStream relay_stream;
    // Completion event for the latest placement update / forward-buffer pre-zero.
    // Train mode tracks one slot per layer; inference mode uses slot 0 as the shared buffer.
    std::vector<std::optional<EventHandle>> placement_ready_events_;
    std::vector<int64_t> placement_ready_stream_ids_;

    // Device-side local replica weight (bf16)/grad (fp32) buffers, shared by layers
    // Allocated via NVSHMEM symmetric heap for cross-GPU access
    // Shape (before flattened): [num_local_redundant_experts, expert_total_numel]
    void* local_replica_weight_buffer = nullptr;
    void* local_replica_grad_buffer = nullptr;
    torch::Tensor local_replica_weight_buffer_tensor;
    torch::Tensor local_replica_grad_buffer_tensor;

    // Host-side remote memory pointers obtained via nvshmem_ptr() for NVL ranks
    // Shape: [num_nvl_ranks,]
    void* global_replica_weight_buffer_ptrs[kernels::kMaxNvlDomainSize] = {nullptr};
    void* global_replica_grad_buffer_ptrs[kernels::kMaxNvlDomainSize] = {nullptr};
    uint64_t* global_weight_sync_ready_flag_ptrs[kernels::kMaxNvlDomainSize] = {nullptr};

    // Intermediate buffers for grad reduce tasks
    kernels::GradReduceTask* _grad_reduce_tasks = nullptr;
    int* _global_task_or_tile_counter = nullptr;
    int* _task_tile_offsets = nullptr;

    // Intermediate buffers for weight sync tasks
    kernels::WeightSyncTask* _weight_sync_tasks = nullptr;
    int _weight_sync_task_capacity = 0;
    int* _weight_sync_task_remaining_tiles = nullptr;
    kernels::WeightSyncTask* _relay_weight_sync_tasks = nullptr;
    int* _relay_task_tile_offsets = nullptr;
    int* _relay_task_metadata = nullptr;
    int* _relay_global_tile_counter = nullptr;
    uint64_t* local_weight_sync_ready_flags = nullptr;
    uint64_t _weight_sync_epoch = 0;

    // Task metadata: [total_tasks, total_tiles], written by task-build kernels and
    // consumed by the stage-1 persistent kernels.
    int* _task_metadata = nullptr;

    // Device task-build support: config + remote pointer tables.
    kernels::TaskBuildConfig* _task_build_config = nullptr;
    void** _remote_weight_ptrs = nullptr;         // [kMaxNvlDomainSize]
    void** _remote_grad_ptrs = nullptr;           // [kMaxNvlDomainSize]
    uint64_t** _remote_ready_flag_ptrs = nullptr; // [kMaxNvlDomainSize]
    // Pre-computed upper bounds for device-path grid sizing
    int _max_ws_total_tiles = 0;
    int _max_gr_total_tasks = 0;
    int _max_gr_total_tiles = 0;

    // Sparse reroute: per-expert round-robin counters [num_global_logical_experts]
    int* _reroute_sparse_counters = nullptr;

    // Early-stop balance threshold for replica allocation
    float balance_threshold_ = 1.0f;

    bool legacy_placement_ = false;
    bool quota_locality_aware_ = true;
    int32_t quota_min_tokens_per_replica_ = 1;
    bool quota_allow_zero_master_quota_ = true;
    float quota_oracle_eps_ = 0.01f;
    int quota_kernel_stage_ = 1;
    bool quota_reroute_interleave_ = true;
    int grad_reduce_num_sms_ = 24;
    int weight_sync_plan_mode_ = static_cast<int>(kernels::WeightSyncPlanMode::kAdaptive);
    int weight_sync_relay_min_replicas_ = 6;
    int weight_sync_relay_max_relays_ = 8;
    int weight_sync_relay_min_fanout_gain_ = 2;
    std::vector<bool> placement_cpu_dirty_;
    // Shape: [num_global_logical_experts]
    int* global_logical_expert_loads = nullptr;      // alloc by nvshmem for allreduce
    int* global_logical_expert_loads_cpu = nullptr;  // host scratch for legacy placement
    int32_t* local_expert_loads = nullptr;           // [L] — symmetric source buffer for allgather
    int32_t* expert_loads_per_rank = nullptr;        // [num_ranks, L] — symmetric allgather output

    int placement_sync_slot(const int layer_id) const { return is_train ? layer_id : 0; }
    void record_placement_ready(const int layer_id, const at::cuda::CUDAStream& stream);
    void wait_for_placement_ready(const int layer_id, const at::cuda::CUDAStream& stream) const;

public:
    Manager(const int& num_layers,
            const int& num_local_master_experts,
            const int& num_local_redundant_experts,
            const int64_t& expert_fc1_numel,
            const int64_t& expert_fc2_numel,
            const bool& is_train,
            const bool& explicitly_destroy,
            const bool& legacy_placement = false,
            const float& balance_threshold = 1.0f,
            const bool& quota_locality_aware = true,
            const int32_t& quota_min_tokens_per_replica = 1,
            const bool& quota_allow_zero_master_quota = true,
            const float& quota_oracle_eps = 0.01f,
            const int& quota_kernel_stage = 1,
            const bool& quota_reroute_interleave = true,
            const int& grad_reduce_num_sms = 24,
            const int& weight_sync_plan_mode = static_cast<int>(kernels::WeightSyncPlanMode::kAdaptive),
            const int& weight_sync_relay_min_replicas = 6,
            const int& weight_sync_relay_max_relays = 8,
            const int& weight_sync_relay_min_fanout_gain = 2);
    ~Manager() noexcept(false);
    void destroy();
    bool is_available() const { return _available; }
    void sync_placement_to_cpu(const int layer_id = -1);

    // Aggregate grad from remote replicas to local master
    // then zero-out replica grad buffers
    // Parameters (ptr tensor of local master grad buffers, for the current layer):
    // - local_master_fc1_grad_ptr_tensor: [num_local_master_experts]
    // - local_master_fc2_grad_ptr_tensor: [num_local_master_experts]
    std::optional<EventHandle> grad_reduce(const int& layer_id,
                                           torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                           torch::Tensor& local_master_fc2_grad_ptr_tensor,
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
    // routing_map: [num_tokens, num_global_logical_experts], bool, logical routing map.
    // Uses the default device placement path unless legacy placement is enabled.
    void update_placement(const int& layer_id, torch::Tensor& routing_map);

    // Sparse variant: compute expert loads from topk_ids [T, K] int64 instead of dense routing_map.
    void update_placement_sparse(const int& layer_id, torch::Tensor& topk_ids);

    // In-place remap topk_ids from logical to physical expert IDs using current placement.
    // topk_ids: [T, K] int64, modified in-place on GPU.
    void reroute_sparse(const int& layer_id, torch::Tensor& topk_ids);

    // Expand logical routing to physical routing in dense [T, L] form.
    std::tuple<torch::Tensor, torch::Tensor> dense_reroute_forward(const int& layer_id,
                                                                   torch::Tensor& probs,
                                                                   torch::Tensor& routing_map);

    torch::Tensor dense_reroute_backward(const int& layer_id,
                                         torch::Tensor& grad_expanded_probs,
                                         torch::Tensor& routing_map,
                                         torch::Tensor& expanded_routing_map);

    torch::Stream get_comm_stream() const { return comm_stream; }

    torch::Tensor get_local_replica_weight_buffer_tensor() const { return local_replica_weight_buffer_tensor; }
    torch::Tensor get_local_replica_grad_buffer_tensor() const {
        EP_HOST_ASSERT(is_train && "Grad buffer not available in inference mode");
        return local_replica_grad_buffer_tensor;
    }
    torch::Tensor get_physical_to_logical_map_tensor() const { return placement.physical_to_logical_map; }
    torch::Tensor get_logical_to_physical_map_tensor() const { return placement.logical_to_physical_map; }
    torch::Tensor get_logical_replica_counts_tensor() const { return placement.logical_replica_counts; }
    torch::Tensor get_logical_instance_quota_tensor() const { return placement.logical_instance_quota; }
    torch::Tensor get_logical_instance_quota_prefix_tensor() const { return placement.logical_instance_quota_prefix; }
    torch::Tensor get_rank_quota_prefix_tensor() const { return placement.rank_quota_prefix; }
    torch::Tensor get_global_logical_expert_loads_tensor() const {
        return make_tensor_from_buffer(global_logical_expert_loads,
                                       {num_global_logical_experts},
                                       torch::kInt32,
                                       torch::Device(torch::kCUDA, runtime::device_id));
    };
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Manager>(m, "Manager")
        .def(pybind11::init<int,
                            int,
                            int,
                            int64_t,
                            int64_t,
                            bool,
                            bool,
                            bool,
                            float,
                            bool,
                            int32_t,
                            bool,
                            float,
                            int,
                            bool,
                            int,
                            int,
                            int,
                            int,
                            int>(),
             pybind11::arg("num_layers"),
             pybind11::arg("num_local_master_experts"),
             pybind11::arg("num_local_redundant_experts"),
             pybind11::arg("expert_fc1_numel"),
             pybind11::arg("expert_fc2_numel"),
             pybind11::arg("is_train"),
             pybind11::arg("explicitly_destroy"),
             pybind11::arg("legacy_placement") = false,
             pybind11::arg("balance_threshold") = 1.0f,
             pybind11::arg("quota_locality_aware") = true,
             pybind11::arg("quota_min_tokens_per_replica") = 1,
             pybind11::arg("quota_allow_zero_master_quota") = true,
             pybind11::arg("quota_oracle_eps") = 0.01f,
             pybind11::arg("quota_kernel_stage") = 1,
             pybind11::arg("quota_reroute_interleave") = true,
             pybind11::arg("grad_reduce_num_sms") = 24,
             pybind11::arg("weight_sync_plan_mode") = static_cast<int>(kernels::WeightSyncPlanMode::kAdaptive),
             pybind11::arg("weight_sync_relay_min_replicas") = 6,
             pybind11::arg("weight_sync_relay_max_relays") = 8,
             pybind11::arg("weight_sync_relay_min_fanout_gain") = 2)
        .def("destroy", &Manager::destroy)
        .def("is_available", &Manager::is_available)
        .def("sync_placement_to_cpu", &Manager::sync_placement_to_cpu, pybind11::arg("layer_id") = -1)
        .def("update_placement", &Manager::update_placement)
        .def("update_placement_sparse", &Manager::update_placement_sparse)
        .def("reroute_sparse", &Manager::reroute_sparse)
        .def("dense_reroute_forward", &Manager::dense_reroute_forward)
        .def("dense_reroute_backward", &Manager::dense_reroute_backward)
        .def("grad_reduce",
             &Manager::grad_reduce,
             pybind11::arg("layer_id"),
             pybind11::arg("local_master_fc1_grad_ptr_tensor"),
             pybind11::arg("local_master_fc2_grad_ptr_tensor"),
             pybind11::arg("previous_event"),
             pybind11::arg("async_finish"))
        .def("weight_sync", &Manager::weight_sync)
        .def("get_comm_stream", &Manager::get_comm_stream)
        .def("get_local_replica_weight_buffer_tensor", &Manager::get_local_replica_weight_buffer_tensor)
        .def("get_local_replica_grad_buffer_tensor", &Manager::get_local_replica_grad_buffer_tensor)
        .def("get_physical_to_logical_map_tensor", &Manager::get_physical_to_logical_map_tensor)
        .def("get_logical_to_physical_map_tensor", &Manager::get_logical_to_physical_map_tensor)
        .def("get_logical_replica_counts_tensor", &Manager::get_logical_replica_counts_tensor)
        .def("get_logical_instance_quota_tensor", &Manager::get_logical_instance_quota_tensor)
        .def("get_logical_instance_quota_prefix_tensor", &Manager::get_logical_instance_quota_prefix_tensor)
        .def("get_rank_quota_prefix_tensor", &Manager::get_rank_quota_prefix_tensor)
        .def("get_global_logical_expert_loads_tensor", &Manager::get_global_logical_expert_loads_tensor);
}

}  // namespace ultra_ep
