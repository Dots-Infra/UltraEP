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
    // CPU tensor views (strided views into cpu_buffer)
    torch::Tensor physical_to_logical_map;
    torch::Tensor logical_to_physical_map;
    torch::Tensor logical_replica_counts;
    torch::Tensor logical_instance_quota;
    torch::Tensor logical_instance_quota_prefix;

    // GPU tensor views (strided views into gpu_buffer)
    torch::Tensor physical_to_logical_map_gpu;
    torch::Tensor logical_to_physical_map_gpu;
    torch::Tensor logical_replica_counts_gpu;
    torch::Tensor logical_instance_quota_gpu;
    torch::Tensor logical_instance_quota_prefix_gpu;
    // Per-rank quota prefix only needs a device copy.
    torch::Tensor rank_quota_prefix_gpu;

    // Contiguous per-layer buffer management.
    // Layout per layer:
    //   [p2l (P) | l2p (L*R) | lcnts (L) | quota (L*R) | quota_prefix (L*R)]
    // with padding to alignment.
    // Cross-layer stride is per_layer_stride_numel (may be > per_layer_data_numel for alignment).
    int32_t* cpu_buffer = nullptr;  // pinned host memory
    int32_t* gpu_buffer = nullptr;  // device memory
    int32_t* quota_buf_cpu = nullptr;
    int32_t* quota_buf_gpu = nullptr;

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

    // Copy placement data from CPU to GPU for one layer (layer_id >= 0) or all layers (layer_id == -1).
    // Uses the given CUDA stream (or current stream if not specified).
    void to_gpu(const int layer_id = -1,
                const bool async = true,
                std::optional<at::cuda::CUDAStream> stream = std::nullopt) const;

    // Copy placement data from GPU to CPU for one layer (layer_id >= 0) or all layers (layer_id == -1).
    // Used to refresh the host-visible mirror on demand for CPU fallbacks and debugging.
    void to_cpu(const int layer_id = -1,
                const bool async = true,
                std::optional<at::cuda::CUDAStream> stream = std::nullopt) const;

    // Raw pointer access for a specific layer.
    std::tuple<int32_t*, int32_t*, int32_t*> get_cpu_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_gpu_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_quota_cpu_ptrs(int layer_id) const;
    std::tuple<int32_t*, int32_t*, int32_t*> get_quota_gpu_ptrs(int layer_id) const;
};

// Pre-allocated CUDA output buffers for reroute (lazy init).
// NOTE: forward buffer is layer-independent in training to ensure correctness.
//       Backward buffer can be reused across layers (layers process sequentially).
class RerouteOutputBuffer {
    bool is_train_;
    int num_layers_;
    int num_global_logical_experts_;
    int num_global_physical_experts_;
    // FWD: [num_layers, T, P] (train), [T, P] (inference)
    torch::Tensor reroute_expand_probs_buf_;  // probs dtype
    // FWD: [num_layers, T, P] (train), [T, P] (inference)
    torch::Tensor reroute_expand_rmap_buf_;  // bool
    // BWD: [T, L] (only for train, shared across layers)
    torch::Tensor reroute_grad_probs_buf_;  // probs dtype
    // FWD scratch: [L, max_num_tiles] int32 — tile counts for two-pass forward
    torch::Tensor reroute_tile_counts_buf_;
    // Flags: track if each layer is zero-filled
    std::vector<bool> reroute_layer_valid_flags_;  // train
    bool reroute_inf_valid_flag_ = false;
    bool reroute_bwd_valid_flag_ = false;
    // Sizes:
    size_t reroute_expand_probs_nbytes_per_layer_ = 0;
    size_t reroute_expand_rmap_nbytes_per_layer_ = 0;

public:
    RerouteOutputBuffer(const int num_layers,
                        const int num_global_logical_experts,
                        const int num_global_physical_experts,
                        const bool is_train);
    std::tuple<void*, bool*> get_or_create_fwd_bufs(const int num_tokens,
                                                    const int layer_id,
                                                    const torch::ScalarType probs_dtype);
    void* get_or_create_bwd_buf(const int num_tokens, const torch::ScalarType probs_dtype);
    int32_t* get_or_create_tile_counts(const int L, const int num_tiles);
    void zero_out_fwd_bufs(const int layer_id, at::cuda::CUDAStream& stream);
    void zero_out_bwd_buf(at::cuda::CUDAStream& stream);
    bool get_fwd_valid_flag(const int layer_id) {
        return is_train_ ? reroute_layer_valid_flags_[layer_id] : reroute_inf_valid_flag_;
    };
    void set_fwd_valid_flag(const int layer_id, bool valid) {
        if (is_train_) {
            reroute_layer_valid_flags_[layer_id] = valid;
        } else {
            reroute_inf_valid_flag_ = valid;
        }
    };
    bool get_bwd_valid_flag() { return reroute_bwd_valid_flag_; };
    void set_bwd_valid_flag(bool valid) { reroute_bwd_valid_flag_ = valid; };

    // Retrieve forward's expanded_routing_map pointer for use in backward.
    // The buffer persists from forward to backward within the same training iteration.
    const bool* get_fwd_expanded_rmap_ptr(const int layer_id) const;
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
    int* _task_tile_offsets_cpu = nullptr;

    // Task metadata: [total_tasks, total_tiles] written by CPU or GPU task build path,
    // read by the main kernel from device memory.
    int* _task_metadata_gpu = nullptr;

    // GPU task build support: config + remote pointer tables + staging buffer
    kernels::TaskBuildConfig* _task_build_config_gpu = nullptr;
    void** _remote_weight_ptrs_gpu = nullptr;  // [MAX_NVL_DOMAIN_SIZE]
    void** _remote_grad_ptrs_gpu = nullptr;    // [MAX_NVL_DOMAIN_SIZE]
    int64_t* _local_master_ptrs_staging_gpu = nullptr;  // [2 * num_local_master_experts]
    // Pre-computed upper bounds for GPU-path grid sizing
    int _max_ws_total_tiles = 0;
    int _max_gr_total_tasks = 0;
    int _max_gr_total_tiles = 0;

    // Sparse reroute: per-expert round-robin counters [num_global_logical_experts]
    int* _reroute_sparse_counters_gpu = nullptr;

    // Early-stop balance threshold for replica allocation
    float balance_threshold_ = 1.0f;

    // Solvers
    std::unique_ptr<solver::PlacementSolver> placement_solver_;
    std::unique_ptr<solver::PlacementSolverGPU> placement_solver_gpu_;
    std::unique_ptr<solver::PlacementSolverQuota> placement_solver_quota_;
    std::unique_ptr<solver::RerouteSolver> reroute_solver_;
    bool use_gpu_solver_ = false;
    bool use_quota_solver_ = false;
    bool quota_locality_aware_ = true;
    int32_t quota_min_tokens_per_replica_ = 1;
    bool quota_allow_zero_master_quota_ = true;
    int quota_solver_version_ = 1;
    int quota_v1_oracle_mode_ = 0;
    float quota_v1_oracle_eps_ = 0.01f;
    int quota_v1_oracle_batch_k_ = 4;
    int quota_v1_kernel_stage_ = 0;
    std::vector<bool> placement_cpu_dirty_;
    // Shape: [num_global_logical_experts]
    int* global_logical_expert_loads_cpu = nullptr;  // pinned memory for CPU-side placement solver
    int* global_logical_expert_loads_gpu = nullptr;  // alloc by nvshmem for allreduce
    int32_t* local_expert_loads_gpu = nullptr;       // [L] — symmetric source buffer for allgather
    int32_t* expert_loads_per_rank_gpu = nullptr;    // [num_ranks, L] — symmetric allgather output

    // Reroute output buffer management
    std::unique_ptr<RerouteOutputBuffer> reroute_output_buffer_;

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
            const bool& use_gpu_solver = false,
            const float& balance_threshold = 1.0f,
            const bool& use_quota_solver = false,
            const bool& quota_locality_aware = true,
            const int32_t& quota_min_tokens_per_replica = 1,
            const bool& quota_allow_zero_master_quota = true,
            const int& quota_solver_version = 1,
            const int& quota_v1_oracle_mode = 0,
            const float& quota_v1_oracle_eps = 0.01f,
            const int& quota_v1_oracle_batch_k = 4,
            const int& quota_v1_kernel_stage = 0);
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
    // routing_map: [num_tokens, num_global_logical_experts], bool, logical routing map.
    // Runs on CPU or GPU depending on use_gpu_solver_. Deterministic across all ranks.
    void update_placement(const int& layer_id, torch::Tensor& routing_map);

    // Sparse variant: compute expert loads from topk_ids [T, K] int64 instead of dense routing_map.
    void update_placement_sparse(const int& layer_id, torch::Tensor& topk_ids);

    // In-place remap topk_ids from logical to physical expert IDs using current placement.
    // topk_ids: [T, K] int64, modified in-place on GPU.
    void reroute_sparse(const int& layer_id, torch::Tensor& topk_ids);

    // Expand logical routing map to physical routing map (CPU path).
    // routing_map: [num_tokens, num_logical_experts], bool, logical routing map.
    // Returns:
    // - token_indices: [num_tokens], int64, token indices in the expanded routing map.
    // - logical_indices: [num_tokens], int64, logical expert indices in the expanded routing map.
    // - physical_indices: [num_tokens], int64, physical expert indices in the expanded routing map.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reroute_cpu(const int& layer_id,
                                                                        torch::Tensor& routing_map);

    // CUDA reroute with pre-allocated output buffers (Manager owns the memory).
    // Buffers are lazily created on first use and reused across layers.
    // Each call returns fresh from_blob views (independent version counters).
    // Forward:  zeros + reroute kernel → (expanded_probs, expanded_routing_map)
    // Backward: zeros + backward kernel → grad_probs
    std::tuple<torch::Tensor, torch::Tensor> reroute_cuda_forward(const int& layer_id,
                                                                  torch::Tensor& probs,
                                                                  torch::Tensor& routing_map);

    torch::Tensor reroute_cuda_backward(const int& layer_id,
                                        torch::Tensor& grad_expanded_probs,
                                        torch::Tensor& routing_map);

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
    torch::Tensor get_rank_quota_prefix_tensor() const { return placement.rank_quota_prefix_gpu; }
    torch::Tensor get_global_logical_expert_loads_tensor() const {
        return make_tensor_from_buffer(global_logical_expert_loads_gpu,
                                       {num_global_logical_experts},
                                       torch::kInt32,
                                       torch::Device(torch::kCUDA, runtime::device_id));
    };
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Manager>(m, "Manager")
        .def(pybind11::init<int, int, int, int64_t, int64_t, bool, bool, bool, float, bool, bool, int32_t, bool, int, int, float, int, int>(),
             pybind11::arg("num_layers"),
             pybind11::arg("num_local_master_experts"),
             pybind11::arg("num_local_redundant_experts"),
             pybind11::arg("expert_fc1_numel"),
             pybind11::arg("expert_fc2_numel"),
             pybind11::arg("is_train"),
             pybind11::arg("explicitly_destroy"),
             pybind11::arg("use_gpu_solver") = false,
             pybind11::arg("balance_threshold") = 1.0f,
             pybind11::arg("use_quota_solver") = false,
             pybind11::arg("quota_locality_aware") = true,
             pybind11::arg("quota_min_tokens_per_replica") = 1,
             pybind11::arg("quota_allow_zero_master_quota") = true,
             pybind11::arg("quota_solver_version") = 1,
             pybind11::arg("quota_v1_oracle_mode") = 0,
             pybind11::arg("quota_v1_oracle_eps") = 0.01f,
             pybind11::arg("quota_v1_oracle_batch_k") = 4,
             pybind11::arg("quota_v1_kernel_stage") = 0)
        .def("destroy", &Manager::destroy)
        .def("is_available", &Manager::is_available)
        .def("sync_placement_to_cpu", &Manager::sync_placement_to_cpu, pybind11::arg("layer_id") = -1)
        .def("update_placement", &Manager::update_placement)
        .def("update_placement_sparse", &Manager::update_placement_sparse)
        .def("reroute_sparse", &Manager::reroute_sparse)
        .def("reroute_cpu", &Manager::reroute_cpu)
        .def("reroute_cuda_forward", &Manager::reroute_cuda_forward)
        .def("reroute_cuda_backward", &Manager::reroute_cuda_backward)
        .def("grad_reduce", &Manager::grad_reduce)
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
