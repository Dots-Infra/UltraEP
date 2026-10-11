#include "ultra_ep.hpp"

#include <algorithm>
#include <cstdio>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

namespace ultra_ep {

namespace {

int64_t checked_num_bytes(const int64_t numel, const int elem_bytes) {
    EP_HOST_ASSERT(numel >= 0);
    EP_HOST_ASSERT(elem_bytes > 0);
    return numel * static_cast<int64_t>(elem_bytes);
}

int64_t align_up_bytes(const int64_t value, const int64_t alignment) {
    EP_HOST_ASSERT(value >= 0 && alignment > 0);
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T>
void copy_remote_pointer_table(const torch::Tensor& table, T**& device_table) {
    EP_HOST_ASSERT(table.defined() && !table.is_cuda() && table.dtype() == torch::kInt64 &&
                   table.is_contiguous() && table.dim() == 1 && table.numel() == runtime::num_nvl_ranks);
    T* host_table[kernels::kMaxNvlDomainSize] = {};
    const auto* values = table.data_ptr<int64_t>();
    for (int rank = 0; rank < runtime::num_nvl_ranks; ++rank) {
        EP_HOST_ASSERT(values[rank] != 0 && "Symmetric Memory returned a null peer pointer");
        host_table[rank] = reinterpret_cast<T*>(static_cast<uintptr_t>(values[rank]));
    }
    CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_table), sizeof(host_table)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(device_table, host_table, sizeof(host_table), cudaMemcpyHostToDevice));
}

void fill_task_build_config(kernels::TaskBuildConfig& config,
                            int num_local_master_experts,
                            int num_local_physical_experts,
                            int num_local_redundant_experts,
                            int64_t expert_fc1_numel,
                            int64_t expert_fc2_numel,
                            int64_t expert_total_numel,
                            int64_t expert_fc1_weight_scale_numel,
                            int64_t expert_fc2_weight_scale_numel,
                            int64_t expert_weight_scale_total_numel,
                            int64_t expert_fc1_weight_scale_bytes,
                            int64_t expert_fc2_weight_scale_bytes,
                            int64_t expert_weight_scale_fc2_offset_bytes,
                            int64_t expert_weight_scale_stride_bytes,
                            int weight_data_element_bytes,
                            int weight_scale_element_bytes,
                            int weight_sync_plan_mode,
                            int weight_sync_relay_min_replicas,
                            int weight_sync_relay_max_relays,
                            int weight_sync_relay_min_fanout_gain) {
    config = {};
    config.rank_idx = runtime::rank_idx;
    config.nvl_rank_idx = runtime::nvl_rank_idx;
    config.num_nvl_ranks = runtime::num_nvl_ranks;
    config.num_local_master_experts = num_local_master_experts;
    config.num_local_physical_experts = num_local_physical_experts;
    config.num_local_redundant_experts = num_local_redundant_experts;
    config.expert_fc1_numel = expert_fc1_numel;
    config.expert_fc2_numel = expert_fc2_numel;
    config.expert_total_numel = expert_total_numel;
    config.expert_fc1_weight_scale_numel = expert_fc1_weight_scale_numel;
    config.expert_fc2_weight_scale_numel = expert_fc2_weight_scale_numel;
    config.expert_weight_scale_total_numel = expert_weight_scale_total_numel;
    config.expert_fc1_weight_scale_bytes = expert_fc1_weight_scale_bytes;
    config.expert_fc2_weight_scale_bytes = expert_fc2_weight_scale_bytes;
    config.expert_weight_scale_fc2_offset_bytes = expert_weight_scale_fc2_offset_bytes;
    config.expert_weight_scale_stride_bytes = expert_weight_scale_stride_bytes;
    config.weight_data_element_bytes = weight_data_element_bytes;
    config.weight_scale_element_bytes = weight_scale_element_bytes;
    config.max_replicas_dim = runtime::num_ranks;
    config.weight_sync_plan_mode = weight_sync_plan_mode;
    config.weight_sync_relay_min_replicas = weight_sync_relay_min_replicas;
    config.weight_sync_relay_max_relays = weight_sync_relay_max_relays;
    config.weight_sync_relay_min_fanout_gain = weight_sync_relay_min_fanout_gain;
}

int weight_sync_num_shards(const int64_t expert_weight_scale_total_numel) {
    return expert_weight_scale_total_numel > 0 ? 4 : 2;
}

int weight_sync_chunks_for_shards(int64_t expert_fc1_numel,
                                  int64_t expert_fc2_numel,
                                  int64_t expert_fc1_weight_scale_bytes,
                                  int64_t expert_fc2_weight_scale_bytes,
                                  int weight_data_element_bytes) {
    return kernels::weight_sync_num_chunks(
               static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_chunks(
               static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc1_weight_scale_bytes)) +
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc2_weight_scale_bytes));
}

int weight_sync_tiles_for_shards(int64_t expert_fc1_numel,
                                 int64_t expert_fc2_numel,
                                 int64_t expert_fc1_weight_scale_bytes,
                                 int64_t expert_fc2_weight_scale_bytes,
                                 int weight_data_element_bytes) {
    return kernels::weight_sync_num_tiles(
               static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_tiles(
               static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc1_weight_scale_bytes)) +
        kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc2_weight_scale_bytes));
}

int max_weight_sync_chunks_per_shard(int64_t expert_fc1_numel,
                                     int64_t expert_fc2_numel,
                                     int64_t expert_fc1_weight_scale_bytes,
                                     int64_t expert_fc2_weight_scale_bytes,
                                     int weight_data_element_bytes) {
    int max_chunks = 0;
    const int chunks[4] = {
        kernels::weight_sync_num_chunks(
            static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))),
        kernels::weight_sync_num_chunks(
            static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))),
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc1_weight_scale_bytes)),
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc2_weight_scale_bytes)),
    };
    for (int value : chunks) {
        max_chunks = std::max(max_chunks, value);
    }
    return max_chunks;
}

}  // namespace

// ============================================================================
// GlobalExpertPlacement
// ============================================================================

void GlobalExpertPlacement::init(int num_layers, int P, int L, int R, int device_id) {
    num_layers_ = num_layers;
    p2l_numel = P;
    l2p_numel = L * R;
    lcnts_numel = L;
    quota_numel = L * R;
    quota_prefix_numel = L * R;
    rank_quota_numel = L * R;
    per_layer_data_numel = p2l_numel + l2p_numel + lcnts_numel + quota_numel + quota_prefix_numel;
    per_layer_data_bytes = per_layer_data_numel * static_cast<int>(sizeof(int32_t));

    // Align stride so each layer starts on a 256-byte boundary (good for DMA)
    per_layer_stride_bytes = (per_layer_data_bytes + ALIGNMENT_BYTES - 1) / ALIGNMENT_BYTES * ALIGNMENT_BYTES;
    per_layer_stride_numel = per_layer_stride_bytes / static_cast<int>(sizeof(int32_t));
    total_bytes = num_layers * per_layer_stride_bytes;

    // Allocate GPU buffer. Individual placement initializers set active fields.
    CUDA_RUNTIME_CHECK(cudaMalloc(&device_buffer, total_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(device_buffer, 0, total_bytes));
    // Create GPU tensor views with strides
    auto device_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    physical_to_logical_map =
        torch::from_blob(device_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, device_opts);
    logical_to_physical_map =
        torch::from_blob(device_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, device_opts);
    logical_replica_counts = torch::from_blob(
        device_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, device_opts);
    logical_instance_quota = torch::from_blob(device_buffer + p2l_numel + l2p_numel + lcnts_numel,
                                              {num_layers, L, R},
                                              {per_layer_stride_numel, R, 1},
                                              device_opts);
    logical_instance_quota_prefix = torch::from_blob(device_buffer + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                                                     {num_layers, L, R},
                                                     {per_layer_stride_numel, R, 1},
                                                     device_opts);
    rank_quota_prefix = torch::zeros({num_layers, L, R}, device_opts);
}

void GlobalExpertPlacement::cleanup() {
    if (device_buffer != nullptr) {
        cudaFree(device_buffer);
        device_buffer = nullptr;
    }
    physical_to_logical_map = torch::Tensor();
    logical_to_physical_map = torch::Tensor();
    logical_replica_counts = torch::Tensor();
    logical_instance_quota = torch::Tensor();
    logical_instance_quota_prefix = torch::Tensor();
    rank_quota_prefix = torch::Tensor();
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_device_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = device_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(base, base + p2l_numel, base + p2l_numel + l2p_numel);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_quota_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = device_buffer + layer_id * per_layer_stride_numel;
    int32_t* rank_quota_base =
        rank_quota_prefix.data_ptr<int32_t>() + static_cast<int64_t>(layer_id) * rank_quota_numel;
    return std::make_tuple(base + p2l_numel + l2p_numel + lcnts_numel,
                           base + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                           rank_quota_base);
}

// ============================================================================
// Manager
// ============================================================================

Manager::Manager(const int& num_layers,
                 const int& num_local_master_experts,
                 const int& num_local_redundant_experts,
                 const int64_t& expert_fc1_numel,
                 const int64_t& expert_fc2_numel,
                 const int& weight_data_element_bytes,
                 const int& weight_scale_element_bytes,
                 const int64_t& expert_fc1_weight_scale_numel,
                 const int64_t& expert_fc2_weight_scale_numel,
                 const int& grad_element_bytes,
                 const bool& is_train,
                 const bool& explicitly_destroy,
                 const bool& legacy_placement,
                 const float& balance_threshold,
                 const bool& quota_locality_aware,
                 const int32_t& quota_min_tokens_per_replica,
                 const bool& quota_allow_zero_master_quota,
                 const float& quota_oracle_eps,
                 const int& quota_kernel_stage,
                 const bool& quota_reroute_interleave,
                 const int& grad_reduce_num_sms,
                 const bool& grad_reduce_deterministic,
                 const int& weight_sync_plan_mode,
                 const int& weight_sync_relay_min_replicas,
                 const int& weight_sync_relay_max_relays,
                 const int& weight_sync_relay_min_fanout_gain,
                 const torch::Tensor& local_replica_weight_buffer_external,
                 const torch::Tensor& local_replica_weight_scale_buffer_external,
                 const torch::Tensor& local_weight_sync_ready_flags_external,
                 const torch::Tensor& local_replica_grad_buffer_external,
                 const torch::Tensor& remote_weight_ptrs_external,
                 const torch::Tensor& remote_weight_scale_ptrs_external,
                 const torch::Tensor& remote_grad_ptrs_external,
                 const torch::Tensor& remote_ready_ptrs_external)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      expert_fc1_weight_scale_numel(expert_fc1_weight_scale_numel),
      expert_fc2_weight_scale_numel(expert_fc2_weight_scale_numel),
      expert_weight_scale_total_numel(expert_fc1_weight_scale_numel + expert_fc2_weight_scale_numel),
      expert_fc1_weight_scale_bytes(0),
      expert_fc2_weight_scale_bytes(0),
      expert_weight_scale_fc2_offset_bytes(0),
      expert_weight_scale_stride_bytes(0),
      weight_data_element_bytes(weight_data_element_bytes),
      weight_scale_element_bytes(weight_scale_element_bytes),
      grad_element_bytes(grad_element_bytes),
      is_train(is_train),
      explicitly_destroy(explicitly_destroy),
      legacy_placement_(legacy_placement),
      quota_locality_aware_(quota_locality_aware),
      quota_min_tokens_per_replica_(quota_min_tokens_per_replica),
      quota_allow_zero_master_quota_(quota_allow_zero_master_quota),
      quota_oracle_eps_(quota_oracle_eps),
      quota_kernel_stage_(quota_kernel_stage),
      quota_reroute_interleave_(quota_reroute_interleave),
      grad_reduce_num_sms_(grad_reduce_num_sms),
      grad_reduce_deterministic_(grad_reduce_deterministic),
      weight_sync_plan_mode_(weight_sync_plan_mode),
      weight_sync_relay_min_replicas_(weight_sync_relay_min_replicas),
      weight_sync_relay_max_relays_(weight_sync_relay_max_relays),
      weight_sync_relay_min_fanout_gain_(weight_sync_relay_min_fanout_gain),
      balance_threshold_(balance_threshold),
      comm_stream(at::cuda::getStreamFromPool(true)),
      relay_stream(at::cuda::getStreamFromPool(true)),
      placement_ready_events_(is_train ? num_layers : 1),
      placement_ready_stream_ids_(is_train ? num_layers : 1, -1)

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    EP_HOST_ASSERT(weight_data_element_bytes > 0);
    EP_HOST_ASSERT(weight_scale_element_bytes > 0);
    EP_HOST_ASSERT(grad_element_bytes == static_cast<int>(sizeof(float)) &&
                   "UltraEP currently supports only fp32 gradients");
    EP_HOST_ASSERT(expert_fc1_numel >= 0 && expert_fc2_numel >= 0);
    EP_HOST_ASSERT(expert_fc1_weight_scale_numel >= 0 && expert_fc2_weight_scale_numel >= 0);
    const int64_t expert_fc1_weight_bytes = checked_num_bytes(expert_fc1_numel, weight_data_element_bytes);
    const int64_t expert_fc2_weight_bytes = checked_num_bytes(expert_fc2_numel, weight_data_element_bytes);
    EP_HOST_ASSERT(expert_fc1_weight_bytes % kernels::kNumTMAAlignBytes == 0 &&
                   "fc1 weight shard bytes must be 16-byte aligned for TMA weight_sync");
    EP_HOST_ASSERT(expert_fc2_weight_bytes % kernels::kNumTMAAlignBytes == 0 &&
                   "fc2 weight shard bytes must be 16-byte aligned for TMA weight_sync");
    expert_fc1_weight_scale_bytes = checked_num_bytes(expert_fc1_weight_scale_numel, weight_scale_element_bytes);
    expert_fc2_weight_scale_bytes = checked_num_bytes(expert_fc2_weight_scale_numel, weight_scale_element_bytes);
    expert_weight_scale_fc2_offset_bytes = align_up_bytes(expert_fc1_weight_scale_bytes, kernels::kNumTMAAlignBytes);
    expert_weight_scale_stride_bytes = align_up_bytes(
        expert_weight_scale_fc2_offset_bytes + expert_fc2_weight_scale_bytes, kernels::kNumTMAAlignBytes);
    EP_HOST_ASSERT(weight_sync_plan_mode_ >= static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
                   weight_sync_plan_mode_ <= static_cast<int>(kernels::WeightSyncPlanMode::kForceRelay));
    EP_HOST_ASSERT(weight_sync_relay_min_replicas_ >= 0);
    EP_HOST_ASSERT(weight_sync_relay_max_relays_ >= 1);
    EP_HOST_ASSERT(weight_sync_relay_min_fanout_gain_ >= 0);
    EP_HOST_ASSERT(grad_reduce_num_sms_ > 0);
    EP_HOST_ASSERT(grad_reduce_num_sms_ % 2 == 0 && "grad_reduce_num_sms must be even");
    grad_reduce_num_sms_ = std::min(grad_reduce_num_sms_, runtime::num_device_sms);
    EP_HOST_ASSERT((quota_kernel_stage_ == 0 || quota_kernel_stage_ == 1) &&
                   "quota kernel_stage supports only {0,1}; stage 2/3 has been removed");
    num_global_physical_experts = num_local_physical_experts * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;
    _weight_sync_task_capacity = num_local_physical_experts *
        weight_sync_chunks_for_shards(expert_fc1_numel,
                                      expert_fc2_numel,
                                      expert_fc1_weight_scale_bytes,
                                      expert_fc2_weight_scale_bytes,
                                      weight_data_element_bytes);

    // Allocate global placement tensors using contiguous per-layer device buffers.
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.init(num_layers,
                   num_global_physical_experts,
                   num_global_logical_experts,
                   num_ranks,  // max_replicas_dim = num_ranks
                   device_id);

    const int64_t expert_total_weight_bytes = checked_num_bytes(expert_total_numel, weight_data_element_bytes);
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id));
    auto validate_cuda_tensor = [&](const torch::Tensor& tensor, torch::ScalarType dtype, const char* name) {
        EP_HOST_ASSERT(tensor.defined() && tensor.is_cuda() && tensor.dtype() == dtype && tensor.is_contiguous());
        EP_HOST_ASSERT(tensor.device().index() == device_id && "Symmetric buffers must use the runtime CUDA device");
        (void)name;
    };

    // The Python side allocates and rendezvouses these buffers with
    // torch.distributed._symmetric_memory.  C++ only keeps tensor references and
    // extracts local/remote addresses; it never owns or frees the allocations.
    validate_cuda_tensor(local_replica_weight_buffer_external, torch::kUInt8, "weight buffer");
    EP_HOST_ASSERT(local_replica_weight_buffer_external.dim() == 2 &&
                   local_replica_weight_buffer_external.size(0) == num_local_redundant_experts &&
                   local_replica_weight_buffer_external.size(1) == expert_total_weight_bytes);
    local_replica_weight_buffer_tensor = local_replica_weight_buffer_external;
    local_replica_weight_buffer = local_replica_weight_buffer_tensor.data_ptr();
    local_replica_fc1_weight_buffer_tensor = local_replica_weight_buffer_tensor.narrow(1, 0, expert_fc1_weight_bytes);
    local_replica_fc2_weight_buffer_tensor =
        local_replica_weight_buffer_tensor.narrow(1, expert_fc1_weight_bytes, expert_fc2_weight_bytes);

    if (expert_weight_scale_total_numel > 0) {
        validate_cuda_tensor(local_replica_weight_scale_buffer_external, torch::kUInt8, "weight-scale buffer");
        EP_HOST_ASSERT(local_replica_weight_scale_buffer_external.dim() == 2 &&
                       local_replica_weight_scale_buffer_external.size(0) == num_local_redundant_experts &&
                       local_replica_weight_scale_buffer_external.size(1) == expert_weight_scale_stride_bytes);
        local_replica_weight_scale_buffer_tensor = local_replica_weight_scale_buffer_external;
        local_replica_weight_scale_buffer = local_replica_weight_scale_buffer_tensor.data_ptr();
        local_replica_fc1_weight_scale_buffer_tensor =
            local_replica_weight_scale_buffer_tensor.narrow(1, 0, expert_fc1_weight_scale_bytes);
        local_replica_fc2_weight_scale_buffer_tensor = local_replica_weight_scale_buffer_tensor.narrow(
            1, expert_weight_scale_fc2_offset_bytes, expert_fc2_weight_scale_bytes);
    } else {
        local_replica_weight_scale_buffer_tensor =
            torch::empty({num_local_redundant_experts, 0}, byte_opts);
        local_replica_fc1_weight_scale_buffer_tensor = local_replica_weight_scale_buffer_tensor;
        local_replica_fc2_weight_scale_buffer_tensor = local_replica_weight_scale_buffer_tensor;
    }

    const int max_relay_chunks_per_shard = max_weight_sync_chunks_per_shard(expert_fc1_numel,
                                                                            expert_fc2_numel,
                                                                            expert_fc1_weight_scale_bytes,
                                                                            expert_fc2_weight_scale_bytes,
                                                                            weight_data_element_bytes);
    const int64_t local_ready_flag_count = static_cast<int64_t>(num_local_redundant_experts) *
        weight_sync_num_shards(expert_weight_scale_total_numel) * max_relay_chunks_per_shard;
    validate_cuda_tensor(local_weight_sync_ready_flags_external, torch::kUInt64, "ready-flag buffer");
    EP_HOST_ASSERT(local_weight_sync_ready_flags_external.dim() == 1 &&
                   local_weight_sync_ready_flags_external.numel() >= std::max<int64_t>(local_ready_flag_count, 1));
    local_weight_sync_ready_flags = local_weight_sync_ready_flags_external.data_ptr<uint64_t>();
    CUDA_RUNTIME_CHECK(cudaMemset(local_weight_sync_ready_flags,
                                  0,
                                  std::max<int64_t>(local_ready_flag_count, 1) * sizeof(uint64_t)));

    // Grad buffer only needed for training
    if (is_train) {
        validate_cuda_tensor(local_replica_grad_buffer_external, torch::kFloat32, "grad buffer");
        EP_HOST_ASSERT(local_replica_grad_buffer_external.dim() == 2 &&
                       local_replica_grad_buffer_external.size(0) == num_local_redundant_experts &&
                       local_replica_grad_buffer_external.size(1) == expert_total_numel);
        local_replica_grad_buffer_tensor = local_replica_grad_buffer_external;
        local_replica_grad_buffer = local_replica_grad_buffer_tensor.data_ptr();
        local_replica_fc1_grad_buffer_tensor = local_replica_grad_buffer_tensor.narrow(1, 0, expert_fc1_numel);
        local_replica_fc2_grad_buffer_tensor =
            local_replica_grad_buffer_tensor.narrow(1, expert_fc1_numel, expert_fc2_numel);
        local_replica_grad_buffer_tensor.zero_();
    } else {
        local_replica_grad_buffer_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(
                                                               torch::Device(torch::kCUDA, device_id)));
    }

    copy_remote_pointer_table(remote_weight_ptrs_external, _remote_weight_ptrs);
    if (expert_weight_scale_total_numel > 0) {
        copy_remote_pointer_table(remote_weight_scale_ptrs_external, _remote_weight_scale_ptrs);
    }
    if (is_train) {
        copy_remote_pointer_table(remote_grad_ptrs_external, _remote_grad_ptrs);
    }
    if (local_weight_sync_ready_flags_external.defined()) {
        copy_remote_pointer_table(remote_ready_ptrs_external, _remote_ready_flag_ptrs);
    }

    // Allocate intermediate buffers for task-build and persistent kernels.
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_grad_reduce_tasks, kernels::kMaxGradReduceTaskCount * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_global_task_or_tile_counter, sizeof(int)));
    const int shared_task_capacity = std::max(kernels::kMaxGradReduceTaskCount, _weight_sync_task_capacity);
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_tile_offsets, (shared_task_capacity + 1) * sizeof(int)));

    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_weight_sync_task_remaining_tiles, _weight_sync_task_capacity * sizeof(int)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_relay_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_task_tile_offsets, (_weight_sync_task_capacity + 1) * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_task_metadata, 2 * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_global_tile_counter, sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_metadata, 2 * sizeof(int)));

    kernels::TaskBuildConfig config_cpu = {};
    fill_task_build_config(config_cpu,
                           num_local_master_experts,
                           num_local_physical_experts,
                           num_local_redundant_experts,
                           expert_fc1_numel,
                           expert_fc2_numel,
                           expert_total_numel,
                           expert_fc1_weight_scale_numel,
                           expert_fc2_weight_scale_numel,
                           expert_weight_scale_total_numel,
                           expert_fc1_weight_scale_bytes,
                           expert_fc2_weight_scale_bytes,
                           expert_weight_scale_fc2_offset_bytes,
                           expert_weight_scale_stride_bytes,
                           weight_data_element_bytes,
                           weight_scale_element_bytes,
                           weight_sync_plan_mode_,
                           weight_sync_relay_min_replicas_,
                           weight_sync_relay_max_relays_,
                           weight_sync_relay_min_fanout_gain_);
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_build_config, sizeof(kernels::TaskBuildConfig)));
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(_task_build_config, &config_cpu, sizeof(kernels::TaskBuildConfig), cudaMemcpyHostToDevice));

    const int max_stage_tiles_per_expert = weight_sync_tiles_for_shards(expert_fc1_numel,
                                                                        expert_fc2_numel,
                                                                        expert_fc1_weight_scale_bytes,
                                                                        expert_fc2_weight_scale_bytes,
                                                                        weight_data_element_bytes);
    _max_ws_total_tiles = num_local_physical_experts * max_stage_tiles_per_expert;

    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_reroute_sparse_counters, num_global_logical_experts * sizeof(int)));

    auto placement_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    global_logical_expert_loads_tensor = torch::empty({num_global_logical_experts}, placement_opts);
    local_expert_loads_tensor = torch::empty({num_global_logical_experts}, placement_opts);
    expert_loads_per_rank_tensor =
        torch::empty({runtime::num_ranks, num_global_logical_experts}, placement_opts);
    // Initialize default placement (master-only) for all layers so sparse reroute
    // remains valid before the first placement update.
    for (int lid = 0; lid < num_layers; ++lid) {
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_device_ptrs(lid);
        auto [quota_ptr, quota_prefix_ptr, rank_quota_prefix_ptr] = placement.get_quota_ptrs(lid);
        kernels::init_master_placement(p2l_ptr,
                                       l2p_ptr,
                                       lcnts_ptr,
                                       quota_ptr,
                                       quota_prefix_ptr,
                                       rank_quota_prefix_ptr,
                                       at::cuda::getCurrentCUDAStream().stream(),
                                       num_global_physical_experts,
                                       num_global_logical_experts,
                                       runtime::num_ranks,
                                       num_local_master_experts,
                                       num_local_redundant_experts,
                                       runtime::num_ranks);
    }
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));

    // Ready to use.  Symmetric Memory handles and ownership stay on the Python
    // side; C++ has retained the local tensor views and device pointer tables.
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

    // The Python owner must keep symmetric allocations alive until this point.
    // C++ only releases its Tensor references after all local work is complete.
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    local_replica_weight_buffer = nullptr;
    local_replica_weight_scale_buffer = nullptr;
    local_weight_sync_ready_flags = nullptr;
    local_replica_grad_buffer = nullptr;
    local_replica_weight_buffer_tensor = torch::Tensor();
    local_replica_fc1_weight_buffer_tensor = torch::Tensor();
    local_replica_fc2_weight_buffer_tensor = torch::Tensor();
    local_replica_weight_scale_buffer_tensor = torch::Tensor();
    local_replica_fc1_weight_scale_buffer_tensor = torch::Tensor();
    local_replica_fc2_weight_scale_buffer_tensor = torch::Tensor();
    local_replica_grad_buffer_tensor = torch::Tensor();
    local_replica_fc1_grad_buffer_tensor = torch::Tensor();
    local_replica_fc2_grad_buffer_tensor = torch::Tensor();
    global_logical_expert_loads_tensor = torch::Tensor();
    local_expert_loads_tensor = torch::Tensor();
    expert_loads_per_rank_tensor = torch::Tensor();

    // Free intermediate CUDA buffers
    CUDA_RUNTIME_CHECK(cudaFree(_grad_reduce_tasks));
    CUDA_RUNTIME_CHECK(cudaFree(_global_task_or_tile_counter));
    CUDA_RUNTIME_CHECK(cudaFree(_task_tile_offsets));
    _grad_reduce_tasks = nullptr;
    _global_task_or_tile_counter = nullptr;
    _task_tile_offsets = nullptr;

    // Free weight sync buffers
    CUDA_RUNTIME_CHECK(cudaFree(_weight_sync_tasks));
    CUDA_RUNTIME_CHECK(cudaFree(_weight_sync_task_remaining_tiles));
    CUDA_RUNTIME_CHECK(cudaFree(_relay_weight_sync_tasks));
    CUDA_RUNTIME_CHECK(cudaFree(_relay_task_tile_offsets));
    CUDA_RUNTIME_CHECK(cudaFree(_relay_task_metadata));
    CUDA_RUNTIME_CHECK(cudaFree(_relay_global_tile_counter));
    _weight_sync_tasks = nullptr;
    _weight_sync_task_remaining_tiles = nullptr;
    _relay_weight_sync_tasks = nullptr;
    _relay_task_tile_offsets = nullptr;
    _relay_task_metadata = nullptr;
    _relay_global_tile_counter = nullptr;
    _weight_sync_task_capacity = 0;
    _weight_sync_epoch = 0;

    // Free task metadata buffer
    CUDA_RUNTIME_CHECK(cudaFree(_task_metadata));
    _task_metadata = nullptr;

    // Free device task build buffers
    if (_task_build_config) {
        CUDA_RUNTIME_CHECK(cudaFree(_task_build_config));
        _task_build_config = nullptr;
    }
    if (_remote_weight_ptrs) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_weight_ptrs));
        _remote_weight_ptrs = nullptr;
    }
    if (_remote_weight_scale_ptrs) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_weight_scale_ptrs));
        _remote_weight_scale_ptrs = nullptr;
    }
    if (_remote_grad_ptrs) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_grad_ptrs));
        _remote_grad_ptrs = nullptr;
    }
    if (_remote_ready_flag_ptrs) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_ready_flag_ptrs));
        _remote_ready_flag_ptrs = nullptr;
    }

    // Free sparse reroute counters
    CUDA_RUNTIME_CHECK(cudaFree(_reroute_sparse_counters));
    _reroute_sparse_counters = nullptr;

    // Free contiguous placement buffers (CPU pinned + GPU)
    placement.cleanup();

    // Bound barrier methods retain the symmetric handles too. Release them
    // before Python frees the owning regions and tears down process groups.
    weight_lifetime_barrier_ = pybind11::function();
    grad_lifetime_barrier_ = pybind11::function();

    // Ready to destroy
    _available = false;
}

void Manager::record_placement_ready(const int layer_id, const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    const int slot = placement_sync_slot(layer_id);
    placement_ready_events_[slot] = EventHandle(stream);
    placement_ready_stream_ids_[slot] = static_cast<int64_t>(stream.id());
}

void Manager::wait_for_placement_ready(const int layer_id, const at::cuda::CUDAStream& stream) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    const int slot = placement_sync_slot(layer_id);
    if (!placement_ready_events_[slot].has_value()) {
        return;
    }
    if (placement_ready_stream_ids_[slot] == static_cast<int64_t>(stream.id())) {
        return;
    }
    stream_wait(stream, placement_ready_events_[slot].value());
}

void Manager::compute_local_loads(torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(routing_map.is_cuda() && routing_map.is_contiguous() &&
                   routing_map.dim() == 2 && routing_map.size(1) == num_global_logical_experts &&
                   routing_map.dtype() == torch::kBool);

    auto stream = at::cuda::getCurrentCUDAStream();

    kernels::rmap_local_sum(routing_map.size(0),
                            num_global_logical_experts,
                            routing_map.data_ptr<bool>(),
                            local_expert_loads_tensor.data_ptr<int32_t>(),
                            stream.stream());
}

void Manager::compute_local_loads_sparse(torch::Tensor& topk_ids) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(topk_ids.is_cuda() && topk_ids.is_contiguous() && topk_ids.dtype() == torch::kInt64);
    EP_HOST_ASSERT(topk_ids.dim() == 2);

    auto stream = at::cuda::getCurrentCUDAStream();
    kernels::topk_local_sum(topk_ids.data_ptr<int64_t>(),
                            topk_ids.size(0),
                            topk_ids.size(1),
                            num_global_logical_experts,
                            local_expert_loads_tensor.data_ptr<int32_t>(),
                            stream.stream());
}

void Manager::solve_placement_legacy(const int& layer_id, torch::Tensor& global_loads) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(legacy_placement_ && layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(global_loads.is_cuda() && global_loads.is_contiguous() &&
                   global_loads.dtype() == torch::kInt32 && global_loads.dim() == 1 &&
                   global_loads.size(0) == num_global_logical_experts);

    auto stream = at::cuda::getCurrentCUDAStream();
    if (global_loads.data_ptr<int32_t>() != global_logical_expert_loads_tensor.data_ptr<int32_t>()) {
        global_logical_expert_loads_tensor.copy_(global_loads);
    }
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);
    kernels::legacy::solve_placement(global_logical_expert_loads_tensor.data_ptr<int32_t>(),
                                     nullptr,
                                     physical_to_logical_map,
                                     logical_to_physical_map,
                                     logical_replica_counts,
                                     logical_instance_quota,
                                     logical_instance_quota_prefix,
                                     rank_quota_prefix,
                                     stream.stream(),
                                     num_global_logical_experts,
                                     runtime::num_ranks,
                                     num_local_master_experts,
                                     num_local_redundant_experts,
                                     runtime::num_nvl_ranks,
                                     runtime::num_ranks,
                                     balance_threshold_,
                                     quota_min_tokens_per_replica_,
                                     quota_allow_zero_master_quota_,
                                     quota_locality_aware_,
                                     quota_oracle_eps_,
                                     quota_kernel_stage_);
    record_placement_ready(layer_id, stream);
}

void Manager::solve_placement(const int& layer_id, torch::Tensor& loads_per_rank) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(!legacy_placement_ && layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(loads_per_rank.is_cuda() && loads_per_rank.is_contiguous() &&
                   loads_per_rank.dtype() == torch::kInt32 && loads_per_rank.dim() == 2 &&
                   loads_per_rank.size(0) == runtime::num_ranks &&
                   loads_per_rank.size(1) == num_global_logical_experts);

    auto stream = at::cuda::getCurrentCUDAStream();
    if (loads_per_rank.data_ptr<int32_t>() != expert_loads_per_rank_tensor.data_ptr<int32_t>()) {
        expert_loads_per_rank_tensor.copy_(loads_per_rank);
    }
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);
    kernels::reduce_per_rank_loads(expert_loads_per_rank_tensor.data_ptr<int32_t>(),
                                   global_logical_expert_loads_tensor.data_ptr<int32_t>(),
                                   runtime::num_ranks,
                                   num_global_logical_experts,
                                   stream.stream());
    kernels::solve_placement(global_logical_expert_loads_tensor.data_ptr<int32_t>(),
                             expert_loads_per_rank_tensor.data_ptr<int32_t>(),
                             physical_to_logical_map,
                             logical_to_physical_map,
                             logical_replica_counts,
                             logical_instance_quota,
                             logical_instance_quota_prefix,
                             rank_quota_prefix,
                             stream.stream(),
                             num_global_logical_experts,
                             runtime::num_ranks,
                             num_local_master_experts,
                             num_local_redundant_experts,
                             runtime::num_nvl_ranks,
                             runtime::num_ranks,
                             balance_threshold_,
                             quota_min_tokens_per_replica_,
                             quota_allow_zero_master_quota_,
                             quota_locality_aware_,
                             quota_oracle_eps_,
                             quota_kernel_stage_);
    record_placement_ready(layer_id, stream);
}

void Manager::reroute_sparse(const int& layer_id, torch::Tensor& topk_ids) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(topk_ids.is_cuda() && topk_ids.dtype() == torch::kInt64);
    EP_HOST_ASSERT(topk_ids.dim() == 2);

    int T = topk_ids.size(0);
    int K = topk_ids.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    wait_for_placement_ready(layer_id, stream);

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;

    if (!legacy_placement_) {
        const int32_t* rank_quota_prefix = std::get<2>(placement.get_quota_ptrs(layer_id));
        kernels::run_sparse_reroute_quota(topk_ids.data_ptr<int64_t>(),
                                          logical_to_physical_map,
                                          logical_replica_counts,
                                          rank_quota_prefix,
                                          _reroute_sparse_counters,
                                          T,
                                          K,
                                          num_global_logical_experts,
                                          runtime::num_ranks,
                                          stream);
    } else {
        kernels::run_sparse_reroute_round_robin(topk_ids.data_ptr<int64_t>(),
                                                logical_to_physical_map,
                                                logical_replica_counts,
                                                _reroute_sparse_counters,
                                                T,
                                                K,
                                                num_global_logical_experts,
                                                runtime::num_ranks,
                                                stream);
    }
}

std::tuple<torch::Tensor, torch::Tensor> Manager::dense_reroute_forward(const int& layer_id,
                                                                        torch::Tensor& probs,
                                                                        torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(routing_map.is_cuda() && probs.is_cuda());
    EP_HOST_ASSERT(routing_map.dtype() == torch::kBool);

    probs = probs.contiguous();
    routing_map = routing_map.contiguous();

    const int T = routing_map.size(0);
    const int L = routing_map.size(1);
    const int P = num_global_physical_experts;
    auto device = routing_map.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    wait_for_placement_ready(layer_id, stream);

    auto expanded_probs = torch::zeros({T, P}, torch::TensorOptions().dtype(probs.scalar_type()).device(device));
    auto expanded_rmap = torch::zeros({T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
    void* expand_probs_ptr = expanded_probs.data_ptr();
    bool* expand_rmap_ptr = expanded_rmap.data_ptr<bool>();

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;

    if (T > 0 && L > 0) {
        constexpr int TILE_T = kernels::kDenseRerouteTileTokens;
        const int num_tiles = (T + TILE_T - 1) / TILE_T;

        int64_t tile_count_numel = static_cast<int64_t>(L) * num_tiles;
        torch::Tensor tile_counts_tensor =
            torch::empty({tile_count_numel}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int32_t* tile_counts_ptr = tile_counts_tensor.data_ptr<int32_t>();

        EP_HOST_ASSERT(probs.scalar_type() == torch::kFloat32);

        if (!legacy_placement_) {
            const int32_t* rank_quota_prefix = std::get<2>(placement.get_quota_ptrs(layer_id));
            kernels::run_dense_reroute_forward_quota(routing_map.data_ptr<bool>(),
                                                     probs.data_ptr(),
                                                     logical_to_physical_map,
                                                     logical_replica_counts,
                                                     rank_quota_prefix,
                                                     expand_rmap_ptr,
                                                     expand_probs_ptr,
                                                     tile_counts_ptr,
                                                     T,
                                                     L,
                                                     P,
                                                     runtime::num_ranks,
                                                     quota_reroute_interleave_,
                                                     stream);
        } else {
            kernels::run_dense_reroute_forward_round_robin(routing_map.data_ptr<bool>(),
                                                           probs.data_ptr(),
                                                           logical_to_physical_map,
                                                           logical_replica_counts,
                                                           expand_rmap_ptr,
                                                           expand_probs_ptr,
                                                           tile_counts_ptr,
                                                           T,
                                                           L,
                                                           P,
                                                           runtime::num_ranks,
                                                           stream);
        }
    }

    return std::make_tuple(expanded_probs, expanded_rmap);
}

torch::Tensor Manager::dense_reroute_backward(const int& layer_id,
                                              torch::Tensor& grad_expanded_probs,
                                              torch::Tensor& routing_map,
                                              torch::Tensor& expanded_routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(grad_expanded_probs.is_cuda() && routing_map.is_cuda());

    grad_expanded_probs = grad_expanded_probs.contiguous();
    routing_map = routing_map.contiguous();

    const int T = routing_map.size(0);
    const int L = routing_map.size(1);
    const int P = grad_expanded_probs.size(1);
    auto device = routing_map.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto grad_probs = torch::zeros({T, L}, torch::TensorOptions().dtype(grad_expanded_probs.dtype()).device(device));
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;
    const bool* rerouted_map = expanded_routing_map.data_ptr<bool>();

    if (T > 0 && L > 0) {
        EP_HOST_ASSERT(grad_expanded_probs.scalar_type() == torch::kFloat32);

        kernels::run_dense_reroute_backward(grad_expanded_probs.data_ptr(),
                                            routing_map.data_ptr<bool>(),
                                            rerouted_map,
                                            logical_to_physical_map,
                                            logical_replica_counts,
                                            grad_probs.data_ptr(),
                                            T,
                                            L,
                                            P,
                                            runtime::num_ranks,
                                            stream);
    }

    return grad_probs;
}

void Manager::set_weight_sync_plan_mode(const int& plan_mode) {
    EP_HOST_ASSERT(plan_mode >= static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
                   plan_mode <= static_cast<int>(kernels::WeightSyncPlanMode::kForceRelay));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    weight_sync_plan_mode_ = plan_mode;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (local_weight_sync_ready_flags != nullptr) {
        const int max_relay_chunks_per_shard = max_weight_sync_chunks_per_shard(expert_fc1_numel,
                                                                                expert_fc2_numel,
                                                                                expert_fc1_weight_scale_bytes,
                                                                                expert_fc2_weight_scale_bytes,
                                                                                weight_data_element_bytes);
        const int64_t local_ready_flag_count = static_cast<int64_t>(num_local_redundant_experts) *
            weight_sync_num_shards(expert_weight_scale_total_numel) * max_relay_chunks_per_shard;
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            local_weight_sync_ready_flags, 0, local_ready_flag_count * sizeof(uint64_t), stream.stream()));
    }
    kernels::TaskBuildConfig config_cpu = {};
    fill_task_build_config(config_cpu,
                           num_local_master_experts,
                           num_local_physical_experts,
                           num_local_redundant_experts,
                           expert_fc1_numel,
                           expert_fc2_numel,
                           expert_total_numel,
                           expert_fc1_weight_scale_numel,
                           expert_fc2_weight_scale_numel,
                           expert_weight_scale_total_numel,
                           expert_fc1_weight_scale_bytes,
                           expert_fc2_weight_scale_bytes,
                           expert_weight_scale_fc2_offset_bytes,
                           expert_weight_scale_stride_bytes,
                           weight_data_element_bytes,
                           weight_scale_element_bytes,
                           weight_sync_plan_mode_,
                           weight_sync_relay_min_replicas_,
                           weight_sync_relay_max_relays_,
                           weight_sync_relay_min_fanout_gain_);
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        _task_build_config, &config_cpu, sizeof(kernels::TaskBuildConfig), cudaMemcpyHostToDevice, stream.stream()));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream.stream()));
}

void Manager::set_grad_reduce_deterministic(const bool& deterministic, const int& grad_reduce_num_sms) {
    EP_HOST_ASSERT(grad_reduce_num_sms > 0);
    EP_HOST_ASSERT(grad_reduce_num_sms % 2 == 0 && "grad_reduce_num_sms must be even");
    grad_reduce_deterministic_ = deterministic;
    grad_reduce_num_sms_ = std::min(grad_reduce_num_sms, runtime::num_device_sms);
}

std::tuple<int32_t*, int32_t*, int32_t*> Manager::get_placement_ptrs(
    int layer_id,
    const std::optional<torch::Tensor>& physical_to_logical_map,
    const std::optional<torch::Tensor>& logical_to_physical_map,
    const std::optional<torch::Tensor>& logical_replica_counts) const {
    TORCH_CHECK(layer_id >= 0 && layer_id < num_layers, "Invalid virtual layer_id");
    const bool external = physical_to_logical_map.has_value();
    TORCH_CHECK(external == logical_to_physical_map.has_value() &&
                    external == logical_replica_counts.has_value(),
                "Supply all three placement tensors or omit all three");
    if (!external) {
        return placement.get_device_ptrs(layer_id);
    }

    auto checked_ptr = [&](const torch::Tensor& map, at::IntArrayRef shape, const char* name) {
        TORCH_CHECK(map.is_cuda() && map.get_device() == runtime::device_id,
                    name, " must be on the manager's CUDA device");
        TORCH_CHECK(map.scalar_type() == torch::kInt32, name, " must have dtype int32");
        TORCH_CHECK(map.sizes() == shape, name, " must have shape ", shape);
        TORCH_CHECK(map.is_contiguous(), name, " must be contiguous");
        // External tensors may be released before the asynchronous task build completes.
        c10::cuda::CUDACachingAllocator::recordStream(map.storage().data_ptr(), comm_stream);
        return map.data_ptr<int32_t>();
    };
    return std::make_tuple(
        checked_ptr(*physical_to_logical_map, {num_global_physical_experts}, "physical_to_logical_map"),
        checked_ptr(*logical_to_physical_map, {num_global_logical_experts, runtime::num_ranks}, "logical_to_physical_map"),
        checked_ptr(*logical_replica_counts, {num_global_logical_experts}, "logical_replica_counts"));
}

void Manager::set_lifetime_barriers(pybind11::function weight_barrier, pybind11::function grad_barrier) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(!weight_lifetime_barrier_ && !grad_lifetime_barrier_);
    weight_lifetime_barrier_ = std::move(weight_barrier);
    grad_lifetime_barrier_ = std::move(grad_barrier);
}

std::optional<EventHandle> Manager::grad_reduce(const int& layer_id,
                                                torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                                torch::Tensor& local_master_fc2_grad_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async,
                                                bool use_current_stream,
                                                const std::optional<torch::Tensor>& physical_to_logical_map_tensor,
                                                const std::optional<torch::Tensor>& logical_to_physical_map_tensor,
                                                const std::optional<torch::Tensor>& logical_replica_counts_tensor) {
    EP_HOST_ASSERT(is_available());
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        get_placement_ptrs(layer_id,
                          physical_to_logical_map_tensor,
                          logical_to_physical_map_tensor,
                          logical_replica_counts_tensor);

    const auto caller_stream = at::cuda::getCurrentCUDAStream();
    // Combined 1F1B can place GR directly on its communication stream.  This removes
    // UltraEP's otherwise-private third stream from the full-layer CUDA graph while
    // preserving the default private-stream behavior for all existing callers.
    const auto launch_stream = use_current_stream ? caller_stream : comm_stream;
    const c10::cuda::CUDAStreamGuard stream_guard(launch_stream);
    EP_HOST_ASSERT(grad_lifetime_barrier_);
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    if (previous_event.has_value()) {
        stream_wait(launch_stream, previous_event.value());
    } else if (launch_stream.id() != caller_stream.id()) {
        stream_wait(launch_stream, caller_stream);
    }

    grad_lifetime_barrier_(0);

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.numel() == num_local_master_experts);

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_grad_ptrs = local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_grad_ptrs = local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>();

    kernels::build_grad_reduce_tasks(_task_build_config,
                                     physical_to_logical_map,
                                     logical_to_physical_map,
                                     logical_replica_counts,
                                     _remote_grad_ptrs,
                                     local_master_fc1_grad_ptrs,
                                     local_master_fc2_grad_ptrs,
                                     _grad_reduce_tasks,
                                     _task_tile_offsets,
                                     _task_metadata,
                                     _global_task_or_tile_counter,
                                     launch_stream);

    kernels::run_grad_reduce(_grad_reduce_tasks,
                             _task_tile_offsets,
                             _task_metadata,
                             _global_task_or_tile_counter,
                             launch_stream,
                             grad_reduce_num_sms_,
                             grad_reduce_deterministic_);

    // Completion includes remote reads and clears before shared-buffer reuse.
    grad_lifetime_barrier_(1);

    // Wait streams
    if (async) {
        event = EventHandle(launch_stream);
    } else if (launch_stream.id() != caller_stream.id()) {
        stream_wait(caller_stream, launch_stream);
    }

    return event;
}

std::optional<EventHandle> Manager::weight_sync(const int& layer_id,
                                                torch::Tensor& local_master_fc1_weight_ptr_tensor,
                                                torch::Tensor& local_master_fc2_weight_ptr_tensor,
                                                torch::Tensor& local_master_fc1_weight_scale_ptr_tensor,
                                                torch::Tensor& local_master_fc2_weight_scale_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async,
                                                const std::optional<torch::Tensor>& physical_to_logical_map_tensor,
                                                const std::optional<torch::Tensor>& logical_to_physical_map_tensor,
                                                const std::optional<torch::Tensor>& logical_replica_counts_tensor) {
    EP_HOST_ASSERT(is_available());
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        get_placement_ptrs(layer_id,
                          physical_to_logical_map_tensor,
                          logical_to_physical_map_tensor,
                          logical_replica_counts_tensor);

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    const c10::cuda::CUDAStreamGuard stream_guard(comm_stream);
    EP_HOST_ASSERT(weight_lifetime_barrier_);
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else if (comm_stream.id() != compute_stream.id()) {
        stream_wait(comm_stream, compute_stream);
    }

    weight_lifetime_barrier_(0);

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.numel() == num_local_master_experts);
    if (expert_weight_scale_total_numel > 0) {
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.dtype() == torch::kInt64);
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.dtype() == torch::kInt64);
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.numel() == num_local_master_experts);
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.numel() == num_local_master_experts);
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.is_cuda());
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.is_cuda());
    }
    const bool enable_relay_stages = weight_sync_plan_mode_ != static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
        num_local_redundant_experts > 0 && runtime::num_nvl_ranks > 2;
    const uint64_t current_epoch = ++_weight_sync_epoch;
    bool launched_stage2 = false;

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_weight_ptrs = local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_weight_ptrs = local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc1_weight_scale_ptrs =
        expert_weight_scale_total_numel > 0 ? local_master_fc1_weight_scale_ptr_tensor.data_ptr<int64_t>() : nullptr;
    int64_t* local_master_fc2_weight_scale_ptrs =
        expert_weight_scale_total_numel > 0 ? local_master_fc2_weight_scale_ptr_tensor.data_ptr<int64_t>() : nullptr;

    kernels::build_weight_sync_task_lists(_task_build_config,
                                          physical_to_logical_map,
                                          logical_to_physical_map,
                                          logical_replica_counts,
                                          _remote_weight_ptrs,
                                          _remote_weight_scale_ptrs,
                                          local_master_fc1_weight_ptrs,
                                          local_master_fc2_weight_ptrs,
                                          local_master_fc1_weight_scale_ptrs,
                                          local_master_fc2_weight_scale_ptrs,
                                          reinterpret_cast<uint8_t*>(local_replica_weight_buffer),
                                          reinterpret_cast<uint8_t*>(local_replica_weight_scale_buffer),
                                          _weight_sync_tasks,
                                          _task_tile_offsets,
                                          _task_metadata,
                                          _weight_sync_task_remaining_tiles,
                                          _global_task_or_tile_counter,
                                          _relay_weight_sync_tasks,
                                          _relay_task_tile_offsets,
                                          _relay_task_metadata,
                                          _relay_global_tile_counter,
                                          comm_stream);
    // Only relay mode needs a task-build event and a second-stream dependency.
    if (enable_relay_stages) {
        stream_wait(relay_stream, comm_stream);
    }

    kernels::run_weight_sync(_weight_sync_tasks,
                             _task_tile_offsets,
                             _task_metadata,
                             _global_task_or_tile_counter,
                             _weight_sync_task_remaining_tiles,
                             local_weight_sync_ready_flags,
                             _remote_ready_flag_ptrs,
                             current_epoch,
                             comm_stream,
                             runtime::num_device_sms,
                             runtime::num_nvl_ranks,
                             _max_ws_total_tiles,
                             2);

    if (enable_relay_stages) {
        kernels::run_weight_sync(_relay_weight_sync_tasks,
                                 _relay_task_tile_offsets,
                                 _relay_task_metadata,
                                 _relay_global_tile_counter,
                                 nullptr,
                                 local_weight_sync_ready_flags,
                                 _remote_ready_flag_ptrs,
                                 current_epoch,
                                 relay_stream,
                                 runtime::num_device_sms,
                                 runtime::num_nvl_ranks,
                                 _max_ws_total_tiles,
                                 1);
        launched_stage2 = true;
    }

    if (launched_stage2) {
        stream_wait(comm_stream, relay_stream);
    }

    // Record/return a single completion event after incoming writes are ready.
    weight_lifetime_barrier_(1);

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else if (compute_stream.id() != comm_stream.id()) {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

}  // namespace ultra_ep
