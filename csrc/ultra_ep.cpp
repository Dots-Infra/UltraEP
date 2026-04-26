#include "ultra_ep.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace ultra_ep {

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

    // Allocate CPU pinned buffer
    CUDA_RUNTIME_CHECK(cudaMallocHost(&cpu_buffer, total_bytes));
    quota_buf_cpu = cpu_buffer + p2l_numel + l2p_numel + lcnts_numel;

    // Initialize: p2l/l2p -> -1, lcnts/quota/quota_prefix -> 0, padding -> 0
    std::memset(cpu_buffer, 0xFF, total_bytes);
    for (int i = 0; i < num_layers; ++i) {
        int32_t* layer_base = cpu_buffer + i * per_layer_stride_numel;
        // Zero out lcnts
        std::memset(layer_base + p2l_numel + l2p_numel, 0, lcnts_numel * sizeof(int32_t));
        // Zero out quota arrays
        std::memset(layer_base + p2l_numel + l2p_numel + lcnts_numel, 0, quota_numel * sizeof(int32_t));
        std::memset(
            layer_base + p2l_numel + l2p_numel + lcnts_numel + quota_numel, 0, quota_prefix_numel * sizeof(int32_t));
        // Zero out padding
        int pad_numel = per_layer_stride_numel - per_layer_data_numel;
        if (pad_numel > 0) {
            std::memset(layer_base + per_layer_data_numel, 0, pad_numel * sizeof(int32_t));
        }
    }

    // Allocate GPU buffer (zero-initialized)
    CUDA_RUNTIME_CHECK(cudaMalloc(&device_buffer, total_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(device_buffer, 0, total_bytes));
    quota_buf_device = device_buffer + p2l_numel + l2p_numel + lcnts_numel;

    // Create CPU tensor views with strides
    // Shape [num_layers, X] with stride[0] = per_layer_stride_numel
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    physical_to_logical_map = torch::from_blob(cpu_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, cpu_opts);
    logical_to_physical_map =
        torch::from_blob(cpu_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, cpu_opts);
    logical_replica_counts =
        torch::from_blob(cpu_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, cpu_opts);
    logical_instance_quota = torch::from_blob(
        cpu_buffer + p2l_numel + l2p_numel + lcnts_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, cpu_opts);
    logical_instance_quota_prefix = torch::from_blob(cpu_buffer + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                                                     {num_layers, L, R},
                                                     {per_layer_stride_numel, R, 1},
                                                     cpu_opts);

    // Create GPU tensor views with strides
    auto device_opts =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    physical_to_logical_map_device =
        torch::from_blob(device_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, device_opts);
    logical_to_physical_map_device =
        torch::from_blob(device_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, device_opts);
    logical_replica_counts_device = torch::from_blob(
        device_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, device_opts);
    logical_instance_quota_device = torch::from_blob(device_buffer + p2l_numel + l2p_numel + lcnts_numel,
                                                     {num_layers, L, R},
                                                     {per_layer_stride_numel, R, 1},
                                                     device_opts);
    logical_instance_quota_prefix_device = torch::from_blob(
        device_buffer + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
        {num_layers, L, R},
        {per_layer_stride_numel, R, 1},
        device_opts);
    rank_quota_prefix = torch::zeros({num_layers, L, R}, device_opts);
}

void GlobalExpertPlacement::cleanup() {
    if (cpu_buffer != nullptr) {
        cudaFreeHost(cpu_buffer);
        cpu_buffer = nullptr;
        quota_buf_cpu = nullptr;
    }
    if (device_buffer != nullptr) {
        cudaFree(device_buffer);
        device_buffer = nullptr;
        quota_buf_device = nullptr;
    }
    rank_quota_prefix = torch::Tensor();
}

void GlobalExpertPlacement::to_device(const int layer_id, const bool async, std::optional<at::cuda::CUDAStream> s) const {
    EP_HOST_ASSERT(cpu_buffer != nullptr && device_buffer != nullptr);
    auto stream = s.value_or(at::cuda::getCurrentCUDAStream());
    if (layer_id >= 0) {
        EP_HOST_ASSERT(layer_id < num_layers_);
        int32_t* src = cpu_buffer + layer_id * per_layer_stride_numel;
        int32_t* dst = device_buffer + layer_id * per_layer_stride_numel;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, per_layer_data_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Sync all layers
        int32_t* src = cpu_buffer;
        int32_t* dst = device_buffer;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyHostToDevice, stream));
    }

    if (!async) {
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
}

void GlobalExpertPlacement::to_cpu(const int layer_id, const bool async, std::optional<at::cuda::CUDAStream> s) const {
    EP_HOST_ASSERT(cpu_buffer != nullptr && device_buffer != nullptr);
    auto stream = s.value_or(at::cuda::getCurrentCUDAStream());
    if (layer_id >= 0) {
        EP_HOST_ASSERT(layer_id < num_layers_);
        int32_t* src = device_buffer + layer_id * per_layer_stride_numel;
        int32_t* dst = cpu_buffer + layer_id * per_layer_stride_numel;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, per_layer_data_bytes, cudaMemcpyDeviceToHost, stream));
    } else {
        // Sync all layers
        int32_t* src = device_buffer;
        int32_t* dst = cpu_buffer;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyDeviceToHost, stream));
    }

    if (!async) {
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_cpu_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = cpu_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(base, base + p2l_numel, base + p2l_numel + l2p_numel);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_device_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = device_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(base, base + p2l_numel, base + p2l_numel + l2p_numel);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_quota_cpu_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = cpu_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(
        base + p2l_numel + l2p_numel + lcnts_numel, base + p2l_numel + l2p_numel + lcnts_numel + quota_numel, nullptr);
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
                 const int& weight_sync_plan_mode,
                 const int& weight_sync_relay_min_replicas,
                 const int& weight_sync_relay_max_relays,
                 const int& weight_sync_relay_min_fanout_gain)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
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
      weight_sync_plan_mode_(weight_sync_plan_mode),
      weight_sync_relay_min_replicas_(weight_sync_relay_min_replicas),
      weight_sync_relay_max_relays_(weight_sync_relay_max_relays),
      weight_sync_relay_min_fanout_gain_(weight_sync_relay_min_fanout_gain),
      balance_threshold_(balance_threshold),
      placement_cpu_dirty_(num_layers, false),
      comm_stream(at::cuda::getStreamFromPool(true)),
      relay_stream(at::cuda::getStreamFromPool(true)),
      placement_ready_events_(is_train ? num_layers : 1),
      placement_ready_stream_ids_(is_train ? num_layers : 1, -1)

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
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
        (kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc1_numel)) +
         kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc2_numel)));

    // Allocate global placement tensors using contiguous per-layer buffers on CPU and GPU.
    // This reduces number of H2D/D2H memory copies.
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.init(num_layers,
                   num_global_physical_experts,
                   num_global_logical_experts,
                   num_ranks,  // max_replicas_dim = num_ranks
                   device_id);

    // Allocate local replica weight buffer via NVSHMEM symmetric heap
    // This enables automatic cross-GPU access within NVL domain
    int64_t local_replica_weight_bytes =
        static_cast<int64_t>(num_local_redundant_experts) * expert_total_numel * kernels::kWeightElementBytes;

    local_replica_weight_buffer = nvshmem::alloc(local_replica_weight_bytes, kernels::kNvshmemAlignment);
    EP_HOST_ASSERT(local_replica_weight_buffer != nullptr && "Failed to allocate NVSHMEM weight buffer");

    local_replica_weight_buffer_tensor = make_tensor_from_buffer(local_replica_weight_buffer,
                                                                 {num_local_redundant_experts, expert_total_numel},
                                                                 torch::kBFloat16,
                                                                 torch::Device(torch::kCUDA, device_id));

    const int max_relay_chunks_per_shard =
        kernels::weight_sync_num_chunks(static_cast<size_t>(std::max(expert_fc1_numel, expert_fc2_numel)));
    const int64_t local_ready_flag_count =
        static_cast<int64_t>(num_local_redundant_experts) * 2 * max_relay_chunks_per_shard;
    local_weight_sync_ready_flags =
        reinterpret_cast<uint64_t*>(
            nvshmem::alloc(local_ready_flag_count * sizeof(uint64_t), kernels::kNvshmemAlignment));
    EP_HOST_ASSERT(local_weight_sync_ready_flags != nullptr &&
                   "Failed to allocate NVSHMEM ready-flag buffer for relay weight sync");
    if (local_ready_flag_count > 0) {
        CUDA_RUNTIME_CHECK(cudaMemset(local_weight_sync_ready_flags, 0, local_ready_flag_count * sizeof(uint64_t)));
    }

    // Grad buffer only needed for training
    if (is_train) {
        int64_t local_replica_grad_bytes =
            static_cast<int64_t>(num_local_redundant_experts) * expert_total_numel * kernels::kGradElementBytes;
        local_replica_grad_buffer = nvshmem::alloc(local_replica_grad_bytes, kernels::kNvshmemAlignment);
        EP_HOST_ASSERT(local_replica_grad_buffer != nullptr && "Failed to allocate NVSHMEM grad buffer");
        local_replica_grad_buffer_tensor = make_tensor_from_buffer(local_replica_grad_buffer,
                                                                   {num_local_redundant_experts, expert_total_numel},
                                                                   torch::kFloat32,
                                                                   torch::Device(torch::kCUDA, device_id));
        local_replica_grad_buffer_tensor.zero_();
    }

    // Synchronize all PEs to ensure buffers are allocated on all ranks
    nvshmem::barrier(true);

    // Obtain remote pointers via nvshmem_ptr() for all NVL ranks
    int num_nvl_ranks = runtime::num_nvl_ranks;
    int rdma_rank_idx = runtime::rdma_rank_idx;
    for (int i = 0; i < num_nvl_ranks; ++i) {
        int target_rank = rdma_rank_idx * num_nvl_ranks + i;
        global_replica_weight_buffer_ptrs[i] = nvshmem::ptr(local_replica_weight_buffer, target_rank);
        EP_HOST_ASSERT(global_replica_weight_buffer_ptrs[i] != nullptr &&
                       "nvshmem_ptr failed for weight buffer - target PE may not be in same NVL domain");
        global_weight_sync_ready_flag_ptrs[i] =
            reinterpret_cast<uint64_t*>(nvshmem::ptr(local_weight_sync_ready_flags, target_rank));
        EP_HOST_ASSERT(global_weight_sync_ready_flag_ptrs[i] != nullptr &&
                       "nvshmem_ptr failed for ready-flag buffer - target PE may not be in same NVL domain");
        if (is_train) {
            global_replica_grad_buffer_ptrs[i] = nvshmem::ptr(local_replica_grad_buffer, target_rank);
            EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[i] != nullptr &&
                           "nvshmem_ptr failed for grad buffer - target PE may not be in same NVL domain");
        }
    }

    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_remote_ready_flag_ptrs, kernels::kMaxNvlDomainSize * sizeof(uint64_t*)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(_remote_ready_flag_ptrs,
                                  global_weight_sync_ready_flag_ptrs,
                                  kernels::kMaxNvlDomainSize * sizeof(uint64_t*),
                                  cudaMemcpyHostToDevice));

    // Allocate intermediate buffers for task-build and persistent kernels.
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_grad_reduce_tasks, kernels::kMaxGradReduceTaskCount * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_global_task_or_tile_counter, sizeof(int)));
    const int shared_task_capacity =
        std::max(kernels::kMaxGradReduceTaskCount, _weight_sync_task_capacity);
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_tile_offsets, (shared_task_capacity + 1) * sizeof(int)));

    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_weight_sync_task_remaining_tiles, _weight_sync_task_capacity * sizeof(int)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_relay_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_task_tile_offsets, (_weight_sync_task_capacity + 1) * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_task_metadata, 2 * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_relay_global_tile_counter, sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_metadata, 2 * sizeof(int)));

    kernels::TaskBuildConfig config_cpu = {};
    config_cpu.rank_idx = runtime::rank_idx;
    config_cpu.nvl_rank_idx = runtime::nvl_rank_idx;
    config_cpu.num_nvl_ranks = runtime::num_nvl_ranks;
    config_cpu.num_local_master_experts = num_local_master_experts;
    config_cpu.num_local_physical_experts = num_local_physical_experts;
    config_cpu.num_local_redundant_experts = num_local_redundant_experts;
    config_cpu.expert_fc1_numel = expert_fc1_numel;
    config_cpu.expert_fc2_numel = expert_fc2_numel;
    config_cpu.expert_total_numel = expert_total_numel;
    config_cpu.max_replicas_dim = runtime::num_ranks;
    config_cpu.weight_sync_plan_mode = weight_sync_plan_mode_;
    config_cpu.weight_sync_relay_min_replicas = weight_sync_relay_min_replicas_;
    config_cpu.weight_sync_relay_max_relays = weight_sync_relay_max_relays_;
    config_cpu.weight_sync_relay_min_fanout_gain = weight_sync_relay_min_fanout_gain_;
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_build_config, sizeof(kernels::TaskBuildConfig)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(
        _task_build_config, &config_cpu, sizeof(kernels::TaskBuildConfig), cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_remote_weight_ptrs, kernels::kMaxNvlDomainSize * sizeof(void*)));
    CUDA_RUNTIME_CHECK(cudaMemcpy(_remote_weight_ptrs,
                                  global_replica_weight_buffer_ptrs,
                                  kernels::kMaxNvlDomainSize * sizeof(void*),
                                  cudaMemcpyHostToDevice));
    if (is_train) {
        CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_remote_grad_ptrs, kernels::kMaxNvlDomainSize * sizeof(void*)));
        CUDA_RUNTIME_CHECK(cudaMemcpy(_remote_grad_ptrs,
                                      global_replica_grad_buffer_ptrs,
                                      kernels::kMaxNvlDomainSize * sizeof(void*),
                                      cudaMemcpyHostToDevice));
    }

    const int max_stage_tiles_per_expert = kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc1_numel)) +
        kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc2_numel));
    _max_ws_total_tiles = num_local_physical_experts * max_stage_tiles_per_expert;

    const int64_t max_fc_numel = std::max(expert_fc1_numel, expert_fc2_numel);
    const int max_replicas = runtime::num_nvl_ranks - 1;
    _max_gr_total_tasks = 2 * num_local_master_experts * max_replicas;
    const int gr_tiles_per_task = static_cast<int>(
        (max_fc_numel + kernels::kGradReduceTileElements - 1) / kernels::kGradReduceTileElements);
    _max_gr_total_tiles = _max_gr_total_tasks * gr_tiles_per_task;

    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_reroute_sparse_counters, num_global_logical_experts * sizeof(int)));

    global_logical_expert_loads =
        reinterpret_cast<int*>(nvshmem::alloc(num_global_logical_experts * sizeof(int), kernels::kNvshmemAlignment));
    local_expert_loads = reinterpret_cast<int32_t*>(
        nvshmem::alloc(num_global_logical_experts * sizeof(int32_t), kernels::kNvshmemAlignment));
    expert_loads_per_rank = reinterpret_cast<int32_t*>(nvshmem::alloc(
        static_cast<size_t>(runtime::num_ranks) * num_global_logical_experts * sizeof(int32_t),
        kernels::kNvshmemAlignment));
    if (legacy_placement_) {
        CUDA_RUNTIME_CHECK(
            cudaMallocHost((void**)&global_logical_expert_loads_cpu, num_global_logical_experts * sizeof(int)));
    }

    // Initialize default placement (master-only) for all layers so sparse reroute
    // remains valid before the first placement update.
    for (int lid = 0; lid < num_layers; ++lid) {
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(lid);
        for (int logical_id = 0; logical_id < num_global_logical_experts; ++logical_id) {
            const int rank = logical_id / num_local_master_experts;
            const int local_idx = logical_id % num_local_master_experts;
            const int physical_id = rank * num_local_physical_experts + local_idx;
            p2l_ptr[physical_id] = logical_id;
            l2p_ptr[logical_id * runtime::num_ranks] = physical_id;
            lcnts_ptr[logical_id] = 1;
        }
    }
    placement.to_device(-1, false);  // sync copy all layers

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
    nvshmem::free(local_replica_weight_buffer);
    local_replica_weight_buffer = nullptr;
    if (local_weight_sync_ready_flags != nullptr) {
        nvshmem::free(local_weight_sync_ready_flags);
        local_weight_sync_ready_flags = nullptr;
    }
    if (local_replica_grad_buffer != nullptr) {
        nvshmem::free(local_replica_grad_buffer);
        local_replica_grad_buffer = nullptr;
    }
    nvshmem::free(global_logical_expert_loads);
    global_logical_expert_loads = nullptr;
    if (local_expert_loads != nullptr) {
        nvshmem::free(local_expert_loads);
        local_expert_loads = nullptr;
    }
    if (expert_loads_per_rank != nullptr) {
        nvshmem::free(expert_loads_per_rank);
        expert_loads_per_rank = nullptr;
    }

    // Clear remote pointers
    for (int i = 0; i < runtime::num_nvl_ranks; ++i) {
        global_replica_weight_buffer_ptrs[i] = nullptr;
        global_replica_grad_buffer_ptrs[i] = nullptr;
        global_weight_sync_ready_flag_ptrs[i] = nullptr;
    }

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

    // Free expert load buffers
    if (global_logical_expert_loads_cpu != nullptr) {
        CUDA_RUNTIME_CHECK(cudaFreeHost(global_logical_expert_loads_cpu));
        global_logical_expert_loads_cpu = nullptr;
    }

    // Free contiguous placement buffers (CPU pinned + GPU)
    placement.cleanup();

    // Free NVSHMEM runtime
    runtime::destroy();

    // Ready to destroy
    _available = false;
}

void Manager::sync_placement_to_cpu(const int layer_id) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= -1 && layer_id < num_layers);

    bool need_sync = false;
    if (layer_id >= 0) {
        need_sync = placement_cpu_dirty_[layer_id];
    } else {
        for (bool dirty : placement_cpu_dirty_) {
            if (dirty) {
                need_sync = true;
                break;
            }
        }
    }
    if (!need_sync) {
        return;
    }

    // Placement writes may have been enqueued on either the caller's current stream
    // or comm_stream. For CPU consumers we can take the slow path and fully
    // synchronize the device before refreshing the host mirror.
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    placement.to_cpu(layer_id, /*async=*/false);

    if (layer_id >= 0) {
        placement_cpu_dirty_[layer_id] = false;
    } else {
        std::fill(placement_cpu_dirty_.begin(), placement_cpu_dirty_.end(), false);
    }
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

void Manager::update_placement(const int& layer_id, torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(routing_map.dim() == 2 && routing_map.size(1) == num_global_logical_experts &&
                   routing_map.dtype() == torch::kBool);

    auto curr_stream = at::cuda::getCurrentCUDAStream();

    kernels::rmap_local_sum(routing_map.size(0),
                            num_global_logical_experts,
                            routing_map.data_ptr<bool>(),
                            global_logical_expert_loads,
                            curr_stream.stream());

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);

    if (legacy_placement_) {
        nvshmem::int32_allreduce(global_logical_expert_loads, num_global_logical_experts, curr_stream.stream());
        kernels::legacy::solve_placement(global_logical_expert_loads,
                                       nullptr,
                                       physical_to_logical_map,
                                       logical_to_physical_map,
                                       logical_replica_counts,
                                       logical_instance_quota,
                                       logical_instance_quota_prefix,
                                       rank_quota_prefix,
                                       curr_stream.stream(),
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
    } else {
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(local_expert_loads,
                                           global_logical_expert_loads,
                                           num_global_logical_experts * sizeof(int32_t),
                                           cudaMemcpyDeviceToDevice,
                                           curr_stream.stream()));
        nvshmem::int32_fcollect(
            expert_loads_per_rank, local_expert_loads, num_global_logical_experts, curr_stream.stream());
        kernels::reduce_per_rank_loads(expert_loads_per_rank,
                                       global_logical_expert_loads,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       curr_stream.stream());
        kernels::solve_placement(global_logical_expert_loads,
                               expert_loads_per_rank,
                               physical_to_logical_map,
                               logical_to_physical_map,
                               logical_replica_counts,
                               logical_instance_quota,
                               logical_instance_quota_prefix,
                               rank_quota_prefix,
                               curr_stream.stream(),
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
    }
    placement_cpu_dirty_[layer_id] = true;
    record_placement_ready(layer_id, curr_stream);
}

void Manager::update_placement_sparse(const int& layer_id, torch::Tensor& topk_ids) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(topk_ids.is_cuda() && topk_ids.dtype() == torch::kInt64);
    EP_HOST_ASSERT(topk_ids.dim() == 2);

    int T = topk_ids.size(0);
    int K = topk_ids.size(1);

    // Use comm_stream for histogram + allreduce (same pattern as update_placement)
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    stream_wait(comm_stream, compute_stream);

    kernels::topk_local_sum(topk_ids.data_ptr<int64_t>(),
                            T,
                            K,
                            num_global_logical_experts,
                            global_logical_expert_loads,
                            comm_stream.stream());

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);

    if (legacy_placement_) {
        nvshmem::int32_allreduce(global_logical_expert_loads, num_global_logical_experts, comm_stream.stream());
        kernels::legacy::solve_placement(global_logical_expert_loads,
                                       nullptr,
                                       physical_to_logical_map,
                                       logical_to_physical_map,
                                       logical_replica_counts,
                                       logical_instance_quota,
                                       logical_instance_quota_prefix,
                                       rank_quota_prefix,
                                       comm_stream.stream(),
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
    } else {
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(local_expert_loads,
                                           global_logical_expert_loads,
                                           num_global_logical_experts * sizeof(int32_t),
                                           cudaMemcpyDeviceToDevice,
                                           comm_stream.stream()));
        nvshmem::int32_fcollect(
            expert_loads_per_rank, local_expert_loads, num_global_logical_experts, comm_stream.stream());
        kernels::reduce_per_rank_loads(expert_loads_per_rank,
                                       global_logical_expert_loads,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       comm_stream.stream());
        kernels::solve_placement(global_logical_expert_loads,
                               expert_loads_per_rank,
                               physical_to_logical_map,
                               logical_to_physical_map,
                               logical_replica_counts,
                               logical_instance_quota,
                               logical_instance_quota_prefix,
                               rank_quota_prefix,
                               comm_stream.stream(),
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
    }
    placement_cpu_dirty_[layer_id] = true;
    record_placement_ready(layer_id, comm_stream);
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

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
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

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;

    if (T > 0 && L > 0) {
        constexpr int TILE_T = kernels::kDenseRerouteTileTokens;
        const int num_tiles = (T + TILE_T - 1) / TILE_T;
        
        int64_t tile_count_numel = static_cast<int64_t>(L) * num_tiles;
        torch::Tensor tile_counts_tensor = torch::empty({tile_count_numel}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int32_t* tile_counts_ptr = tile_counts_tensor.data_ptr<int32_t>();

        EP_HOST_ASSERT(probs.scalar_type() == torch::kFloat32);

        if (!legacy_placement_) {
            auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
                placement.get_quota_ptrs(layer_id);
            (void)logical_instance_quota;
            (void)logical_instance_quota_prefix;
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
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
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

std::optional<EventHandle> Manager::grad_reduce(const int& layer_id,
                                                torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                                torch::Tensor& local_master_fc2_grad_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async) {
    EP_HOST_ASSERT(is_available());

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.numel() == num_local_master_experts);

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_grad_ptrs = local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_grad_ptrs = local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>();

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
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
                                     comm_stream);

    kernels::run_grad_reduce(
        _grad_reduce_tasks, _task_tile_offsets, _task_metadata, _global_task_or_tile_counter, comm_stream, grad_reduce_num_sms_);

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

std::optional<EventHandle> Manager::weight_sync(const int& layer_id,
                                                torch::Tensor& local_master_fc1_weight_ptr_tensor,
                                                torch::Tensor& local_master_fc2_weight_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async) {
    EP_HOST_ASSERT(is_available());

    auto compute_stream = at::cuda::getCurrentCUDAStream();
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
        stream_wait(relay_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
        stream_wait(relay_stream, compute_stream);
    }

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.numel() == num_local_master_experts);
    const bool enable_relay_stages = weight_sync_plan_mode_ != static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
        num_local_redundant_experts > 0 && runtime::num_nvl_ranks > 2;
    const uint64_t current_epoch = ++_weight_sync_epoch;
    bool launched_stage2 = false;

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_weight_ptrs = local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_weight_ptrs = local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>();

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] = placement.get_device_ptrs(layer_id);
    kernels::build_weight_sync_task_lists(_task_build_config,
                                          physical_to_logical_map,
                                          logical_to_physical_map,
                                          logical_replica_counts,
                                          _remote_weight_ptrs,
                                          local_master_fc1_weight_ptrs,
                                          local_master_fc2_weight_ptrs,
                                          reinterpret_cast<__nv_bfloat16*>(local_replica_weight_buffer),
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
    EventHandle task_build_ready(comm_stream);

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
                             _max_ws_total_tiles,
                             2);

    if (enable_relay_stages) {
        stream_wait(relay_stream, task_build_ready);
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
                                 _max_ws_total_tiles,
                                 1);
        launched_stage2 = true;
    }

    if (launched_stage2) {
        stream_wait(comm_stream, relay_stream);
    }

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

}  // namespace ultra_ep
