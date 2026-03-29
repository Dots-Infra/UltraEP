#include "ultra_ep.hpp"

#include <cuda_bf16.h>

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
    CUDA_RUNTIME_CHECK(cudaMalloc(&gpu_buffer, total_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(gpu_buffer, 0, total_bytes));
    quota_buf_gpu = gpu_buffer + p2l_numel + l2p_numel + lcnts_numel;

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
    auto gpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    physical_to_logical_map_gpu = torch::from_blob(gpu_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, gpu_opts);
    logical_to_physical_map_gpu =
        torch::from_blob(gpu_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, gpu_opts);
    logical_replica_counts_gpu =
        torch::from_blob(gpu_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, gpu_opts);
    logical_instance_quota_gpu = torch::from_blob(
        gpu_buffer + p2l_numel + l2p_numel + lcnts_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, gpu_opts);
    logical_instance_quota_prefix_gpu = torch::from_blob(gpu_buffer + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                                                         {num_layers, L, R},
                                                         {per_layer_stride_numel, R, 1},
                                                         gpu_opts);
    rank_quota_prefix_gpu = torch::zeros({num_layers, L, R}, gpu_opts);
}

void GlobalExpertPlacement::cleanup() {
    if (cpu_buffer != nullptr) {
        cudaFreeHost(cpu_buffer);
        cpu_buffer = nullptr;
        quota_buf_cpu = nullptr;
    }
    if (gpu_buffer != nullptr) {
        cudaFree(gpu_buffer);
        gpu_buffer = nullptr;
        quota_buf_gpu = nullptr;
    }
    rank_quota_prefix_gpu = torch::Tensor();
}

void GlobalExpertPlacement::to_gpu(const int layer_id, const bool async, std::optional<at::cuda::CUDAStream> s) const {
    EP_HOST_ASSERT(cpu_buffer != nullptr && gpu_buffer != nullptr);
    auto stream = s.value_or(at::cuda::getCurrentCUDAStream());
    if (layer_id >= 0) {
        EP_HOST_ASSERT(layer_id < num_layers_);
        int32_t* src = cpu_buffer + layer_id * per_layer_stride_numel;
        int32_t* dst = gpu_buffer + layer_id * per_layer_stride_numel;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, per_layer_data_bytes, cudaMemcpyHostToDevice, stream));
    } else {
        // Sync all layers
        int32_t* src = cpu_buffer;
        int32_t* dst = gpu_buffer;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyHostToDevice, stream));
    }

    if (!async) {
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    }
}

void GlobalExpertPlacement::to_cpu(const int layer_id, const bool async, std::optional<at::cuda::CUDAStream> s) const {
    EP_HOST_ASSERT(cpu_buffer != nullptr && gpu_buffer != nullptr);
    auto stream = s.value_or(at::cuda::getCurrentCUDAStream());
    if (layer_id >= 0) {
        EP_HOST_ASSERT(layer_id < num_layers_);
        int32_t* src = gpu_buffer + layer_id * per_layer_stride_numel;
        int32_t* dst = cpu_buffer + layer_id * per_layer_stride_numel;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, per_layer_data_bytes, cudaMemcpyDeviceToHost, stream));
    } else {
        // Sync all layers
        int32_t* src = gpu_buffer;
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

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_gpu_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = gpu_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(base, base + p2l_numel, base + p2l_numel + l2p_numel);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_quota_cpu_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = cpu_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(
        base + p2l_numel + l2p_numel + lcnts_numel, base + p2l_numel + l2p_numel + lcnts_numel + quota_numel, nullptr);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_quota_gpu_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = gpu_buffer + layer_id * per_layer_stride_numel;
    int32_t* rank_quota_base =
        rank_quota_prefix_gpu.data_ptr<int32_t>() + static_cast<int64_t>(layer_id) * rank_quota_numel;
    return std::make_tuple(base + p2l_numel + l2p_numel + lcnts_numel,
                           base + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                           rank_quota_base);
}

// ============================================================================
// RerouteOutputBuffer
// ============================================================================

RerouteOutputBuffer::RerouteOutputBuffer(const int num_layers,
                                         const int num_global_logical_experts,
                                         const int num_global_physical_experts,
                                         const bool is_train)
    : num_layers_(num_layers),
      num_global_logical_experts_(num_global_logical_experts),
      num_global_physical_experts_(num_global_physical_experts),
      is_train_(is_train) {
    if (is_train_) {
        reroute_layer_valid_flags_.resize(num_layers_, false);
    }
}

std::tuple<void*, bool*> RerouteOutputBuffer::get_or_create_fwd_bufs(const int num_tokens,
                                                                     const int layer_id,
                                                                     const torch::ScalarType probs_dtype) {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int T = num_tokens;
    int P = num_global_physical_experts_;
    int L = num_global_logical_experts_;
    auto device = torch::kCUDA;
    void* expand_probs_ptr = nullptr;
    bool* expand_rmap_ptr = nullptr;

    if (is_train_) {
        if (!reroute_expand_probs_buf_.defined() || !reroute_expand_rmap_buf_.defined()) {
            reroute_expand_probs_buf_ =
                torch::zeros({num_layers_, T, P}, torch::TensorOptions().dtype(probs_dtype).device(device));
            reroute_expand_rmap_buf_ =
                torch::zeros({num_layers_, T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
            std::fill(reroute_layer_valid_flags_.begin(), reroute_layer_valid_flags_.end(), true);
            reroute_expand_probs_nbytes_per_layer_ = T * P * reroute_expand_probs_buf_.element_size();
            reroute_expand_rmap_nbytes_per_layer_ = T * P * sizeof(bool);
        }
        expand_probs_ptr = reinterpret_cast<void*>((int8_t*)reroute_expand_probs_buf_.data_ptr() +
                                                   layer_id * reroute_expand_probs_nbytes_per_layer_);
        expand_rmap_ptr = reinterpret_cast<bool*>((int8_t*)reroute_expand_rmap_buf_.data_ptr() +
                                                  layer_id * reroute_expand_rmap_nbytes_per_layer_);
    } else {  // inference, use shared output buffer
        if (!reroute_expand_probs_buf_.defined() || reroute_expand_probs_buf_.size(0) != T) {
            reroute_expand_probs_buf_ = torch::zeros({T, P}, torch::TensorOptions().dtype(probs_dtype).device(device));
            reroute_expand_rmap_buf_ = torch::zeros({T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
            reroute_expand_probs_nbytes_per_layer_ = T * P * reroute_expand_probs_buf_.element_size();
            reroute_expand_rmap_nbytes_per_layer_ = T * P * sizeof(bool);
        }
        expand_probs_ptr = reinterpret_cast<void*>(reroute_expand_probs_buf_.data_ptr());
        expand_rmap_ptr = reinterpret_cast<bool*>(reroute_expand_rmap_buf_.data_ptr());
    }
    return std::make_tuple(expand_probs_ptr, expand_rmap_ptr);
}

void* RerouteOutputBuffer::get_or_create_bwd_buf(const int num_tokens, const torch::ScalarType probs_dtype) {
    EP_HOST_ASSERT(is_train_);
    int T = num_tokens;
    int L = num_global_logical_experts_;
    auto device = torch::kCUDA;
    if (!reroute_grad_probs_buf_.defined()) {
        reroute_grad_probs_buf_ = torch::zeros({T, L}, torch::TensorOptions().dtype(probs_dtype).device(device));
        reroute_bwd_valid_flag_ = true;
    }
    return reroute_grad_probs_buf_.data_ptr();
}

void RerouteOutputBuffer::zero_out_fwd_bufs(const int layer_id, at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    if (reroute_expand_probs_buf_.defined() && reroute_expand_rmap_buf_.defined()) {
        if (is_train_) {
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(reinterpret_cast<int8_t*>(reroute_expand_probs_buf_.data_ptr()) +
                                                   layer_id * reroute_expand_probs_nbytes_per_layer_,
                                               0,
                                               reroute_expand_probs_nbytes_per_layer_,
                                               stream));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(reinterpret_cast<int8_t*>(reroute_expand_rmap_buf_.data_ptr()) +
                                                   layer_id * reroute_expand_rmap_nbytes_per_layer_,
                                               0,
                                               reroute_expand_rmap_nbytes_per_layer_,
                                               stream));
            reroute_layer_valid_flags_[layer_id] = true;
        } else {
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(
                reroute_expand_probs_buf_.data_ptr(), 0, reroute_expand_probs_nbytes_per_layer_, stream));
            CUDA_RUNTIME_CHECK(
                cudaMemsetAsync(reroute_expand_rmap_buf_.data_ptr(), 0, reroute_expand_rmap_nbytes_per_layer_, stream));
            reroute_inf_valid_flag_ = true;
        }
    }
}

void RerouteOutputBuffer::zero_out_bwd_buf(at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(is_train_);
    if (reroute_grad_probs_buf_.defined()) {
        CUDA_RUNTIME_CHECK(
            cudaMemsetAsync(reroute_grad_probs_buf_.data_ptr(), 0, reroute_grad_probs_buf_.nbytes(), stream));
        reroute_bwd_valid_flag_ = true;
    }
}

int32_t* RerouteOutputBuffer::get_or_create_tile_counts(const int L, const int num_tiles) {
    int64_t needed = static_cast<int64_t>(L) * num_tiles;
    if (!reroute_tile_counts_buf_.defined() || reroute_tile_counts_buf_.numel() < needed) {
        reroute_tile_counts_buf_ =
            torch::empty({needed}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
    return reroute_tile_counts_buf_.data_ptr<int32_t>();
}

const bool* RerouteOutputBuffer::get_fwd_expanded_rmap_ptr(const int layer_id) const {
    EP_HOST_ASSERT(reroute_expand_rmap_buf_.defined());
    if (is_train_) {
        EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
        return reinterpret_cast<const bool*>(reinterpret_cast<const int8_t*>(reroute_expand_rmap_buf_.data_ptr()) +
                                             layer_id * reroute_expand_rmap_nbytes_per_layer_);
    } else {
        return reinterpret_cast<const bool*>(reroute_expand_rmap_buf_.data_ptr());
    }
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
                 const bool& use_gpu_solver,
                 const float& balance_threshold,
                 const bool& use_quota_solver,
                 const bool& quota_locality_aware,
                 const int32_t& quota_min_tokens_per_replica,
                 const bool& quota_allow_zero_master_quota,
                 const float& quota_oracle_eps)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      is_train(is_train),
      explicitly_destroy(explicitly_destroy),
      use_gpu_solver_(use_gpu_solver),
      use_quota_solver_(use_quota_solver),
      quota_locality_aware_(quota_locality_aware),
      quota_min_tokens_per_replica_(quota_min_tokens_per_replica),
      quota_allow_zero_master_quota_(quota_allow_zero_master_quota),
      quota_oracle_eps_(quota_oracle_eps),
      balance_threshold_(balance_threshold),
      placement_cpu_dirty_(num_layers, false),
      comm_stream(at::cuda::getStreamFromPool(true)),
      placement_ready_events_(is_train ? num_layers : 1),
      placement_ready_stream_ids_(is_train ? num_layers : 1, -1)

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    EP_HOST_ASSERT(
        !(use_quota_solver_ && use_gpu_solver_) &&
        "use_quota_solver and use_gpu_solver are mutually exclusive; quota solver runs on GPU independently");
    num_global_physical_experts = num_local_physical_experts * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;

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
        (int64_t)num_local_redundant_experts * expert_total_numel * WEIGHT_ELEMENT_SIZE;

    local_replica_weight_buffer = nvshmem::alloc(local_replica_weight_bytes, NVSHMEM_ALIGNMENT);
    EP_HOST_ASSERT(local_replica_weight_buffer != nullptr && "Failed to allocate NVSHMEM weight buffer");

    local_replica_weight_buffer_tensor = make_tensor_from_buffer(local_replica_weight_buffer,
                                                                 {num_local_redundant_experts, expert_total_numel},
                                                                 torch::kBFloat16,
                                                                 torch::Device(torch::kCUDA, device_id));

    // Grad buffer only needed for training
    if (is_train) {
        int64_t local_replica_grad_bytes =
            (int64_t)num_local_redundant_experts * expert_total_numel * GRAD_ELEMENT_SIZE;
        local_replica_grad_buffer = nvshmem::alloc(local_replica_grad_bytes, NVSHMEM_ALIGNMENT);
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
        if (is_train) {
            global_replica_grad_buffer_ptrs[i] = nvshmem::ptr(local_replica_grad_buffer, target_rank);
            EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[i] != nullptr &&
                           "nvshmem_ptr failed for grad buffer - target PE may not be in same NVL domain");
        }
    }

    // Allocate intermediate buffers for grad reduce tasks (regular CUDA memory)
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&_grad_reduce_tasks_cpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_grad_reduce_tasks_gpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_global_task_or_tile_counter_gpu, sizeof(int)));
    // +1 for the final offset (total tile count)
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_tile_offsets_gpu, (MAX_GRAD_REDUCE_TASK_NUM + 1) * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMallocHost((void**)&_task_tile_offsets_cpu, 3 * num_local_master_experts * sizeof(int)));

    // Allocate intermediate buffers for weight sync tasks
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&_weight_sync_tasks_cpu, MAX_WEIGHT_SYNC_TASK_NUM * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_weight_sync_tasks_gpu, MAX_WEIGHT_SYNC_TASK_NUM * sizeof(kernels::WeightSyncTask)));

    // Task metadata buffer: [total_tasks, total_tiles] — shared between weight_sync and grad_reduce
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_metadata_gpu, 2 * sizeof(int)));

    // GPU task build support for modes that keep placement on device.
    // Quota solver is also GPU-resident, so it should share this path.
    if (use_gpu_solver_ || use_quota_solver_) {
        // Copy immutable config to GPU (once)
        kernels::TaskBuildConfig config_cpu;
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
        CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_build_config_gpu, sizeof(kernels::TaskBuildConfig)));
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(_task_build_config_gpu, &config_cpu, sizeof(kernels::TaskBuildConfig), cudaMemcpyHostToDevice));

        // Copy remote pointer tables to GPU (one-time, immutable after constructor)
        CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_remote_weight_ptrs_gpu, MAX_NVL_DOMAIN_SIZE * sizeof(void*)));
        CUDA_RUNTIME_CHECK(cudaMemcpy(_remote_weight_ptrs_gpu,
                                      global_replica_weight_buffer_ptrs,
                                      MAX_NVL_DOMAIN_SIZE * sizeof(void*),
                                      cudaMemcpyHostToDevice));
        if (is_train) {
            CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_remote_grad_ptrs_gpu, MAX_NVL_DOMAIN_SIZE * sizeof(void*)));
            CUDA_RUNTIME_CHECK(cudaMemcpy(_remote_grad_ptrs_gpu,
                                          global_replica_grad_buffer_ptrs,
                                          MAX_NVL_DOMAIN_SIZE * sizeof(void*),
                                          cudaMemcpyHostToDevice));
        }

        // Staging buffer for master pointers (H2D copied each call, ~64 bytes)
        CUDA_RUNTIME_CHECK(
            cudaMalloc((void**)&_local_master_ptrs_staging_gpu, 2 * num_local_master_experts * sizeof(int64_t)));

        // Pre-compute upper bounds for GPU-path grid sizing
        int64_t max_fc_numel = std::max(expert_fc1_numel, expert_fc2_numel);
        int ws_tiles_per_task =
            static_cast<int>((max_fc_numel + WEIGHT_SYNC_TILE_ELEMENTS - 1) / WEIGHT_SYNC_TILE_ELEMENTS);
        _max_ws_total_tiles = 2 * num_local_master_experts * ws_tiles_per_task;

        int max_replicas = runtime::num_nvl_ranks - 1;
        _max_gr_total_tasks = 2 * num_local_master_experts * max_replicas;
        int gr_tiles_per_task =
            static_cast<int>((max_fc_numel + GRAD_REDUCE_TILE_ELEMENTS - 1) / GRAD_REDUCE_TILE_ELEMENTS);
        _max_gr_total_tiles = _max_gr_total_tasks * gr_tiles_per_task;
    }

    // Sparse reroute counters
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_reroute_sparse_counters_gpu, num_global_logical_experts * sizeof(int)));

    // Create pre-allocated placement solver (zero-alloc on hot path)
    placement_solver_ = std::make_unique<solver::PlacementSolver>(num_global_logical_experts,
                                                                  runtime::num_ranks,
                                                                  num_local_master_experts,
                                                                  num_local_redundant_experts,
                                                                  runtime::num_nvl_ranks,
                                                                  runtime::num_ranks  // max_replicas_dim = num_ranks
    );
    // Optionally create GPU placement solver
    if (use_gpu_solver_) {
        placement_solver_gpu_ = std::make_unique<solver::PlacementSolverGPU>(num_global_logical_experts,
                                                                             runtime::num_ranks,
                                                                             num_local_master_experts,
                                                                             num_local_redundant_experts,
                                                                             runtime::num_nvl_ranks,
                                                                             runtime::num_ranks);
    }
    if (use_quota_solver_) {
        placement_solver_quota_ = std::make_unique<solver::PlacementSolverQuota>(num_global_logical_experts,
                                                                                 runtime::num_ranks,
                                                                                 num_local_master_experts,
                                                                                 num_local_redundant_experts,
                                                                                 runtime::num_nvl_ranks,
                                                                                 runtime::num_ranks);
    }
    // Allocate global logical expert load buffer
    global_logical_expert_loads_gpu =
        reinterpret_cast<int*>(nvshmem::alloc(num_global_logical_experts * sizeof(int), NVSHMEM_ALIGNMENT));
    if (use_quota_solver_) {
        local_expert_loads_gpu =
            reinterpret_cast<int32_t*>(nvshmem::alloc(num_global_logical_experts * sizeof(int32_t), NVSHMEM_ALIGNMENT));
        expert_loads_per_rank_gpu = reinterpret_cast<int32_t*>(nvshmem::alloc(
            static_cast<size_t>(runtime::num_ranks) * num_global_logical_experts * sizeof(int32_t), NVSHMEM_ALIGNMENT));
    }
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&global_logical_expert_loads_cpu, num_global_logical_experts * sizeof(int)));
    // Create pre-allocated reroute solver
    reroute_solver_ = std::make_unique<solver::RerouteSolver>(num_global_logical_experts,
                                                              num_global_physical_experts,
                                                              runtime::num_ranks  // max_replicas_dim = num_ranks
    );
    // Create pre-allocated reroute output buffer
    reroute_output_buffer_ = std::make_unique<RerouteOutputBuffer>(
        num_layers, num_global_logical_experts, num_global_physical_experts, is_train);

    // Initialize default placement (master-only) for all layers.
    // This ensures reroute_sparse works correctly even before the first update_placement call.
    std::memset(global_logical_expert_loads_cpu, 0, num_global_logical_experts * sizeof(int));
    for (int lid = 0; lid < num_layers; ++lid) {
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(lid);
        placement_solver_->solve(global_logical_expert_loads_cpu, p2l_ptr, l2p_ptr, lcnts_ptr);
    }
    placement.to_gpu(-1, false);  // sync copy all layers

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
    if (local_replica_grad_buffer != nullptr) {
        nvshmem::free(local_replica_grad_buffer);
        local_replica_grad_buffer = nullptr;
    }
    nvshmem::free(global_logical_expert_loads_gpu);
    global_logical_expert_loads_gpu = nullptr;
    if (local_expert_loads_gpu != nullptr) {
        nvshmem::free(local_expert_loads_gpu);
        local_expert_loads_gpu = nullptr;
    }
    if (expert_loads_per_rank_gpu != nullptr) {
        nvshmem::free(expert_loads_per_rank_gpu);
        expert_loads_per_rank_gpu = nullptr;
    }

    // Clear remote pointers
    for (int i = 0; i < runtime::num_nvl_ranks; ++i) {
        global_replica_weight_buffer_ptrs[i] = nullptr;
        global_replica_grad_buffer_ptrs[i] = nullptr;
    }

    // Free intermediate CUDA buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(_grad_reduce_tasks_cpu));
    CUDA_RUNTIME_CHECK(cudaFree(_grad_reduce_tasks_gpu));
    CUDA_RUNTIME_CHECK(cudaFree(_global_task_or_tile_counter_gpu));
    CUDA_RUNTIME_CHECK(cudaFree(_task_tile_offsets_gpu));
    CUDA_RUNTIME_CHECK(cudaFreeHost(_task_tile_offsets_cpu));
    _grad_reduce_tasks_cpu = nullptr;
    _grad_reduce_tasks_gpu = nullptr;
    _global_task_or_tile_counter_gpu = nullptr;
    _task_tile_offsets_gpu = nullptr;
    _task_tile_offsets_cpu = nullptr;

    // Free weight sync buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(_weight_sync_tasks_cpu));
    CUDA_RUNTIME_CHECK(cudaFree(_weight_sync_tasks_gpu));
    _weight_sync_tasks_cpu = nullptr;
    _weight_sync_tasks_gpu = nullptr;

    // Free task metadata buffer
    CUDA_RUNTIME_CHECK(cudaFree(_task_metadata_gpu));
    _task_metadata_gpu = nullptr;

    // Free GPU task build buffers
    if (_task_build_config_gpu) {
        CUDA_RUNTIME_CHECK(cudaFree(_task_build_config_gpu));
        _task_build_config_gpu = nullptr;
    }
    if (_remote_weight_ptrs_gpu) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_weight_ptrs_gpu));
        _remote_weight_ptrs_gpu = nullptr;
    }
    if (_remote_grad_ptrs_gpu) {
        CUDA_RUNTIME_CHECK(cudaFree(_remote_grad_ptrs_gpu));
        _remote_grad_ptrs_gpu = nullptr;
    }
    if (_local_master_ptrs_staging_gpu) {
        CUDA_RUNTIME_CHECK(cudaFree(_local_master_ptrs_staging_gpu));
        _local_master_ptrs_staging_gpu = nullptr;
    }

    // Free sparse reroute counters
    CUDA_RUNTIME_CHECK(cudaFree(_reroute_sparse_counters_gpu));
    _reroute_sparse_counters_gpu = nullptr;

    // Free expert load buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(global_logical_expert_loads_cpu));
    global_logical_expert_loads_cpu = nullptr;

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

    if (!use_gpu_solver_ && !use_quota_solver_) {
        return;
    }

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
                            global_logical_expert_loads_gpu,
                            curr_stream.stream());

    if (use_quota_solver_) {
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(local_expert_loads_gpu,
                                           global_logical_expert_loads_gpu,
                                           num_global_logical_experts * sizeof(int32_t),
                                           cudaMemcpyDeviceToDevice,
                                           curr_stream.stream()));
        nvshmem::int32_fcollect(
            expert_loads_per_rank_gpu, local_expert_loads_gpu, num_global_logical_experts, curr_stream.stream());
        kernels::reduce_per_rank_loads(expert_loads_per_rank_gpu,
                                       global_logical_expert_loads_gpu,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       curr_stream.stream());

        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        auto [quota_gpu, quota_prefix_gpu, rank_quota_prefix_gpu] = placement.get_quota_gpu_ptrs(layer_id);
        placement_solver_quota_->solve(global_logical_expert_loads_gpu,
                                       expert_loads_per_rank_gpu,
                                       p2l_gpu,
                                       l2p_gpu,
                                       lcnts_gpu,
                                       quota_gpu,
                                       quota_prefix_gpu,
                                       rank_quota_prefix_gpu,
                                       curr_stream.stream(),
                                       balance_threshold_,
                                       quota_min_tokens_per_replica_,
                                       quota_allow_zero_master_quota_,
                                       quota_locality_aware_,
                                       quota_oracle_eps_);
        placement_cpu_dirty_[layer_id] = true;
    } else if (use_gpu_solver_) {
        nvshmem::int32_allreduce(global_logical_expert_loads_gpu, num_global_logical_experts, curr_stream.stream());
        // GPU path: solver reads directly from GPU loads, writes to GPU placement buffer.
        // No D2H/sync/H2D on the hot path: the CPU mirror is refreshed only on demand.
        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        placement_solver_gpu_->solve(
            global_logical_expert_loads_gpu, p2l_gpu, l2p_gpu, lcnts_gpu, curr_stream.stream(), balance_threshold_);
        placement_cpu_dirty_[layer_id] = true;
    } else {
        nvshmem::int32_allreduce(global_logical_expert_loads_gpu, num_global_logical_experts, curr_stream.stream());
        // CPU path (original): D2H → sync → CPU solve → H2D
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(global_logical_expert_loads_cpu,
                                           global_logical_expert_loads_gpu,
                                           num_global_logical_experts * sizeof(int),
                                           cudaMemcpyDeviceToHost,
                                           curr_stream.stream()));

        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);

        // Ensure data readiness for CPU-side placement solver
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(curr_stream.stream()));

        placement_solver_->solve(global_logical_expert_loads_cpu, p2l_ptr, l2p_ptr, lcnts_ptr, balance_threshold_);

        // Move placement to GPU for later use
        placement.to_gpu(layer_id);
    }
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
                            global_logical_expert_loads_gpu,
                            comm_stream.stream());

    if (use_quota_solver_) {
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(local_expert_loads_gpu,
                                           global_logical_expert_loads_gpu,
                                           num_global_logical_experts * sizeof(int32_t),
                                           cudaMemcpyDeviceToDevice,
                                           comm_stream.stream()));
        nvshmem::int32_fcollect(
            expert_loads_per_rank_gpu, local_expert_loads_gpu, num_global_logical_experts, comm_stream.stream());
        kernels::reduce_per_rank_loads(expert_loads_per_rank_gpu,
                                       global_logical_expert_loads_gpu,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       comm_stream.stream());

        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        auto [quota_gpu, quota_prefix_gpu, rank_quota_prefix_gpu] = placement.get_quota_gpu_ptrs(layer_id);
        placement_solver_quota_->solve(global_logical_expert_loads_gpu,
                                       expert_loads_per_rank_gpu,
                                       p2l_gpu,
                                       l2p_gpu,
                                       lcnts_gpu,
                                       quota_gpu,
                                       quota_prefix_gpu,
                                       rank_quota_prefix_gpu,
                                       comm_stream.stream(),
                                       balance_threshold_,
                                       quota_min_tokens_per_replica_,
                                       quota_allow_zero_master_quota_,
                                       quota_locality_aware_,
                                       quota_oracle_eps_);
        placement_cpu_dirty_[layer_id] = true;
        record_placement_ready(layer_id, comm_stream);
    } else if (use_gpu_solver_) {
        nvshmem::int32_allreduce(global_logical_expert_loads_gpu, num_global_logical_experts, comm_stream.stream());
        // GPU path: solver on comm_stream after allreduce
        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        placement_solver_gpu_->solve(
            global_logical_expert_loads_gpu, p2l_gpu, l2p_gpu, lcnts_gpu, comm_stream.stream(), balance_threshold_);
        placement_cpu_dirty_[layer_id] = true;
        record_placement_ready(layer_id, comm_stream);
    } else {
        nvshmem::int32_allreduce(global_logical_expert_loads_gpu, num_global_logical_experts, comm_stream.stream());
        // CPU path: D2H → sync → CPU solve → H2D
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(global_logical_expert_loads_cpu,
                                           global_logical_expert_loads_gpu,
                                           num_global_logical_experts * sizeof(int),
                                           cudaMemcpyDeviceToHost,
                                           comm_stream.stream()));

        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream.stream()));

        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);
        placement_solver_->solve(global_logical_expert_loads_cpu, p2l_ptr, l2p_ptr, lcnts_ptr, balance_threshold_);

        placement.to_gpu(layer_id);
        record_placement_ready(layer_id, compute_stream);
    }
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

    auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);

    if (use_quota_solver_) {
        const int32_t* rank_quota_prefix_gpu = std::get<2>(placement.get_quota_gpu_ptrs(layer_id));
        kernels::run_reroute_sparse_quota(topk_ids.data_ptr<int64_t>(),
                                          l2p_gpu,
                                          lcnts_gpu,
                                          rank_quota_prefix_gpu,
                                          _reroute_sparse_counters_gpu,
                                          T,
                                          K,
                                          num_global_logical_experts,
                                          runtime::num_ranks,  // max_replicas
                                          stream);
    } else {
        kernels::run_reroute_sparse(topk_ids.data_ptr<int64_t>(),
                                    l2p_gpu,
                                    lcnts_gpu,
                                    _reroute_sparse_counters_gpu,
                                    T,
                                    K,
                                    num_global_logical_experts,
                                    runtime::num_ranks,  // max_replicas
                                    stream);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Manager::reroute_cpu(const int& layer_id,
                                                                             torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(!use_quota_solver_ && "CPU reroute path is not supported in quota mode");

    sync_placement_to_cpu(layer_id);
    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);
    return reroute_solver_->solve(routing_map, l2p_ptr, lcnts_ptr);
}

// CUDA reroute with pre-allocated output buffers
std::tuple<torch::Tensor, torch::Tensor> Manager::reroute_cuda_forward(const int& layer_id,
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

    // Get GPU placement pointers (H2D already done by update_placement)
    auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);

    if (T > 0 && L > 0) {
        constexpr int TILE_T = REROUTE_FWD_TILE_T;
        const int num_tiles = (T + TILE_T - 1) / TILE_T;
        int32_t* tile_counts_ptr = reroute_output_buffer_->get_or_create_tile_counts(L, num_tiles);

        EP_HOST_ASSERT(probs.scalar_type() == torch::kFloat32);

        if (use_quota_solver_) {
            auto [quota_gpu, quota_prefix_gpu, rank_quota_prefix_gpu] = placement.get_quota_gpu_ptrs(layer_id);
            kernels::run_reroute_forward_quota(routing_map.data_ptr<bool>(),
                                               probs.data_ptr(),
                                               l2p_gpu,
                                               lcnts_gpu,
                                               rank_quota_prefix_gpu,
                                               expand_rmap_ptr,
                                               expand_probs_ptr,
                                               tile_counts_ptr,
                                               T,
                                               L,
                                               P,
                                               runtime::num_ranks,
                                               stream);
        } else {
            kernels::run_reroute_forward(routing_map.data_ptr<bool>(),
                                         probs.data_ptr(),
                                         l2p_gpu,
                                         lcnts_gpu,
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

    // // Return fresh from_blob views — independent version counters so autograd
    // // will not see in-place conflicts when the buffer is reused for the next layer.
    // auto result_probs =
    //     torch::from_blob(expand_probs_ptr, {T, P}, torch::TensorOptions().dtype(probs.dtype()).device(device));
    // auto result_map =
    //     torch::from_blob(expand_rmap_ptr, {T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));

    return std::make_tuple(expanded_probs, expanded_rmap);
}

torch::Tensor Manager::reroute_cuda_backward(const int& layer_id,
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

    // Lazy allocation / reallocation of backward buffer
    auto bwd_buf_ptr = reroute_output_buffer_->get_or_create_bwd_buf(T, grad_expanded_probs.scalar_type());

    if (!reroute_output_buffer_->get_bwd_valid_flag()) {  // not zero-out
        reroute_output_buffer_->zero_out_bwd_buf(stream);
    }
    // Reset layer valid flag
    reroute_output_buffer_->set_bwd_valid_flag(false);

    auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);

    // // Retrieve forward's expanded_routing_map for the row-parallel backward gather.
    // const bool* fwd_expanded_rmap = reroute_output_buffer_->get_fwd_expanded_rmap_ptr(layer_id);
    const bool* fwd_expanded_rmap = expanded_routing_map.data_ptr<bool>();

    if (T > 0 && L > 0) {
        EP_HOST_ASSERT(grad_expanded_probs.scalar_type() == torch::kFloat32);

        kernels::run_reroute_backward(grad_expanded_probs.data_ptr(),
                                      routing_map.data_ptr<bool>(),
                                      fwd_expanded_rmap,
                                      l2p_gpu,
                                      lcnts_gpu,
                                      bwd_buf_ptr,
                                      T,
                                      L,
                                      P,
                                      runtime::num_ranks,
                                      stream);
    }

    auto result =
        torch::from_blob(bwd_buf_ptr, {T, L}, torch::TensorOptions().dtype(grad_expanded_probs.dtype()).device(device));

    return result;
}

std::optional<EventHandle> Manager::grad_reduce(const int& layer_id,
                                                torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                                torch::Tensor& local_master_fc2_grad_ptr_tensor,
                                                std::string& mode,
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

    if (use_gpu_solver_ || use_quota_solver_) {
        // GPU task build path: tasks built entirely on GPU
        int64_t* local_master_fc1_grad_ptrs_gpu = nullptr;
        int64_t* local_master_fc2_grad_ptrs_gpu = nullptr;
        if (local_master_fc1_grad_ptr_tensor.is_cuda()) {
            EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.is_cuda());
            local_master_fc1_grad_ptrs_gpu = local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>();
            local_master_fc2_grad_ptrs_gpu = local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>();
        } else {
            // Legacy compatibility: accept CPU ptr tensors and stage them to GPU.
            void** local_master_fc1_grad_ptrs =
                reinterpret_cast<void**>(local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>());
            void** local_master_fc2_grad_ptrs =
                reinterpret_cast<void**>(local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>());
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(_local_master_ptrs_staging_gpu,
                                               local_master_fc1_grad_ptrs,
                                               num_local_master_experts * sizeof(int64_t),
                                               cudaMemcpyHostToDevice,
                                               comm_stream));
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(_local_master_ptrs_staging_gpu + num_local_master_experts,
                                               local_master_fc2_grad_ptrs,
                                               num_local_master_experts * sizeof(int64_t),
                                               cudaMemcpyHostToDevice,
                                               comm_stream));
            local_master_fc1_grad_ptrs_gpu = _local_master_ptrs_staging_gpu;
            local_master_fc2_grad_ptrs_gpu = _local_master_ptrs_staging_gpu + num_local_master_experts;
        }

        // Build tasks on GPU
        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        kernels::build_grad_reduce_tasks(_task_build_config_gpu,
                                         p2l_gpu,
                                         l2p_gpu,
                                         lcnts_gpu,
                                         _remote_grad_ptrs_gpu,
                                         local_master_fc1_grad_ptrs_gpu,
                                         local_master_fc2_grad_ptrs_gpu,
                                         _grad_reduce_tasks_gpu,
                                         _task_tile_offsets_gpu,
                                         _task_metadata_gpu,
                                         _global_task_or_tile_counter_gpu,
                                         comm_stream);

        // Launch main kernel using GPU-resident tasks
        if (mode == "low_sm") {
            kernels::run_grad_reduce_low_sm_from_gpu(_grad_reduce_tasks_gpu,
                                                     _global_task_or_tile_counter_gpu,
                                                     _task_metadata_gpu,
                                                     comm_stream,
                                                     runtime::num_device_sms,
                                                     _max_gr_total_tasks);
        } else if (mode == "high_sm") {
            kernels::run_grad_reduce_high_sm_from_gpu(_grad_reduce_tasks_gpu,
                                                      _task_tile_offsets_gpu,
                                                      _task_metadata_gpu,
                                                      _global_task_or_tile_counter_gpu,
                                                      comm_stream,
                                                      runtime::num_device_sms,
                                                      _max_gr_total_tiles);
        } else {
            EP_HOST_ASSERT(false && "Invalid grad reduce mode");
        }
    } else {
        // CPU task build path (original)
        EP_HOST_ASSERT(!local_master_fc1_grad_ptr_tensor.is_cuda());
        EP_HOST_ASSERT(!local_master_fc2_grad_ptr_tensor.is_cuda());
        void** local_master_fc1_grad_ptrs =
            reinterpret_cast<void**>(local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>());
        void** local_master_fc2_grad_ptrs =
            reinterpret_cast<void**>(local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>());
        int num_tasks = 0;
        sync_placement_to_cpu(layer_id);
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);
        for (int i = 0; i < num_local_master_experts; ++i) {
            int master_global_phy_idx = runtime::rank_idx * num_local_physical_experts + i;
            int master_global_log_idx = p2l_ptr[master_global_phy_idx];
            int num_replicas = lcnts_ptr[master_global_log_idx];
            float* local_master_fc1_grad_ptr = reinterpret_cast<float*>(local_master_fc1_grad_ptrs[i]);
            float* local_master_fc2_grad_ptr = reinterpret_cast<float*>(local_master_fc2_grad_ptrs[i]);

            for (int j = 1; j < num_replicas; ++j) {  // skip the master itself
                int replica_global_phy_idx = l2p_ptr[master_global_log_idx * runtime::num_ranks + j];
                int replica_global_rank_idx = replica_global_phy_idx / num_local_physical_experts;
                EP_HOST_ASSERT(
                    is_in_same_nvl_domain(runtime::rank_idx, replica_global_rank_idx, runtime::num_nvl_ranks) &&
                    "Replica rank is not in the same NVL domain as the master rank");
                int replica_nvl_rank_idx = replica_global_rank_idx % runtime::num_nvl_ranks;
                EP_HOST_ASSERT(replica_nvl_rank_idx != runtime::nvl_rank_idx &&
                               "Replica rank is the same as the master rank, which is not allowed");
                EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[replica_nvl_rank_idx] != nullptr);
                int replica_local_offset =
                    replica_global_phy_idx % num_local_physical_experts - num_local_master_experts;
                EP_HOST_ASSERT(replica_local_offset >= 0 and replica_local_offset < num_local_redundant_experts);
                float* replica_remote_grad_buffer_ptr =
                    reinterpret_cast<float*>(global_replica_grad_buffer_ptrs[replica_nvl_rank_idx]);
                float* replica_remote_fc1_grad_ptr =
                    replica_remote_grad_buffer_ptr + replica_local_offset * expert_total_numel;
                float* replica_remote_fc2_grad_ptr = replica_remote_fc1_grad_ptr + expert_fc1_numel;
                _grad_reduce_tasks_cpu[num_tasks++] = {
                    local_master_fc1_grad_ptr, replica_remote_fc1_grad_ptr, static_cast<size_t>(expert_fc1_numel)};
                _grad_reduce_tasks_cpu[num_tasks++] = {
                    local_master_fc2_grad_ptr, replica_remote_fc2_grad_ptr, static_cast<size_t>(expert_fc2_numel)};
            }
        }
        if (num_tasks == 0) {
            if (async) {
                event = EventHandle(comm_stream);
            }
            return event;
        }

        // Call device-side kernels
        if (mode == "low_sm") {
            kernels::run_grad_reduce_low_sm(_grad_reduce_tasks_cpu,
                                            _grad_reduce_tasks_gpu,
                                            _global_task_or_tile_counter_gpu,
                                            _task_metadata_gpu,
                                            num_tasks,
                                            comm_stream,
                                            runtime::num_device_sms);
        } else if (mode == "high_sm") {
            kernels::run_grad_reduce_high_sm(_grad_reduce_tasks_cpu,
                                             _grad_reduce_tasks_gpu,
                                             _global_task_or_tile_counter_gpu,
                                             _task_tile_offsets_gpu,
                                             _task_metadata_gpu,
                                             num_tasks,
                                             comm_stream,
                                             runtime::num_device_sms);
        } else {
            EP_HOST_ASSERT(false && "Invalid grad reduce mode");
        }
    }

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
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.numel() == num_local_master_experts);

    if (use_gpu_solver_ || use_quota_solver_) {
        // GPU task build path: tasks built entirely on GPU
        int64_t* local_master_fc1_weight_ptrs_gpu = nullptr;
        int64_t* local_master_fc2_weight_ptrs_gpu = nullptr;
        if (local_master_fc1_weight_ptr_tensor.is_cuda()) {
            EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.is_cuda());
            local_master_fc1_weight_ptrs_gpu = local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>();
            local_master_fc2_weight_ptrs_gpu = local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>();
        } else {
            // Legacy compatibility: accept CPU ptr tensors and stage them to GPU.
            void** local_master_fc1_weight_ptrs =
                reinterpret_cast<void**>(local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>());
            void** local_master_fc2_weight_ptrs =
                reinterpret_cast<void**>(local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>());
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(_local_master_ptrs_staging_gpu,
                                               local_master_fc1_weight_ptrs,
                                               num_local_master_experts * sizeof(int64_t),
                                               cudaMemcpyHostToDevice,
                                               comm_stream));
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(_local_master_ptrs_staging_gpu + num_local_master_experts,
                                               local_master_fc2_weight_ptrs,
                                               num_local_master_experts * sizeof(int64_t),
                                               cudaMemcpyHostToDevice,
                                               comm_stream));
            local_master_fc1_weight_ptrs_gpu = _local_master_ptrs_staging_gpu;
            local_master_fc2_weight_ptrs_gpu = _local_master_ptrs_staging_gpu + num_local_master_experts;
        }

        auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);
        kernels::build_weight_sync_tasks(_task_build_config_gpu,
                                         p2l_gpu,
                                         l2p_gpu,
                                         lcnts_gpu,
                                         _remote_weight_ptrs_gpu,
                                         local_master_fc1_weight_ptrs_gpu,
                                         local_master_fc2_weight_ptrs_gpu,
                                         _weight_sync_tasks_gpu,
                                         _task_tile_offsets_gpu,
                                         _task_metadata_gpu,
                                         _global_task_or_tile_counter_gpu,
                                         comm_stream);

        kernels::run_weight_sync_from_gpu(_weight_sync_tasks_gpu,
                                          _task_tile_offsets_gpu,
                                          _task_metadata_gpu,
                                          _global_task_or_tile_counter_gpu,
                                          comm_stream,
                                          runtime::num_device_sms,
                                          _max_ws_total_tiles);
    } else {
        // CPU task build path (original)
        EP_HOST_ASSERT(!local_master_fc1_weight_ptr_tensor.is_cuda());
        EP_HOST_ASSERT(!local_master_fc2_weight_ptr_tensor.is_cuda());
        void** local_master_fc1_weight_ptrs =
            reinterpret_cast<void**>(local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>());
        void** local_master_fc2_weight_ptrs =
            reinterpret_cast<void**>(local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>());
        int num_tasks = 0;
        sync_placement_to_cpu(layer_id);
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);
        for (int i = 0; i < num_local_master_experts; ++i) {
            int master_global_phy_idx = runtime::rank_idx * num_local_physical_experts + i;
            int master_global_log_idx = p2l_ptr[master_global_phy_idx];
            int num_replicas = lcnts_ptr[master_global_log_idx] - 1;  // Exclude master itself

            if (num_replicas == 0) {
                continue;  // No replicas to sync to
            }

            __nv_bfloat16* local_master_fc1_weight_ptr =
                reinterpret_cast<__nv_bfloat16*>(local_master_fc1_weight_ptrs[i]);
            __nv_bfloat16* local_master_fc2_weight_ptr =
                reinterpret_cast<__nv_bfloat16*>(local_master_fc2_weight_ptrs[i]);

            // Create FC1 task
            kernels::WeightSyncTask& fc1_task = _weight_sync_tasks_cpu[num_tasks];
            fc1_task.master_local_addr = local_master_fc1_weight_ptr;
            fc1_task.num_replicas = num_replicas;
            fc1_task.numel = static_cast<size_t>(expert_fc1_numel);

            // Create FC2 task
            kernels::WeightSyncTask& fc2_task = _weight_sync_tasks_cpu[num_tasks + 1];
            fc2_task.master_local_addr = local_master_fc2_weight_ptr;
            fc2_task.num_replicas = num_replicas;
            fc2_task.numel = static_cast<size_t>(expert_fc2_numel);

            // Fill replica addresses for both tasks
            for (int j = 0; j < num_replicas; ++j) {
                // j+1 because index 0 is the master itself in logical_to_physical_map
                int replica_global_phy_idx = l2p_ptr[master_global_log_idx * runtime::num_ranks + j + 1];
                int replica_global_rank_idx = replica_global_phy_idx / num_local_physical_experts;
                EP_HOST_ASSERT(
                    is_in_same_nvl_domain(runtime::rank_idx, replica_global_rank_idx, runtime::num_nvl_ranks) &&
                    "Replica rank is not in the same NVL domain as the master rank");
                int replica_nvl_rank_idx = replica_global_rank_idx % runtime::num_nvl_ranks;
                EP_HOST_ASSERT(replica_nvl_rank_idx != runtime::nvl_rank_idx &&
                               "Replica rank is the same as the master rank, which is not allowed");
                EP_HOST_ASSERT(global_replica_weight_buffer_ptrs[replica_nvl_rank_idx] != nullptr);

                int replica_local_offset =
                    replica_global_phy_idx % num_local_physical_experts - num_local_master_experts;
                EP_HOST_ASSERT(replica_local_offset >= 0 && replica_local_offset < num_local_redundant_experts);

                __nv_bfloat16* replica_remote_weight_buffer_ptr =
                    reinterpret_cast<__nv_bfloat16*>(global_replica_weight_buffer_ptrs[replica_nvl_rank_idx]);
                __nv_bfloat16* replica_remote_fc1_weight_ptr =
                    replica_remote_weight_buffer_ptr + replica_local_offset * expert_total_numel;
                __nv_bfloat16* replica_remote_fc2_weight_ptr = replica_remote_fc1_weight_ptr + expert_fc1_numel;

                fc1_task.replica_remote_addrs[j] = replica_remote_fc1_weight_ptr;
                fc2_task.replica_remote_addrs[j] = replica_remote_fc2_weight_ptr;
            }

            num_tasks += 2;
        }

        if (num_tasks == 0) {
            if (async) {
                event = EventHandle(comm_stream);
            }
            return event;
        }
        // Ensure the task tile offsets buffer is large enough
        EP_HOST_ASSERT(num_tasks + 1 < 3 * num_local_master_experts);

        // Call device-side kernel
        kernels::run_weight_sync(_weight_sync_tasks_cpu,
                                 _weight_sync_tasks_gpu,
                                 _global_task_or_tile_counter_gpu,
                                 _task_tile_offsets_gpu,
                                 _task_tile_offsets_cpu,
                                 _task_metadata_gpu,
                                 num_tasks,
                                 comm_stream,
                                 runtime::num_device_sms);
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
