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
    per_layer_data_numel = p2l_numel + l2p_numel + lcnts_numel;
    per_layer_data_bytes = per_layer_data_numel * static_cast<int>(sizeof(int32_t));

    // Align stride so each layer starts on a 256-byte boundary (good for DMA)
    per_layer_stride_bytes = (per_layer_data_bytes + ALIGNMENT_BYTES - 1) / ALIGNMENT_BYTES * ALIGNMENT_BYTES;
    per_layer_stride_numel = per_layer_stride_bytes / static_cast<int>(sizeof(int32_t));
    total_bytes = num_layers * per_layer_stride_bytes;

    // Allocate CPU pinned buffer
    CUDA_RUNTIME_CHECK(cudaMallocHost(&cpu_buffer, total_bytes));

    // Initialize: p2l and l2p → -1, lcnts → 0, padding → 0
    std::memset(cpu_buffer, 0xFF, total_bytes);
    for (int i = 0; i < num_layers; ++i) {
        int32_t* layer_base = cpu_buffer + i * per_layer_stride_numel;
        // Zero out lcnts
        std::memset(layer_base + p2l_numel + l2p_numel, 0, lcnts_numel * sizeof(int32_t));
        // Zero out padding
        int pad_numel = per_layer_stride_numel - per_layer_data_numel;
        if (pad_numel > 0) {
            std::memset(layer_base + per_layer_data_numel, 0, pad_numel * sizeof(int32_t));
        }
    }

    // Allocate GPU buffer (zero-initialized)
    CUDA_RUNTIME_CHECK(cudaMalloc(&gpu_buffer, total_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(gpu_buffer, 0, total_bytes));

    // Create CPU tensor views with strides
    // Shape [num_layers, X] with stride[0] = per_layer_stride_numel
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    physical_to_logical_map = torch::from_blob(cpu_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, cpu_opts);
    logical_to_physical_map =
        torch::from_blob(cpu_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, cpu_opts);
    logical_replica_counts =
        torch::from_blob(cpu_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, cpu_opts);

    // Create GPU tensor views with strides
    auto gpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    physical_to_logical_map_gpu = torch::from_blob(gpu_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, gpu_opts);
    logical_to_physical_map_gpu =
        torch::from_blob(gpu_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, gpu_opts);
    logical_replica_counts_gpu =
        torch::from_blob(gpu_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, gpu_opts);
}

void GlobalExpertPlacement::cleanup() {
    if (cpu_buffer != nullptr) {
        cudaFreeHost(cpu_buffer);
        cpu_buffer = nullptr;
    }
    if (gpu_buffer != nullptr) {
        cudaFree(gpu_buffer);
        gpu_buffer = nullptr;
    }
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
                 const bool& explicitly_destroy)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      is_train(is_train),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      memset_stream(at::cuda::getStreamFromPool(false))

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
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

    // Allocate local replica weight and grad buffers via NVSHMEM symmetric heap
    // This enables automatic cross-GPU access within NVL domain
    int64_t local_replica_weight_bytes =
        (int64_t)num_local_redundant_experts * expert_total_numel * WEIGHT_ELEMENT_SIZE;
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
    local_replica_grad_buffer_tensor.zero_();

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
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&_grad_reduce_tasks_cpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_grad_reduce_tasks_gpu, MAX_GRAD_REDUCE_TASK_NUM * sizeof(kernels::GradReduceTask)));
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_global_task_or_tile_counter_gpu, sizeof(int)));
    // +1 for the final offset (total tile count)
    CUDA_RUNTIME_CHECK(cudaMalloc((void**)&_task_tile_offsets_gpu, (MAX_GRAD_REDUCE_TASK_NUM + 1) * sizeof(int)));
    CUDA_RUNTIME_CHECK(cudaMallocHost((void**)&_task_tile_offsets_cpu, 3 * num_local_master_experts * sizeof(int)));

    // Allocate intermediate buffers for weight sync tasks
    // For weight sync, each local master expert creates one broadcast task
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&_weight_sync_tasks_cpu, MAX_WEIGHT_SYNC_TASK_NUM * sizeof(kernels::WeightSyncTask)));
    CUDA_RUNTIME_CHECK(
        cudaMalloc((void**)&_weight_sync_tasks_gpu, MAX_WEIGHT_SYNC_TASK_NUM * sizeof(kernels::WeightSyncTask)));

    // Create pre-allocated placement solver (zero-alloc on hot path)
    placement_solver_ = std::make_unique<solver::PlacementSolver>(num_global_logical_experts,
                                                                  runtime::num_ranks,
                                                                  num_local_master_experts,
                                                                  num_local_redundant_experts,
                                                                  runtime::num_nvl_ranks,
                                                                  runtime::num_ranks  // max_replicas_dim = num_ranks
    );
    // Allocate global logical expert load buffer
    global_logical_expert_loads_gpu =
        reinterpret_cast<int*>(nvshmem::alloc(num_global_logical_experts * sizeof(int), NVSHMEM_ALIGNMENT));
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
    nvshmem::free(global_logical_expert_loads_gpu);
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

void Manager::update_placement(const int& layer_id, torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(routing_map.dim() == 2 && routing_map.size(1) == num_global_logical_experts &&
                   routing_map.dtype() == torch::kBool);

    auto curr_stream = at::cuda::getCurrentCUDAStream();

    kernels::rmap_local_sum_and_allreduce(routing_map.size(0),
                                          num_global_logical_experts,
                                          routing_map.data_ptr<bool>(),
                                          global_logical_expert_loads_gpu,
                                          curr_stream.stream());

    // Copy global logical expert loads from GPU to CPU
    // Must use current stream to ensure data readiness
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(global_logical_expert_loads_cpu,
                                       global_logical_expert_loads_gpu,
                                       num_global_logical_experts * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       curr_stream.stream()));

    // Zero-out reroute buffer in advance for overlapping
    reroute_output_buffer_->zero_out_fwd_bufs(layer_id, memset_stream);

    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);

    // Ensure data readiness for CPU-side placement solver
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(curr_stream.stream()));

    placement_solver_->solve(global_logical_expert_loads_cpu, p2l_ptr, l2p_ptr, lcnts_ptr);

    // Move placement to GPU for later use
    placement.to_gpu(layer_id);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Manager::reroute_cpu(const int& layer_id,
                                                                             torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);

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

    // Lazy allocation / reallocation of output buffers
    // These buffers are async zero-out during update_placement for zero-overhead
    auto [expand_probs_ptr, expand_rmap_ptr] =
        reroute_output_buffer_->get_or_create_fwd_bufs(T, layer_id, probs.scalar_type());

    // Get GPU placement pointers (H2D already done by update_placement)
    auto [p2l_gpu, l2p_gpu, lcnts_gpu] = placement.get_gpu_ptrs(layer_id);

    if (!reroute_output_buffer_->get_fwd_valid_flag(layer_id)) {  // not zero-out
        reroute_output_buffer_->zero_out_fwd_bufs(layer_id, memset_stream);
    }
    // Can wait for memset in update_place or here above
    stream_wait(stream, memset_stream);
    // Reset layer valid flag
    reroute_output_buffer_->set_fwd_valid_flag(layer_id, false);

    if (T > 0 && L > 0) {
        constexpr int TILE_T = REROUTE_FWD_TILE_T;
        const int num_tiles = (T + TILE_T - 1) / TILE_T;
        int32_t* tile_counts_ptr = reroute_output_buffer_->get_or_create_tile_counts(L, num_tiles);

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
                                     probs.scalar_type(),
                                     stream);
    }

    // Return fresh from_blob views — independent version counters so autograd
    // will not see in-place conflicts when the buffer is reused for the next layer.
    auto result_probs =
        torch::from_blob(expand_probs_ptr, {T, P}, torch::TensorOptions().dtype(probs.dtype()).device(device));
    auto result_map =
        torch::from_blob(expand_rmap_ptr, {T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));

    return std::make_tuple(result_probs, result_map);
}

torch::Tensor Manager::reroute_cuda_backward(const int& layer_id,
                                             torch::Tensor& grad_expanded_probs,
                                             torch::Tensor& routing_map) {
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

    // Retrieve forward's expanded_routing_map for the row-parallel backward gather.
    const bool* fwd_expanded_rmap = reroute_output_buffer_->get_fwd_expanded_rmap_ptr(layer_id);

    if (T > 0 && L > 0) {
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
                                      grad_expanded_probs.scalar_type(),
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

    void** local_master_fc1_grad_ptrs = reinterpret_cast<void**>(local_master_fc1_grad_ptr_tensor.data<int64_t>());
    void** local_master_fc2_grad_ptrs = reinterpret_cast<void**>(local_master_fc2_grad_ptr_tensor.data<int64_t>());

    // Flatten task list (host-side)
    int num_tasks = 0;
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
            EP_HOST_ASSERT(is_in_same_nvl_domain(runtime::rank_idx, replica_global_rank_idx, runtime::num_nvl_ranks) &&
                           "Replica rank is not in the same NVL domain as the master rank");
            int replica_nvl_rank_idx = replica_global_rank_idx % runtime::num_nvl_ranks;
            EP_HOST_ASSERT(replica_nvl_rank_idx != runtime::nvl_rank_idx &&
                           "Replica rank is the same as the master rank, which is not allowed");
            EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[replica_nvl_rank_idx] != nullptr);
            int replica_local_offset = replica_global_phy_idx % num_local_physical_experts - num_local_master_experts;
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
                                        num_tasks,
                                        comm_stream,
                                        runtime::num_device_sms);
    } else if (mode == "high_sm") {
        kernels::run_grad_reduce_high_sm(_grad_reduce_tasks_cpu,
                                         _grad_reduce_tasks_gpu,
                                         _global_task_or_tile_counter_gpu,
                                         _task_tile_offsets_gpu,
                                         num_tasks,
                                         comm_stream,
                                         runtime::num_device_sms);
    } else {
        EP_HOST_ASSERT(false && "Invalid grad reduce mode");
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

    void** local_master_fc1_weight_ptrs = reinterpret_cast<void**>(local_master_fc1_weight_ptr_tensor.data<int64_t>());
    void** local_master_fc2_weight_ptrs = reinterpret_cast<void**>(local_master_fc2_weight_ptr_tensor.data<int64_t>());

    // Build broadcast tasks: each local master broadcasts to all its replicas
    // Each master creates two tasks: one for FC1, one for FC2
    int num_tasks = 0;
    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_cpu_ptrs(layer_id);
    for (int i = 0; i < num_local_master_experts; ++i) {
        int master_global_phy_idx = runtime::rank_idx * num_local_physical_experts + i;
        int master_global_log_idx = p2l_ptr[master_global_phy_idx];
        int num_replicas = lcnts_ptr[master_global_log_idx] - 1;  // Exclude master itself

        if (num_replicas == 0) {
            continue;  // No replicas to sync to
        }

        __nv_bfloat16* local_master_fc1_weight_ptr = reinterpret_cast<__nv_bfloat16*>(local_master_fc1_weight_ptrs[i]);
        __nv_bfloat16* local_master_fc2_weight_ptr = reinterpret_cast<__nv_bfloat16*>(local_master_fc2_weight_ptrs[i]);

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
            EP_HOST_ASSERT(is_in_same_nvl_domain(runtime::rank_idx, replica_global_rank_idx, runtime::num_nvl_ranks) &&
                           "Replica rank is not in the same NVL domain as the master rank");
            int replica_nvl_rank_idx = replica_global_rank_idx % runtime::num_nvl_ranks;
            EP_HOST_ASSERT(replica_nvl_rank_idx != runtime::nvl_rank_idx &&
                           "Replica rank is the same as the master rank, which is not allowed");
            EP_HOST_ASSERT(global_replica_weight_buffer_ptrs[replica_nvl_rank_idx] != nullptr);

            int replica_local_offset = replica_global_phy_idx % num_local_physical_experts - num_local_master_experts;
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
                             num_tasks,
                             comm_stream,
                             runtime::num_device_sms);

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

}  // namespace ultra_ep