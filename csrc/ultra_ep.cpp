#include "ultra_ep.hpp"

#include <cuda_bf16.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace ultra_ep {

Manager::Manager(const int& num_layers,
                 const int& num_local_master_experts,
                 const int& num_local_redundant_experts,
                 const int64_t& expert_fc1_numel,
                 const int64_t& expert_fc2_numel,
                 const bool& explicitly_destroy)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true))

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    num_global_physical_experts = num_local_physical_experts * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;

    // Allocate global placement tensors on CPU
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.physical_to_logical_map =
        torch::full({num_layers, num_global_physical_experts},
                    -1,
                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    placement.logical_to_physical_map =
        torch::full({num_layers, num_global_logical_experts, num_ranks},
                    -1,
                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    placement.logical_replica_counts =
        torch::zeros({num_layers, num_global_logical_experts},
                     torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));

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
    CUDA_RUNTIME_CHECK(
        cudaMallocHost((void**)&global_logical_expert_loads_cpu, num_global_logical_experts * sizeof(int)));
    // Create pre-allocated reroute solver
    reroute_solver_ = std::make_unique<solver::RerouteSolver>(num_global_logical_experts,
                                                              num_global_physical_experts,
                                                              runtime::num_ranks  // max_replicas_dim = num_ranks
    );

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
    _grad_reduce_tasks_cpu = nullptr;
    _grad_reduce_tasks_gpu = nullptr;
    _global_task_or_tile_counter_gpu = nullptr;
    _task_tile_offsets_gpu = nullptr;

    // Free weight sync buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(_weight_sync_tasks_cpu));
    CUDA_RUNTIME_CHECK(cudaFree(_weight_sync_tasks_gpu));
    _weight_sync_tasks_cpu = nullptr;
    _weight_sync_tasks_gpu = nullptr;

    // Free expert load buffers
    CUDA_RUNTIME_CHECK(cudaFreeHost(global_logical_expert_loads_cpu));
    global_logical_expert_loads_cpu = nullptr;

    // Free NVSHMEM runtime
    runtime::destroy();

    // Ready to destroy
    _available = false;
}

void Manager::update_placement(const int& layer_id, torch::Tensor& expert_loads) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(expert_loads.dim() == 1 && expert_loads.size(0) == num_global_logical_experts &&
                   expert_loads.dtype() == torch::kInt32);

    int* expert_loads_ptr = nullptr;
    if (expert_loads.is_cuda()) {
        CUDA_RUNTIME_CHECK(cudaMemcpy(global_logical_expert_loads_cpu,
                                      expert_loads.data<int32_t>(),
                                      num_global_logical_experts * sizeof(int),
                                      cudaMemcpyDeviceToHost));
        expert_loads_ptr = global_logical_expert_loads_cpu;
    } else {
        expert_loads_ptr = expert_loads.data<int32_t>();
    }
    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = get_placement_map_ptrs(layer_id);

    placement_solver_->solve(expert_loads_ptr, p2l_ptr, l2p_ptr, lcnts_ptr);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Manager::reroute(const int& layer_id,
                                                                         torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);

    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = get_placement_map_ptrs(layer_id);
    return reroute_solver_->solve(routing_map, l2p_ptr, lcnts_ptr);
}

std::tuple<int32_t*, int32_t*, int32_t*> Manager::get_placement_map_ptrs(const int& layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    int p2l_offset = layer_id * num_global_physical_experts;
    int l2p_offset = layer_id * num_global_logical_experts * runtime::num_ranks;
    int lcnts_offset = layer_id * num_global_logical_experts;
    return std::make_tuple(placement.physical_to_logical_map.data<int32_t>() + p2l_offset,
                           placement.logical_to_physical_map.data<int32_t>() + l2p_offset,
                           placement.logical_replica_counts.data<int32_t>() + lcnts_offset);
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
    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = get_placement_map_ptrs(layer_id);
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
    auto [p2l_ptr, l2p_ptr, lcnts_ptr] = get_placement_map_ptrs(layer_id);
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

    // Call device-side kernel
    kernels::run_weight_sync(_weight_sync_tasks_cpu,
                             _weight_sync_tasks_gpu,
                             _global_task_or_tile_counter_gpu,
                             _task_tile_offsets_gpu,
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