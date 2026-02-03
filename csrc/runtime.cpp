#include "runtime.hpp"

namespace ultra_ep::runtime {

bool is_runtime_initialized = false;

int rank_idx = -1, nvl_rank_idx = -1, rdma_rank_idx = -1;
int num_ranks = 0, num_nvl_ranks = 0, num_rdma_ranks = 0;
int device_id = -1, num_device_sms = 0;

at::cuda::CUDAStream get_global_comm_stream() {
    static std::optional<at::cuda::CUDAStream> comm_stream = std::nullopt;
    if (not comm_stream.has_value())
        comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream.value();
}

pybind11::bytes get_local_nvshmem_unique_id(const int& rank) {
    EP_HOST_ASSERT(rank == 0 and "Only rank 0 can get NVSHMEM unique ID");
    const auto unique_id = nvshmem::get_unique_id();
    return pybind11::bytes(reinterpret_cast<const char*>(unique_id.data()), unique_id.size());
}

void init_runtime(const int& rank_idx_,
                  const int& num_ranks_,
                  const int& max_nvl_peers_,
                  const pybind11::bytes& root_unique_id) {
    std::string root_unique_id_str = root_unique_id;
    std::vector<uint8_t> root_unique_id_bytes(root_unique_id_str.begin(), root_unique_id_str.end());
    EP_HOST_ASSERT(rank_idx_ == nvshmem::init(root_unique_id_bytes, rank_idx_, num_ranks_, 0));

    // Support both nvl and rdma ranks
    num_ranks = num_ranks_;
    num_nvl_ranks = max_nvl_peers_;
    EP_HOST_ASSERT(num_nvl_ranks <= MAX_NVL_DOMAIN_SIZE);
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    num_rdma_ranks = num_ranks / num_nvl_ranks;
    rank_idx = rank_idx_;
    nvl_rank_idx = rank_idx_ % num_nvl_ranks;
    rdma_rank_idx = rank_idx_ / num_nvl_ranks;

    // Get device info
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop = {};
    CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Available to create buffers
    is_runtime_initialized = true;
}

void destroy() {
    EP_HOST_ASSERT(is_runtime_initialized);

    // Leverage NVSHMEM
    nvshmem::finalize();

    // Cannot use anymore
    rank_idx = nvl_rank_idx = rdma_rank_idx = -1;
    num_ranks = num_nvl_ranks = num_rdma_ranks = 0;
    is_runtime_initialized = false;
}

void register_apis(pybind11::module_& m) {
    m.def("get_local_nvshmem_unique_id", &get_local_nvshmem_unique_id);
    m.def("is_runtime_initialized", []() { return is_runtime_initialized; });
    m.def("init_runtime", &init_runtime);
}

}  // namespace ultra_ep::runtime
