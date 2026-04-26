#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels/config.cuh"
#include "utils/exception.cuh"
#include "utils/nvshmem.cuh"

namespace ultra_ep::runtime {

extern bool is_runtime_initialized;

extern int rank_idx, nvl_rank_idx, rdma_rank_idx;
extern int num_ranks, num_nvl_ranks, num_rdma_ranks;
extern int device_id, num_device_sms;

at::cuda::CUDAStream get_global_comm_stream();

pybind11::bytes get_local_nvshmem_unique_id(const int& rank);

void init_runtime(const int& rank_idx_,
                  const int& num_ranks_,
                  const int& max_nvl_peers_,
                  const pybind11::bytes& root_unique_id);

void destroy();

void register_apis(pybind11::module_& m);

}  // namespace ultra_ep::runtime
