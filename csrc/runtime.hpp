#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels/config.cuh"
#include "utils/exception.cuh"

namespace ultra_ep::runtime {

extern bool is_runtime_initialized;

extern int rank_idx, nvl_rank_idx;
extern int num_ranks, num_nvl_ranks;
extern int device_id, num_device_sms;

void init_runtime(const int& rank_idx_, const int& num_ranks_, const int& num_nvl_ranks_);

void destroy();

void register_apis(pybind11::module_& m);

}  // namespace ultra_ep::runtime
