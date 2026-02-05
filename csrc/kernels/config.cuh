#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../config.hpp"
#include "../utils/exception.cuh"
#include "ptx.cuh"

// Grad reduce hyperparameters
// Max shared memory per block is ~227KB on Hopper/Blackwell
constexpr int GRAD_REDUCE_TILE_SIZE_BYTES = 64 * 1024;
constexpr int GRAD_REDUCE_TILE_ELEMENTS = GRAD_REDUCE_TILE_SIZE_BYTES / sizeof(float);
constexpr int GRAD_REDUCE_PIPELINE_STAGES = 2;
constexpr int GRAD_REDUCE_THREADS_PER_BLOCK = 256;