#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../config.hpp"
#include "../utils/exception.cuh"
#include "ptx.cuh"

// Grad reduce hyperparameters
// Max shared memory per block is ~227KB on Hopper/Blackwell
// Use 64KB * 3 stages = 192KB to stay within limit while maintaining pipeline depth
constexpr int GRAD_REDUCE_TILE_SIZE_BYTES = 64 * 1024;  // 64kB SMEM for TMA
constexpr int GRAD_REDUCE_TILE_ELEMENTS = GRAD_REDUCE_TILE_SIZE_BYTES / sizeof(float);
constexpr int GRAD_REDUCE_PIPELINE_STAGES = 3;
constexpr int GRAD_REDUCE_THREADS_PER_BLOCK = 256;
