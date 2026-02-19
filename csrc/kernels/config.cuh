#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
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

// Weight sync hyperparameters
// Use smaller tiles for weight sync since we may have multiple replicas to store to
// This balances SMEM usage and parallelism
// 32KB tile = 16K bf16 elements
constexpr int WEIGHT_SYNC_TILE_SIZE_BYTES = 32 * 1024;
constexpr int WEIGHT_SYNC_TILE_ELEMENTS = WEIGHT_SYNC_TILE_SIZE_BYTES / sizeof(__nv_bfloat16);
constexpr int WEIGHT_SYNC_THREADS_PER_BLOCK = 256;
// Double buffer for pipelining: load next tile while storing current
constexpr int WEIGHT_SYNC_PIPELINE_STAGES = 2;

// Reroute kernel hyperparameters
// One warp per logical expert, SMEM stores transposed routing_map tile
// TILE_T: number of token rows per tile (processed in warp-scan steps of 32)
constexpr int REROUTE_TILE_T = 256;
// Number of warps (= experts) per CTA
constexpr int REROUTE_WARPS_PER_BLOCK = 8;
constexpr int REROUTE_THREADS_PER_BLOCK = REROUTE_WARPS_PER_BLOCK * 32;
// Padding to eliminate SMEM bank conflicts when reading columns
// With bool (1 byte) and 32 banks of 4 bytes each, stride must not be a multiple of 128
// Adding 4 bytes shifts successive columns to different banks
constexpr int REROUTE_SMEM_PAD = 4;