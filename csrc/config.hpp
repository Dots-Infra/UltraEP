#pragma once

// Max NVL domain size in existing scale-up domains, not the actual size
#define MAX_NVL_DOMAIN_SIZE 72

// Equal to max #experts per layer of a model
#define MAX_GRAD_REDUCE_TASK_NUM 256
#define MAX_WEIGHT_SYNC_TASK_NUM 256

// NVSHMEM alignment for symmetric heap
#define NVSHMEM_ALIGNMENT 16

// Suppose BF16 weight data
#define WEIGHT_ELEMENT_SIZE 2
// Suppose FP32 grad data
#define GRAD_ELEMENT_SIZE 4

// Tile sizes for weight_sync and grad_reduce kernels (derived from kernel config).
// Used by Manager for pre-computing grid size upper bounds.
#define WEIGHT_SYNC_TILE_ELEMENTS (32 * 1024 / WEIGHT_ELEMENT_SIZE)  // 16384
#define GRAD_REDUCE_TILE_ELEMENTS (64 * 1024 / GRAD_ELEMENT_SIZE)    // 16384

// Reroute forward tile size (tokens per tile in the two-pass forward kernel).
// Defined here so it is accessible from both .cu (via config.cuh) and .cpp files.
#define REROUTE_FWD_TILE_T 128
