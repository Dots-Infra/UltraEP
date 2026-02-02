#pragma once

// Max NVL domain size in existing scale-up domains, not the actual size
#define MAX_NVL_DOMAIN_SIZE 72

// Equal to max #experts per layer of a model
#define MAX_GRAD_REDUCE_TASK_NUM 256

// Suppose BF16 weight data
#define WEIGHT_ELEMENT_SIZE 2
// Suppose FP32 grad data
#define GRAD_ELEMENT_SIZE 4
