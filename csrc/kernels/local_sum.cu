#include "../utils/exception.cuh"
#include "api.cuh"

namespace ultra_ep::kernels {

__global__ void rmap_local_sum_kernel(const bool* __restrict__ routing_map,
                                      int32_t* __restrict__ expert_loads,
                                      int T,
                                      int L) {
    // index for global_logical_expert (dimension 1), in range [0, L-1]
    int tx = threadIdx.x;
    // row group index within current Block (dimension 0), in range [0, blockDim.y-1]
    int ty = threadIdx.y;

    int sum = 0;
    int stride = gridDim.x * blockDim.y;

    // Grid-stride and block-stride loop: handle all tokens (T)
    // Memory coalescing: within a warp, consecutive tx access consecutive bool values in row-major order, maximizing
    // bandwidth
    for (int r = blockIdx.x * blockDim.y + ty; r < T; r += stride) {
        // use mask to protect read, threads where tx >= L will not access memory, sum will be 0
        if (tx < L) {
            sum += static_cast<int>(routing_map[r * L + tx]);
        }
    }

    // Use dynamic shared memory for reduction along the Y dimension (row direction) within the block
    extern __shared__ int smem_sum[];

    // Write to shared memory; consecutive tx map to consecutive memory locations for zero bank conflict
    smem_sum[ty * blockDim.x + tx] = sum;
    __syncthreads();

    // Threads where ty == 0 will aggregate sums for their column and write to global memory
    if (ty == 0) {
        int col_sum = 0;
#pragma unroll
        for (int i = 0; i < blockDim.y; ++i) {
            col_sum += smem_sum[i * blockDim.x + tx];
        }

        // Accumulate across blocks into final expert_loads using atomicAdd
        if (tx < L) {
            atomicAdd(&expert_loads[tx], col_sum);
        }
    }
}

void rmap_local_sum(int T,
                    int L,
                    const bool* routing_map_ptr,  // [T, L] bool
                    int32_t* expert_loads_ptr,    // [L] int32, alloc by nvshmem
                    cudaStream_t stream) {
    // 1. Zero out the target array (since we accumulate via atomicAdd)
    // For 128/256 int32s, cudaMemsetAsync has very little overhead on the stream
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(expert_loads_ptr, 0, L * sizeof(int32_t), stream));

    // 2. Launch local sum kernel
    int block_x = ((L + 31) / 32) * 32;  // block_x = 32 * ceil(L/32)
    int threads_y = 1024 / block_x;      // maximize parallelism along Y dimension
    dim3 block(block_x, threads_y);

    // Launch enough blocks to fill all SMs (H100 has 132, Blackwell even more, here at most 256)
    int num_blocks = min(256, (T + threads_y - 1) / threads_y);
    dim3 grid(num_blocks);

    size_t smem_size = block.y * block.x * sizeof(int);

    CUDA_RUNTIME_CHECK(
        cudaFuncSetAttribute(rmap_local_sum_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    rmap_local_sum_kernel<<<grid, block, smem_size, stream>>>(routing_map_ptr, expert_loads_ptr, T, L);
}

// ---------------------------------------------------------------------------
// sparse_topk_histogram_kernel
//
// Computes per-expert token counts from sparse topk_ids [T, K].
// Each thread processes one topk entry via grid-stride loop.
// Uses shared-memory reduction per block to minimize global atomics.
// ---------------------------------------------------------------------------

static constexpr int HIST_BLOCK_SIZE = 256;

__global__ void sparse_topk_histogram_kernel(const int64_t* __restrict__ topk_ids,
                                             int32_t* __restrict__ expert_loads,
                                             const int num_entries,
                                             const int num_experts) {
    extern __shared__ int32_t smem_hist[];

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        smem_hist[i] = 0;
    }
    __syncthreads();

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_entries; idx += gridDim.x * blockDim.x) {
        int64_t eid = topk_ids[idx];
        if (eid >= 0 && eid < num_experts) {
            atomicAdd(&smem_hist[eid], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        if (smem_hist[i] > 0) {
            atomicAdd(&expert_loads[i], smem_hist[i]);
        }
    }
}

void topk_local_sum(const int64_t* topk_ids_ptr,
                    const int num_tokens,
                    const int top_k,
                    const int num_global_logical_experts,
                    int32_t* expert_loads_ptr,
                    cudaStream_t stream) {
    int num_entries = num_tokens * top_k;
    int L = num_global_logical_experts;

    CUDA_RUNTIME_CHECK(cudaMemsetAsync(expert_loads_ptr, 0, L * sizeof(int32_t), stream));

    if (num_entries > 0) {
        int smem_bytes = L * sizeof(int32_t);
        int num_blocks = min(256, (num_entries + HIST_BLOCK_SIZE - 1) / HIST_BLOCK_SIZE);

        CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
            sparse_topk_histogram_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

        sparse_topk_histogram_kernel<<<num_blocks, HIST_BLOCK_SIZE, smem_bytes, stream>>>(
            topk_ids_ptr, expert_loads_ptr, num_entries, L);
    }
}

}  // namespace ultra_ep::kernels