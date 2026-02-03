#include "api.cuh"
#include "config.cuh"

namespace ultra_ep::kernels {

__device__ __forceinline__ void memset_zero(void* __restrict__ ptr, size_t total_bytes) {
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(ptr);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    // Calculate alignment boundary (Target: 16-byte alignment for uint4)
    size_t align_offset = (16 - (addr & 15)) & 15;

    // Edge case: if total length is not enough to fill the aligned region, or just enough
    if (total_bytes <= align_offset) {
        for (size_t i = tid; i < total_bytes; i += stride) {
            base_ptr[i] = 0;
        }
        return;
    }

    // Divide into regions: Head (Bytes) | Body (Uint4) | Tail (Bytes)
    uint8_t* body_start_ptr = base_ptr + align_offset;
    size_t body_bytes = total_bytes - align_offset;
    size_t n_vectors = body_bytes / 16;  // number of 128-bit blocks
    size_t tail_offset = align_offset + n_vectors * 16;
    size_t tail_bytes = total_bytes - tail_offset;

    // Head: handle non-aligned bytes
    // Only Global Thread 0 handles at most 15 bytes, avoid parallel overhead for small data
    if (tid == 0 && align_offset > 0) {
        for (int i = 0; i < 15; ++i) {
            if (i < align_offset)
                base_ptr[i] = 0;
        }
    }

    // Body: 128-bit vectorized store
    uint4* vec_ptr = reinterpret_cast<uint4*>(body_start_ptr);
    for (size_t i = tid; i < n_vectors; i += stride) {
        // Store cache streaming with L2 bypassing
        // Equivalent to make_uint4(0, 0, 0, 0)
        ptx::st_global_v4_u32_streaming(vec_ptr + i, 0, 0, 0, 0);
    }

    // Tail: handle remaining bytes
    // Same as Head: Global Thread 0 handles at most 15 bytes
    if (tid == 0 && tail_bytes > 0) {
        for (int i = 0; i < 15; ++i) {
            if (i < tail_bytes)
                base_ptr[tail_offset + i] = 0;
        }
    }
}

__global__ void grad_reduce_kernel(const int total_tasks,
                                   const GradReduceTask* grad_reduce_tasks,
                                   int* global_task_counter) {
    extern __shared__ char smem_buffer[];

    // Shared memory mbarriers for TMA pipelining (one per pipeline stage)
    __shared__ __align__(8) ptx::mbarrier mbarriers[GRAD_REDUCE_PIPELINE_STAGES];
    __shared__ ptx::arrival_phase phases[GRAD_REDUCE_PIPELINE_STAGES];

    // Initialize mbarriers (thread 0 only, once per block)
    if (threadIdx.x == 0) {
        for (int i = 0; i < GRAD_REDUCE_PIPELINE_STAGES; ++i) {
            ptx::mbarrier_init(&mbarriers[i], 1);  // Expect 1 arrival (from TMA completion)
            phases[i] = 0;
        }
    }
    __syncthreads();

    float* smem_pool = reinterpret_cast<float*>(smem_buffer);

    // Persistent loop: fetch tasks until the task list is empty
    while (true) {
        // Fetch task (leader thread only)
        __shared__ int my_task_idx;
        if (threadIdx.x == 0) {
            my_task_idx = atomicAdd(global_task_counter, 1);
        }
        __syncthreads();
        if (my_task_idx >= total_tasks)
            break;

        GradReduceTask current_task = grad_reduce_tasks[my_task_idx];
        size_t task_numel = current_task.numel;
        size_t num_tiles = (task_numel + GRAD_REDUCE_TILE_ELEMENTS - 1) / GRAD_REDUCE_TILE_ELEMENTS;

        // Prologue: issue first GRAD_REDUCE_PIPELINE_STAGES TMA loads
        for (int i = 0; i < GRAD_REDUCE_PIPELINE_STAGES; ++i) {
            if (i < num_tiles) {
                size_t offset = i * GRAD_REDUCE_TILE_ELEMENTS;
                // Round up to 16-byte alignment for TMA
                size_t bytes = min((size_t)GRAD_REDUCE_TILE_SIZE_BYTES, (task_numel - offset) * sizeof(float));
                bytes = (bytes + 15) & ~15;  // Align to 16 bytes

                int stage = i % GRAD_REDUCE_PIPELINE_STAGES;
                // Thread 0: set expected TX bytes and issue TMA load
                if (threadIdx.x == 0) {
                    ptx::mbarrier_arrive_and_set_tx(&mbarriers[stage], bytes);
                    ptx::tma_load_1d(smem_pool + stage * GRAD_REDUCE_TILE_ELEMENTS,
                                     current_task.replica_remote_addr + offset,
                                     &mbarriers[stage],
                                     bytes,
                                     ptx::TMACacheHint::kEvictFirst);
                }
            }
        }

        for (int step = 0; step < num_tiles; ++step) {
            int stage = step % GRAD_REDUCE_PIPELINE_STAGES;

            // Wait for TMA to complete on this stage
            if (threadIdx.x == 0) {
                ptx::mbarrier_wait_and_flip_phase(&mbarriers[stage], phases[stage]);
            }
            __syncthreads();

            size_t offset = step * GRAD_REDUCE_TILE_ELEMENTS;
            int count = min(GRAD_REDUCE_TILE_ELEMENTS, (int)(task_numel - offset));
            float* curr_smem = smem_pool + stage * GRAD_REDUCE_TILE_ELEMENTS;

            // Reduce with AtomicAdd for multiple replicas added to same master
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                atomicAdd(&current_task.master_local_addr[offset + i], curr_smem[i]);
            }

            // Zero-out remote replica grad buffer (pipelined)
            __syncthreads();  // Sync read
            memset_zero(current_task.replica_remote_addr + offset, count * sizeof(float));
            __threadfence_system();  // Sync write

            // Issue next TMA load (reuse this stage's mbarrier)
            int next_step = step + GRAD_REDUCE_PIPELINE_STAGES;
            if (next_step < num_tiles) {
                size_t next_offset = next_step * GRAD_REDUCE_TILE_ELEMENTS;
                // Round up to 16-byte alignment for TMA
                size_t bytes = min((size_t)GRAD_REDUCE_TILE_SIZE_BYTES, (task_numel - next_offset) * sizeof(float));
                bytes = (bytes + 15) & ~15;  // Align to 16 bytes

                // Thread 0: set expected TX bytes and issue TMA load
                if (threadIdx.x == 0) {
                    ptx::mbarrier_arrive_and_set_tx(&mbarriers[stage], bytes);
                    ptx::tma_load_1d(smem_pool + stage * GRAD_REDUCE_TILE_ELEMENTS,
                                     current_task.replica_remote_addr + next_offset,
                                     &mbarriers[stage],
                                     bytes,
                                     ptx::TMACacheHint::kEvictFirst);
                }
            }
        }
    }
}

void run_grad_reduce(const GradReduceTask* grad_reduce_tasks_cpu,
                     GradReduceTask* grad_reduce_tasks_gpu,
                     int* global_task_counter_gpu,
                     const int total_tasks,
                     cudaStream_t stream,
                     const int num_device_sms) {
    // Copy tasks from CPU to GPU to avoid kernel param overflow
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(grad_reduce_tasks_gpu,
                                       grad_reduce_tasks_cpu,
                                       total_tasks * sizeof(GradReduceTask),
                                       cudaMemcpyHostToDevice,
                                       stream));
    CUDA_RUNTIME_CHECK(cudaMemsetAsync(global_task_counter_gpu, 0, sizeof(int), stream));

    // Call device-side kernel
    dim3 grid(num_device_sms * 2);
    dim3 block(256);
    int smem_size = GRAD_REDUCE_TILE_SIZE_BYTES * GRAD_REDUCE_PIPELINE_STAGES;
    CUDA_RUNTIME_CHECK(
        cudaFuncSetAttribute(grad_reduce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    grad_reduce_kernel<<<grid, block, smem_size, stream>>>(total_tasks, grad_reduce_tasks_gpu, global_task_counter_gpu);
}

}  // namespace ultra_ep::kernels