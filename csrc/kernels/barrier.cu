#include <cuda/atomic>

#include "../utils/exception.cuh"
#include "api.cuh"
#include "launch.cuh"

namespace ultra_ep::kernels {

// Block barriers transfer all load/store ordering to/from the signal threads.
__device__ void nvl_domain_barrier(int32_t** peers, int rank, int G, int offset) {
    __syncthreads();
    for (int peer = threadIdx.x; peer < G; peer += blockDim.x) {
        if (peer == rank)
            continue;
        cuda::atomic_ref<int32_t, cuda::thread_scope_system> send(peers[peer][offset + rank]);
        send.store(1, cuda::memory_order_release);
        cuda::atomic_ref<int32_t, cuda::thread_scope_system> recv(peers[rank][offset + peer]);
        while (recv.load(cuda::memory_order_acquire) == 0) {
        }
        recv.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();
}

__global__ void nvl_barrier_kernel(int32_t** peers, int rank, int G, int offset) {
    nvl_domain_barrier(peers, rank, G, offset);
}

void nvl_barrier(int32_t** peers, int rank, int G, int channel, cudaStream_t stream) {
    EP_HOST_ASSERT(channel >= 0 && channel < kNumLifetimeBarrierChannels);
    const int threads = align_up(G, 32);
    launch_kernel(nvl_barrier_kernel, make_launch_config(1, threads, stream), peers, rank, G, channel * G);
}

}  // namespace ultra_ep::kernels
