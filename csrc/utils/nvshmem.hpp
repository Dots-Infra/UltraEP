#pragma once

#include <device_host_transport/nvshmem_common_ibgda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstring>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#include <optional>
#include <vector>

#include "exception.hpp"

namespace ultra_ep::nvshmem {

inline nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
inline nvshmem_team_config_t cpu_rdma_team_config;

inline std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

inline void* alloc(const size_t& size, const size_t& alignment) {
    return nvshmem_align(alignment, size);
}

inline void free(void* ptr) {
    nvshmem_free(ptr);
}

// Get pointer to symmetric memory on a remote PE
// Returns nullptr if the remote PE is not in the same NVL domain (e.g., RDMA)
inline void* ptr(void* local_ptr, int pe) {
    return nvshmem_ptr(local_ptr, pe);
}

inline void barrier(const bool with_cpu_sync = false, const std::optional<cudaStream_t>& stream_opt = std::nullopt) {
    // Wait all streams to finish on this GPU
    if (with_cpu_sync)
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // NOTES: this only launches kernels at GPU
    if (stream_opt.has_value()) {
        nvshmemx_barrier_all_on_stream(stream_opt.value());
    } else {
        nvshmem_barrier_all();
    }

    // Let CPU wait
    if (with_cpu_sync)
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

inline int init(const std::vector<uint8_t>& root_unique_id_val, const int& rank, const int& num_ranks, const int& team_split_stride) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    if (team_split_stride > 0 and num_ranks > team_split_stride) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % team_split_stride == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % team_split_stride,
                                                  team_split_stride,
                                                  num_ranks / team_split_stride,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    // Wait all GPUs to get ready
    barrier(true);
    return nvshmem_my_pe();
}

inline void finalize() {
    barrier(true);
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}

}  // namespace ultra_ep::nvshmem