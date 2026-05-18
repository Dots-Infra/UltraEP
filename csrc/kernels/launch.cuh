#pragma once

#include <cuda_runtime.h>

#include <tuple>
#include <type_traits>
#include <utility>

#include "../utils/exception.cuh"

namespace ultra_ep::kernels {

struct LaunchConfig {
    dim3 grid = dim3(1, 1, 1);
    dim3 block = dim3(1, 1, 1);
    size_t shared_memory_bytes = 0;
    cudaStream_t stream = nullptr;
};

inline LaunchConfig make_launch_config(const dim3 grid,
                                       const dim3 block,
                                       cudaStream_t stream,
                                       const size_t shared_memory_bytes = 0) {
    return LaunchConfig{grid, block, shared_memory_bytes, stream};
}

inline int clamp_num_ctas(const int requested_ctas, const int max_work_items) {
    if (max_work_items <= 0) {
        return 0;
    }
    const int capped_ctas = requested_ctas < max_work_items ? requested_ctas : max_work_items;
    return capped_ctas > 0 ? capped_ctas : 1;
}

template <typename Kernel>
inline void maybe_set_dynamic_shared_memory(Kernel kernel, const size_t shared_memory_bytes) {
    if (shared_memory_bytes == 0) {
        return;
    }
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_bytes)));
}

template <typename Kernel, typename Tuple, size_t... Indices>
inline void launch_kernel_impl(Kernel kernel,
                               const LaunchConfig& config,
                               Tuple& stored_args,
                               std::index_sequence<Indices...>) {
    void* raw_args[] = {static_cast<void*>(&std::get<Indices>(stored_args))...};
    CUDA_RUNTIME_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
                                        config.grid,
                                        config.block,
                                        raw_args,
                                        config.shared_memory_bytes,
                                        config.stream));
}

template <typename Kernel, typename... Args>
inline void launch_kernel(Kernel kernel, const LaunchConfig& config, Args&&... args) {
    maybe_set_dynamic_shared_memory(kernel, config.shared_memory_bytes);

    if constexpr (sizeof...(Args) == 0) {
        CUDA_RUNTIME_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
                                            config.grid,
                                            config.block,
                                            nullptr,
                                            config.shared_memory_bytes,
                                            config.stream));
    } else {
        auto stored_args = std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
        launch_kernel_impl(kernel, config, stored_args, std::index_sequence_for<Args...>{});
    }
}

}  // namespace ultra_ep::kernels
