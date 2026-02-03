// Deprecated for remote memory management
// Use NVSHMEM symmetric heap instead

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/torch.h>

#include <cstring>

#include "exception.hpp"

namespace ultra_ep::ipc {

struct MemHandle {
    union MemHandleInner {
        cudaIpcMemHandle_t cuda_ipc_mem_handle;
        CUmemFabricHandle cu_mem_fabric_handle;
    } inner;
    size_t size;
    char src_hostname[256];
};

constexpr size_t HANDLE_SIZE = sizeof(MemHandle);

class IpcManager {
public:
    IpcManager();
    ~IpcManager();

    void malloc(void** ptr, size_t size_raw);

    void free(void* ptr);

    void get_handle(MemHandle* mem_handle, void* ptr);

    void open_handle(void** ptr, MemHandle* mem_handle);

    void close_handle(void* ptr);

    // Check if MNNVL fabric is supported on this device
    // User do not need to set fabric, the allocator will detect it automatically
    bool is_fabric_supported() const { return support_fabric_; }

    // Check if a memory handle is accessible from the current rank
    // @param mem_handle: The memory handle to check
    bool is_accessible(MemHandle* mem_handle);

    // @param process_group: The process group for the hybrid ep.
    // @return num_accessible_ranks: The number of accessible ranks.
    int detect_accessible_ranks(pybind11::object process_group);

private:
    bool support_fabric_ = false;
    size_t fabric_granularity_;
    CUdevice device_;
    CUmemAllocationProp fabric_prop_ = {};
    CUmemAccessDesc access_desc = {};
    char hostname_[256];

    // Test memory for accessing check.
    int* test_memory_ = nullptr;
    MemHandle test_mem_handle_;

    bool support_fabric();
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<IpcManager>(m, "IpcManager")
        .def(pybind11::init<>())
        .def("detect_accessible_ranks", &IpcManager::detect_accessible_ranks, py::arg("process_group"))
        .def("is_fabric_supported", &IpcManager::is_fabric_supported);
}

}  // namespace ultra_ep::ipc