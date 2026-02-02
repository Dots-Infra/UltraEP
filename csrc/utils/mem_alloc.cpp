// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "mem_alloc.hpp"

namespace ultra_ep::ipc {

// Round-up allocation size to fabric granularity.
size_t inline get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

RemoteMemAllocator::RemoteMemAllocator() {
    this->support_fabric_ = support_fabric();
    if (gethostname(hostname_, sizeof(hostname_)) != 0) {
        perror("gethostname");
        std::snprintf(hostname_, sizeof(hostname_), "unknown");
    }

    // It seems a dummy call to set the device. but it is useful to prevent the invalid device context error in gb..
    int device_id = -1;
    CUDA_RUNTIME_CHECK(cudaGetDevice(&device_id));
    CUDA_RUNTIME_CHECK(cudaSetDevice(device_id));

    if (this->support_fabric_) {
        // Get the device context.
        CUDA_DRIVER_CHECK(cuCtxGetDevice(&device_));
        fabric_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        fabric_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        fabric_prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        fabric_prop_.location.id = device_;
        CUDA_DRIVER_CHECK(cuMemGetAllocationGranularity(&fabric_granularity_, &fabric_prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Test the fabric support
    // Somtimes support_fabric() returns true, but the fabric can not be used.
    if (this->support_fabric_) {
        size_t size = get_size_align_to_granularity(128, fabric_granularity_);
        CUmemGenericAllocationHandle handle;
        if (CUDA_SUCCESS != cuMemCreate(&handle, size, &fabric_prop_, 0)) {
            this->support_fabric_ = false;
        } else {
            cuMemRelease(handle);
        }
        cudaGetLastError();  // Clear the last error
    }

    this->malloc((void**)&test_memory_, 128 * sizeof(int));
    this->get_handle(&test_mem_handle_, test_memory_);
}

RemoteMemAllocator::~RemoteMemAllocator() {
    this->free((void*)test_memory_);
    test_memory_ = nullptr;
}

// Check if the current device supports fabric.
bool RemoteMemAllocator::support_fabric() {
    int device_count;
    CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&device_count));

    for (int device = 0; device < device_count; ++device) {
        int support = 0;
        CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
        if (!support) {
            return false;
        }
    }
    return true;
}

void RemoteMemAllocator::malloc(void** ptr, size_t size_raw) {
    if (support_fabric_) {
        size_t size = get_size_align_to_granularity(size_raw, fabric_granularity_);
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemCreate(&handle, size, &fabric_prop_, 0));
        CUDA_DRIVER_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, fabric_granularity_, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        CUDA_DRIVER_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
    } else {
        CUDA_RUNTIME_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void RemoteMemAllocator::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
        size_t size = 0;
        CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
        CUDA_DRIVER_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemRelease(handle));
    } else {
        CUDA_RUNTIME_CHECK(cudaFree(ptr));
    }
}

void RemoteMemAllocator::malloc_pinned(void** ptr, size_t size_raw) {
    CUDA_RUNTIME_CHECK(cudaMallocHost(ptr, size_raw));
}

void RemoteMemAllocator::free_pinned(void* ptr) {
    CUDA_RUNTIME_CHECK(cudaFreeHost(ptr));
}

void RemoteMemAllocator::get_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
        CUDA_DRIVER_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }

    // Record the source hostname
    strncpy(mem_handle->src_hostname, hostname_, sizeof(mem_handle->src_hostname));
}

void RemoteMemAllocator::open_handle(void** ptr, MemHandle* mem_handle) {
    if (support_fabric_) {
        size_t size = mem_handle->size;
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
        CUDA_DRIVER_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        CUDA_DRIVER_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void RemoteMemAllocator::close_handle(void* ptr) {
    if (support_fabric_) {
        size_t size = 0;
        CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
        CUDA_DRIVER_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    } else {
        CUDA_RUNTIME_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}

bool RemoteMemAllocator::is_accessible(MemHandle* mem_handle) {
    bool accessible = false;
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        auto ret = cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC);
        accessible = ret == CUDA_SUCCESS;
        if (accessible) {
            cuMemRelease(handle);
        } else {
            if (ret != CUDA_SUCCESS) {
                const char* errStr;
                cuGetErrorString(ret, &errStr);
                fprintf(stderr, "[Error] Failed to import the fabric handle: %s\n", errStr);
                fflush(stderr);
            }
        }
    } else {
        // Check if the source hostname is the same as the current hostname
        accessible = strncmp(mem_handle->src_hostname, hostname_, sizeof(hostname_)) == 0;
    }
    return accessible;
}

int RemoteMemAllocator::detect_accessible_ranks(pybind11::object process_group) {
    auto torch_distributed = py::module_::import("torch.distributed");
    int world_size = process_group.attr("size")().cast<int>();
    int current_rank = process_group.attr("rank")().cast<int>();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Put the test memory handle on a CUDA tensor
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor test_tensor = torch::empty({static_cast<long>(sizeof(MemHandle))}, opts);
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(test_tensor.data_ptr(), &test_mem_handle_, sizeof(MemHandle), cudaMemcpyHostToDevice, stream));

    // All gather the test memory
    py::list test_handle_list;
    for (int i = 0; i < world_size; i++) {
        test_handle_list.append(torch::empty_like(test_tensor));
    }
    torch_distributed.attr("all_gather")(test_handle_list, test_tensor, process_group);

    // Check if the test memory is accessible on each rank
    int num_accessible_ranks = 1;  // include the current rank
    for (int i = 0; i < world_size; i++) {
        if (i != current_rank) {
            MemHandle test_handle;
            torch::Tensor gathered = test_handle_list[i].cast<torch::Tensor>();
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(&test_handle, gathered.data_ptr(), sizeof(MemHandle), cudaMemcpyDeviceToHost, stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
            if (is_accessible(&test_handle)) {
                num_accessible_ranks++;
            }
        }
    }

    return num_accessible_ranks;
}

}  // namespace ultra_ep::ipc