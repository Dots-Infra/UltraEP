#pragma once

#include <torch/extension.h>

#include <vector>

namespace ultra_ep {

/**
 * @brief Create a torch::Tensor from a raw buffer pointer.
 *
 * @param ptr The raw pointer to the memory buffer.
 * @param shape The desired shape of the tensor.
 * @param dtype The scalar type of the tensor.
 * @param device The device where the buffer resides.
 * @return torch::Tensor A tensor sharing the same memory as the buffer.
 */
inline torch::Tensor make_tensor_from_buffer(void* ptr,
                                             const std::vector<int64_t>& shape,
                                             torch::ScalarType dtype,
                                             const torch::Device& device) {
    if (ptr == nullptr) {
        return torch::Tensor();
    }
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    return torch::from_blob(ptr, shape, options);
}

}  // namespace ultra_ep
