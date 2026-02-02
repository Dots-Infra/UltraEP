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

/**
 * @brief Check if two ranks are in the same NVL domain.
 *
 * @param rank1 The rank of the first rank.
 * @param rank2 The rank of the second rank.
 * @param max_nvl_peers The maximum number of NVL peers.
 * @return bool True if the two ranks are in the same NVL domain, false otherwise.
 */
inline bool is_in_same_nvl_domain(int rank1, int rank2, int max_nvl_peers) {
    return rank1 / max_nvl_peers == rank2 / max_nvl_peers;
}

}  // namespace ultra_ep
