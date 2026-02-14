#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "runtime.hpp"
#include "solvers/api.hpp"
#include "ultra_ep.hpp"
#include "utils/ipc_manager.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

namespace ultra_ep {}  // namespace ultra_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "UltraEP: an efficient expert-parallel load balancing library";

    // Register UltraEP APIs
    ultra_ep::register_apis(m);
    ultra_ep::runtime::register_apis(m);
    ultra_ep::ipc::register_apis(m);
    ultra_ep::event::register_apis(m);
    ultra_ep::solver::register_apis(m);
}
