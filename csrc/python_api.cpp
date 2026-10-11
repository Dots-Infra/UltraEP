#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "runtime.hpp"
#include "ultra_ep.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "UltraEP: an efficient expert-parallel load balancing library";

    // Register UltraEP APIs
    ultra_ep::register_apis(m);
    ultra_ep::runtime::register_apis(m);
    ultra_ep::event::register_apis(m);
}
