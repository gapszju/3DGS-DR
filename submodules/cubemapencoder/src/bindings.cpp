#include <torch/extension.h>

#include "cubemapencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cubemap_encode_forward", &cubemap_encode_forward, "cubemap encode forward (CUDA)");
    m.def("cubemap_encode_backward", &cubemap_encode_backward, "cubemap encode backward (CUDA)");
}

