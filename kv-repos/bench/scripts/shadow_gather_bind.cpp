#include <torch/extension.h>
#include "functions.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets);
    m.def("gather_copy_d2d_with_offsets", &gather_copy_d2d_with_offsets);
    m.def("gather_copy_with_offsets", &gather_copy_with_offsets);
}
