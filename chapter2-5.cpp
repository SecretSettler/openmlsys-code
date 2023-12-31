//custom_add.cpp
#include <torch/extension.h>

torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &custom_add, "A custom add function");
}
