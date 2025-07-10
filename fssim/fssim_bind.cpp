#include <torch/extension.h>

torch::Tensor multiply_by_two_cuda(const torch::Tensor& input);

torch::Tensor multiply_by_two(const torch::Tensor& input) {
    return multiply_by_two_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply_by_two", &multiply_by_two, "Multiply tensor by two using CUDA");
}
