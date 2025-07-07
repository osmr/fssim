#include <torch/extension.h>

torch::Tensor multiply_by_two_cuda(torch::Tensor input);

torch::Tensor multiply_by_two(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 is supported");
    return multiply_by_two_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply_by_two", &multiply_by_two, "Multiply tensor by two using CUDA");
}
