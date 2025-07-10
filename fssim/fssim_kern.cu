#include <torch/extension.h>

__global__ void multiply_kernel(float* output, const float* input, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor multiply_by_two_cuda(const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 is supported");

    const int size = input.numel();

    if (size == 0) {
        return torch::empty_like(input);
    }

    torch::Tensor output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    multiply_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        size
    );

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}
