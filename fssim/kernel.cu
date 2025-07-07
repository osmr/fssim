#include <torch/extension.h>

__global__ void multiply_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

torch::Tensor multiply_by_two_cuda(torch::Tensor input) {
    auto output = input.contiguous();  // важно для .data_ptr()
    int size = output.numel();
    float* data = output.data_ptr<float>();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    multiply_kernel<<<blocks, threads>>>(data, size);
    return output;
}
