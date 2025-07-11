#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ssim_cuda(float C1,
                                                                                 float C2,
                                                                                 torch::Tensor& img1,
                                                                                 torch::Tensor& img2,
                                                                                 bool train);

torch::Tensor ssim_backward_cuda(float C1,
                                 float C2,
                                 torch::Tensor& img1,
                                 torch::Tensor& img2,
                                 torch::Tensor& dL_dmap,
                                 torch::Tensor& dm_dmu1,
                                 torch::Tensor& dm_dsigma1_sq,
                                 torch::Tensor& dm_dsigma12);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ssim", &ssim_cuda, "SSIM calculation using CUDA");
  m.def("ssim_backward", &ssim_backward_cuda, "SSIM backward calculation using CUDA");
}
