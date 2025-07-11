#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ssim_cuda(const float C1,
                                                                                 const float C2,
                                                                                 const torch::Tensor& img1,
                                                                                 const torch::Tensor& img2,
                                                                                 const bool train);

torch::Tensor ssim_backward_cuda(const float C1,
                                 const float C2,
                                 const torch::Tensor& img1,
                                 const torch::Tensor& img2,
                                 const torch::Tensor& dL_dmap,
                                 const torch::Tensor& dm_dmu1,
                                 const torch::Tensor& dm_dsigma1_sq,
                                 const torch::Tensor& dm_dsigma12);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ssim", &ssim_cuda, "SSIM calculation using CUDA");
  m.def("ssim_backward", &ssim_backward_cuda, "SSIM backward calculation using CUDA");
}
