#include "common.cuh"
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void fusedssimCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;  // batch index
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ float xconv[CONV_Y][CONV_X][5];

    // Each block processes B x C sub-batches. We loop over channels:
    for (int c = 0; c < CH; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads = BLOCK_X * BLOCK_Y;
            const int steps = (tileSize + threads - 1) / threads;

            const int tileStartY = block.group_index().y * BLOCK_Y;
            const int tileStartX = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / SHARED_X;
                    int local_x = tid % SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;

                    float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                    float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                    sTile[local_y][local_x][0] = X;
                    sTile[local_y][local_x][1] = Y;
                }
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;  // skip left halo

            float sumX   = 0.f;
            float sumX2  = 0.f;
            float sumY   = 0.f;
            float sumY2  = 0.f;
            float sumXY  = 0.f;

            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];

                sumX  += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX  += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY  += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
            }

            // Write out partial sums
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX   = 0.f; sumX2  = 0.f;
                sumY   = 0.f; sumY2  = 0.f;
                sumXY  = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft  = sTile[ly2][lx - d][0];
                    float Yleft  = sTile[ly2][lx - d][1];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];

                    sumX  += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY  += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX  += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY  += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            // center
            {
                float wC = cGauss[HALO];
                float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

                if (dm_dmu1) {
                    // partial derivatives
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    // Launch config
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Output SSIM map
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    fusedssimCUDA<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr
    );

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}
