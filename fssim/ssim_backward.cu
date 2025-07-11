#include "common.cuh"
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------------------------------------------------
// Backward Kernel: Apply chain rule to get dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12) and dL/dmap (the gradient from above).
// ------------------------------------------------------------------------------------
__global__ void ssim_backward_kernel(const int H,
                                     const int W,
                                     const int CH,
                                     const float C1,
                                     const float C2,
                                     const float* __restrict__ img1,
                                     const float* __restrict__ img2,
                                     const float* __restrict__ dL_dmap,
                                     float* __restrict__ dL_dimg1,
                                     const float* __restrict__ dm_dmu1,
                                     const float* __restrict__ dm_dsigma1_sq,
                                     const float* __restrict__ dm_dsigma12) {
    auto block = cg::this_thread_block();
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;
    const int bIdx   = block.group_index().z;

    // Shared memory for the fused data:
    // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
    __shared__ float s_data[3][SHARED_Y][SHARED_X];
    __shared__ float s_scratch[CONV_Y][CONV_X][3];

    for (int c = 0; c < CH; ++c) {
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H) {
            p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // (1) Load + fuse multiplication
        {
            const int start_y = block.group_index().y * BLOCK_Y;
            const int start_x = block.group_index().x * BLOCK_X;

            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            const int totalThreads = BLOCK_X * BLOCK_Y;
            const int num_warps = (totalThreads + 31) / 32;

            for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                const int gy = start_y + row - HALO;
                for (int col = lane_id; col < SHARED_X; col += 32) {
                    const int gx = start_x + col - HALO;

                    const float chain = get_pix_value(dL_dmap,      bIdx, c, gy, gx, CH, H, W);
                    const float vmu   = get_pix_value(dm_dmu1,      bIdx, c, gy, gx, CH, H, W);
                    const float vs1   = get_pix_value(dm_dsigma1_sq,bIdx, c, gy, gx, CH, H, W);
                    const float vs12  = get_pix_value(dm_dsigma12,  bIdx, c, gy, gx, CH, H, W);

                    s_data[0][row][col] = vmu  * chain;
                    s_data[1][row][col] = vs1  * chain;
                    s_data[2][row][col] = vs12 * chain;
                }
            }
        }
        block.sync();

        // (2) Horizontal pass
        {
            const int ly = threadIdx.y;
            const int lx = threadIdx.x + HALO;

            for (int pass = 0; pass < 2; ++pass) {
                const int yy = ly + pass * BLOCK_Y;
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;

                    #pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        const float w = gauss_coefs[HALO - d];
                        const float left0  = s_data[0][yy][lx - d];
                        const float left1  = s_data[1][yy][lx - d];
                        const float left2  = s_data[2][yy][lx - d];

                        const float right0 = s_data[0][yy][lx + d];
                        const float right1 = s_data[1][yy][lx + d];
                        const float right2 = s_data[2][yy][lx + d];

                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    // center
                    {
                        const float wc = gauss_coefs[HALO];
                        const float c0 = s_data[0][yy][lx];
                        const float c1 = s_data[1][yy][lx];
                        const float c2 = s_data[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }

                    s_scratch[yy][threadIdx.x][0] = accum0;
                    s_scratch[yy][threadIdx.x][1] = accum1;
                    s_scratch[yy][threadIdx.x][2] = accum2;
                }
            }
        }
        block.sync();

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            const int ly = threadIdx.y + HALO;
            const int lx = threadIdx.x;

            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                const float w = gauss_coefs[HALO - d];
                const float* top = s_scratch[ly - d][lx];
                const float* bot = s_scratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                const float wc = gauss_coefs[HALO];
                const float* ctr = s_scratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            // final accumulation
            const float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

            const int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }
        block.sync();
    }
}

// ------------------------------------------------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------------------------------------------------
torch::Tensor ssim_backward_cuda(const float C1,
                                 const float C2,
                                 const torch::Tensor& img1,
                                 const torch::Tensor& img2,
                                 const torch::Tensor& dL_dmap,
                                 const torch::Tensor& dm_dmu1,
                                 const torch::Tensor& dm_dsigma1_sq,
                                 const torch::Tensor& dm_dsigma12) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const int B  = img1.size(0);
    const int CH = img1.size(1);
    const int H  = img1.size(2);
    const int W  = img1.size(3);

    auto dL_dimg1 = torch::zeros_like(img1);

    const dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                    (H + BLOCK_Y - 1) / BLOCK_Y,
                    B);
    const dim3 block(BLOCK_X, BLOCK_Y);

    ssim_backward_kernel<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>()
    );

    return dL_dimg1;
}