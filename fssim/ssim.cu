#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;


// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float gauss_coefs[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float get_pix_value(const float* img, const int b, const int c, const int y, const int x,
                                               const int CH, const int H, const int W) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}


//-------------------------------------------------------------------------------------------


/**
 * @brief Forward Kernel: Fused SSIM.
 *
 * Fused SSIM Map Calculation (CUDA Kernel)
 *  - Two-pass convolution to get mu1, mu2, sigma1_sq, sigma2_sq, sigma12, etc.
 *  - Writes final SSIM map to ssim_map
 *  - Optionally writes partial derivatives to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
 */
__global__ void ssim_kernel(const int H,
                            const int W,
                            const int CH,
                            const float C1,
                            const float C2,
                            const float* __restrict__ img1,
                            const float* __restrict__ img2,
                            float* __restrict__ ssim_map,
                            float* __restrict__ dm_dmu1,
                            float* __restrict__ dm_dsigma1_sq,
                            float* __restrict__ dm_dsigma12) {
    auto block = cg::this_thread_block();
    const int bIdx   = block.group_index().z;  // batch index
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    __shared__ float s_tile[SHARED_Y][SHARED_X][2];
    // After horizontal pass, store partial sums here
    // x_conv[y][x] -> (sum_x, sum_x^2, sum_y, sum_y^2, sum_xY)
    __shared__ float x_conv[CONV_Y][CONV_X][5];

    // Each block processes B x C sub-batches. We loop over channels:
    for (int c = 0; c < CH; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tile_size = SHARED_Y * SHARED_X;
            const int threads = BLOCK_X * BLOCK_Y;
            const int steps = (tile_size + threads - 1) / threads;

            const int tile_start_y = block.group_index().y * BLOCK_Y;
            const int tile_start_x = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tile_size) {
                    const int local_y = tid / SHARED_X;
                    const int local_x = tid % SHARED_X;
                    const int gy = tile_start_y + local_y - HALO;
                    const int gx = tile_start_x + local_x - HALO;

                    const float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                    const float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                    s_tile[local_y][local_x][0] = X;
                    s_tile[local_y][local_x][1] = Y;
                }
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            const int ly = threadIdx.y;
            const int lx = threadIdx.x + HALO;  // skip left halo

            float sum_x   = 0.f;
            float sum_x2  = 0.f;
            float sum_y   = 0.f;
            float sum_y2  = 0.f;
            float sum_xY  = 0.f;

            // #pragma unroll for those 5 pairs
            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                const float w = gauss_coefs[HALO - d];
                const float x_left  = s_tile[ly][lx - d][0];
                const float y_left  = s_tile[ly][lx - d][1];
                const float x_right = s_tile[ly][lx + d][0];
                const float y_right = s_tile[ly][lx + d][1];

                sum_x  += (x_left + x_right) * w;
                sum_x2 += ((x_left * x_left) + (x_right * x_right)) * w;
                sum_y  += (y_left + y_right) * w;
                sum_y2 += ((y_left * y_left) + (y_right * y_right)) * w;
                sum_xY += ((x_left * y_left) + (x_right * y_right)) * w;
            }
            // center
            {
                const float center_x = s_tile[ly][lx][0];
                const float center_y = s_tile[ly][lx][1];
                const float wc = gauss_coefs[HALO];
                sum_x  += center_x * wc;
                sum_x2 += (center_x * center_x) * wc;
                sum_y  += center_y * wc;
                sum_y2 += (center_y * center_y) * wc;
                sum_xY += (center_x * center_y) * wc;
            }

            // Write out partial sums
            x_conv[ly][threadIdx.x][0] = sum_x;
            x_conv[ly][threadIdx.x][1] = sum_x2;
            x_conv[ly][threadIdx.x][2] = sum_y;
            x_conv[ly][threadIdx.x][3] = sum_y2;
            x_conv[ly][threadIdx.x][4] = sum_xY;

            // Possibly handle second row in same warp
            const int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sum_x   = 0.f; sum_x2  = 0.f;
                sum_y   = 0.f; sum_y2  = 0.f;
                sum_xY  = 0.f;

                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    const float w = gauss_coefs[HALO - d];
                    const float x_left  = s_tile[ly2][lx - d][0];
                    const float y_left  = s_tile[ly2][lx - d][1];
                    const float x_right = s_tile[ly2][lx + d][0];
                    const float y_right = s_tile[ly2][lx + d][1];

                    sum_x  += (x_left + x_right) * w;
                    sum_x2 += ((x_left * x_left) + (x_right * x_right)) * w;
                    sum_y  += (y_left + y_right) * w;
                    sum_y2 += ((y_left * y_left) + (y_right * y_right)) * w;
                    sum_xY += ((x_left * y_left) + (x_right * y_right)) * w;
                }
                // center
                {
                    const float cx = s_tile[ly2][lx][0];
                    const float cy = s_tile[ly2][lx][1];
                    const float wc = gauss_coefs[HALO];
                    sum_x  += cx * wc;
                    sum_x2 += (cx * cx) * wc;
                    sum_y  += cy * wc;
                    sum_y2 += (cy * cy) * wc;
                    sum_xY += (cx * cy) * wc;
                }
                x_conv[ly2][threadIdx.x][0] = sum_x;
                x_conv[ly2][threadIdx.x][1] = sum_x2;
                x_conv[ly2][threadIdx.x][2] = sum_y;
                x_conv[ly2][threadIdx.x][3] = sum_y2;
                x_conv[ly2][threadIdx.x][4] = sum_xY;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            const int ly = threadIdx.y + HALO;
            const int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

            #pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                const float w = gauss_coefs[HALO - d];
                const float* top = x_conv[ly - d][lx];
                const float* bot = x_conv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            // center
            {
                const float wC = gauss_coefs[HALO];
                const float* ctr = x_conv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                const float mu1 = out0;
                const float mu2 = out2;
                const float mu1_sq = mu1 * mu1;
                const float mu2_sq = mu2 * mu2;

                const float sigma1_sq = out1 - mu1_sq;
                const float sigma2_sq = out3 - mu2_sq;
                const float sigma12   = out4 - mu1 * mu2;

                const float A = mu1_sq + mu2_sq + C1;
                const float B = sigma1_sq + sigma2_sq + C2;
                const float C_ = 2.f * mu1 * mu2 + C1;
                const float D_ = 2.f * sigma12 + C2;

                const float val = (C_ * D_) / (A * B);

                const int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

                if (dm_dmu1) {
                    // partial derivatives
                    const float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    const float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    const float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

/**
 * @brief PyTorch Interface (Forward).
 *
 * PyTorch Interface for SSIM Map calculation (Forward pass)
 *   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
 *   If train=false, derivative Tensors are empty.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ssim_cuda(float C1,
                                                                                 float C2,
                                                                                 const torch::Tensor& img1,
                                                                                 const torch::Tensor& img2,
                                                                                 const bool train) {
    TORCH_CHECK(img1.device().is_cuda(), "Tensor img1 must be on CUDA device");
    TORCH_CHECK(img2.device().is_cuda(), "Tensor img2 must be on CUDA device");
    TORCH_CHECK(img1.get_device() == img2.get_device(), "Input tensors must be on the same device");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    const int B  = img1.size(0);
    const int CH = img1.size(1);
    const int H  = img1.size(2);
    const int W  = img1.size(3);

    // Launch config
    const dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                    (H + BLOCK_Y - 1) / BLOCK_Y,
                    B);
    const dim3 block(BLOCK_X, BLOCK_Y);

    // Output SSIM map
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    ssim_kernel<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr
    );

    const cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}


//-------------------------------------------------------------------------------------------


/**
 * @brief Backward Kernel: Fused SSIM.
 *
 * Backward pass for fused SSIM Map Calculation (CUDA Kernel): Apply chain rule to get dL/d(img1) from partial
 *    derivatives (dm_dmu1, dm_dsigma1_sq, dm_dsigma12) and dL/dmap (the gradient from above).
 */
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

/**
 * @brief PyTorch Interface (Backward).
 *
 * PyTorch Interface for SSIM Map calculation (Backward pass)
 *   Takes the gradient wrt the SSIM map and
 *   the partial derivatives from forward;
 *   returns dL/d(img1).
 */
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

    const cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return dL_dimg1;
}
