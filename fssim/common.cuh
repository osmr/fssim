#pragma once
#ifndef COMMON_CUH
#define COMMON_CUH

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
extern __constant__ float cGauss[11];

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
__device__ __forceinline__ float get_pix_value(const float* img, int b, int c, int y, int x, int CH, int H, int W);

#endif  // COMMON_CUH
