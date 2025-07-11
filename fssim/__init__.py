"""
    Fused Differentiable SSIM.
"""

__version__ = "0.0.1"

import torch
from .fssim_cuda import ssim, ssim_backward


class SsimMap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = ssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None


def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         padding: str = "same",
         train: bool = True):
    assert padding in ["same", "valid"]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.contiguous()
    map = SsimMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()
