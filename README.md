# Fused Differentiable SSIM

Refactored copy of project https://github.com/rahul-goel/fused-ssim.

## Prerequisites

- python >= 3.10.
- cuda >= 11.8.
- torch.


## Installation

```bash
pip install git+https://github.com/osmr/fused_ssim
```
or
```bash
git clone git@github.com:osmr/fused_ssim.git
pip install ./fused_ssim/
```

## Usage

For training:
```python
import torch
from fused_ssim import fused_ssim

gt_image = torch.rand(2, 3, 1080, 1920)
predicted_image = torch.nn.Parameter(torch.rand_like(gt_image))
ssim_value = fused_ssim(predicted_image, gt_image)
```

For inference:
```python
with torch.no_grad():
  ssim_value = fused_ssim(predicted_image, gt_image, train=False)
```

## Constraints

- Only one of the images is allowed to be differentiable i.e. only the first image can be `nn.Parameter`.
- Limited to 2D images.
- Images must be normalized to range `[0, 1]`.
- Standard `11x11` convolutions supported.
