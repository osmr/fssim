import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from PIL import Image
from torchvision.transforms import ToTensor


def load_image_as_tensor(image_path):
    """Load image and convert to PyTorch tensor"""
    img = Image.open(image_path).convert('RGB')  # Ensure RGB format
    tensor = ToTensor()(img)  # Convert to [C, H, W] tensor
    return tensor.unsqueeze(0)  # Add batch dimension


def calculate_ssim(img1_path, img2_path):
    """Compute SSIM between two images"""
    # Load images
    img1 = load_image_as_tensor(img1_path).cuda()
    img2 = load_image_as_tensor(img2_path).cuda()

    # Verify dimensions
    if img1.shape != img2.shape:
        raise ValueError(f"Image dimensions mismatch: {img1.shape} vs {img2.shape}")

    # Initialize SSIM metric
    ssim_metric = SSIM(
        data_range=1.0,  # Input range [0, 1]
        reduction='none'  # Return raw score
    )

    # Compute and return SSIM
    return ssim_metric(img1, img2).item()


if __name__ == "__main__":
    # image_path_1 = "../../fssim_data/test1a.png"
    # image_path_2 = "../../fssim_data/test2a.png"
    image_path_1 = "../../fssim_data/test1b.png"
    image_path_2 = "../../fssim_data/test2b.png"

    try:
        ssim_score = calculate_ssim(image_path_1, image_path_2)
        print(f"SSIM score: {ssim_score:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")
pass