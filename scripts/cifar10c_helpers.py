"""
CIFAR-10C Helpers: High-frequency noise injection for robustness testing.
"""
import torch
import torch.nn.functional as F


def add_highfreq_noise(x: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """
    Add high-frequency noise to images for robustness testing.
    
    Args:
        x: Input tensor [B, C, H, W] in range [0, 1]
        sigma: Noise standard deviation
        
    Returns:
        Noisy tensor with high-frequency components emphasized
    """
    if sigma <= 0:
        return x
    
    # Add Gaussian noise
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # Apply Laplacian filter to emphasize high frequencies
    # Laplacian kernel (discrete approximation)
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=x.dtype, device=x.device)
    
    # Expand to match input channels
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    
    # Apply convolution
    highfreq = F.conv2d(x_noisy, laplacian_kernel, padding=1, groups=x.shape[1])
    
    # Mix original noisy image with high-frequency component
    x_out = x_noisy + 0.1 * highfreq
    
    # Clip to valid range
    x_out = torch.clamp(x_out, 0.0, 1.0)
    
    return x_out


def apply_corruption(x: torch.Tensor, corruption_type: str = "noise", severity: int = 3) -> torch.Tensor:
    """
    Apply CIFAR-10C style corruptions.
    
    Args:
        x: Input tensor [B, C, H, W] in range [0, 1]
        corruption_type: Type of corruption ("noise", "blur", "weather", etc.)
        severity: Corruption severity (1-5)
        
    Returns:
        Corrupted tensor
    """
    if corruption_type == "noise":
        sigma = 0.05 * severity
        return add_highfreq_noise(x, sigma)
    elif corruption_type == "gaussian_blur":
        # Simple Gaussian blur
        kernel_size = 3 + 2 * (severity - 1)
        return gaussian_blur(x, kernel_size)
    else:
        return x


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply Gaussian blur to images.
    
    Args:
        x: Input tensor [B, C, H, W]
        kernel_size: Size of Gaussian kernel (odd number)
        
    Returns:
        Blurred tensor
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    sigma = kernel_size / 6.0
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=x.device, dtype=x.dtype)
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel_2d = kernel_2d.repeat(x.shape[1], 1, 1, 1)
    
    # Apply convolution
    padding = kernel_size // 2
    x_blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])
    
    return x_blurred
