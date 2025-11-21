#!/usr/bin/env python3
"""
Generate individual airplane images with FALCON frequency filtering at different levels.
Creates clean, centered images for each epoch without cropping from composite figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import urllib.request

def load_airplane_sample():
    """Load a CIFAR-10 airplane image."""
    # Try to use existing airplane from the full figure
    full_img_path = Path('../public/fig_real_image_filtering.png')

    if full_img_path.exists():
        print("Extracting airplane from existing figure...")
        img = Image.open(full_img_path)
        width, height = img.size

        # The image has 4 rows and 7 columns
        row_height = height // 4
        col_width = width // 7

        # Extract just the original airplane (row 0, column 0)
        # Zoom out and position - remove all text
        left = int(col_width * 0.08)
        top = int(row_height * 0.68)  # Move down more to skip title
        right = int(col_width * 0.92)
        bottom = int(row_height * 0.96)

        airplane = img.crop((left, top, right, bottom))
        return np.array(airplane.convert('RGB'))

    return None

def apply_frequency_filter(image, retain_energy=0.95):
    """
    Apply FALCON-style frequency filtering to an image.

    Args:
        image: numpy array (H, W, 3) in range [0, 255]
        retain_energy: fraction of energy to keep (0.95 = 95%)

    Returns:
        filtered_image
    """
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0

    # Process each channel separately
    filtered_channels = []

    for c in range(3):
        channel = img_float[:, :, c]

        # Apply FFT
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        # Compute energy
        energy = magnitude ** 2
        total_energy = np.sum(energy)

        # Sort by energy and find threshold
        sorted_energies = np.sort(energy.flatten())[::-1]
        cumsum = np.cumsum(sorted_energies)
        threshold_idx = np.where(cumsum >= retain_energy * total_energy)[0][0]
        threshold = sorted_energies[threshold_idx]

        # Create mask
        mask = (energy >= threshold).astype(float)

        # Apply mask
        fft_filtered = fft_shifted * mask

        # Inverse FFT
        filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered))

        filtered_channels.append(np.real(filtered))

    # Stack channels
    filtered_image = np.stack(filtered_channels, axis=2)

    # Clip to valid range
    filtered_image = np.clip(filtered_image, 0, 1)

    return filtered_image

def create_airplane_epoch_images():
    """Create individual airplane images for each epoch."""
    print("Loading CIFAR-10 airplane sample...")
    airplane = load_airplane_sample()

    if airplane is None:
        print("Error: Could not load airplane image")
        return

    print(f"Loaded airplane image: {airplane.shape}")

    # Define epoch mappings with filtering levels
    epoch_configs = [
        (1, 0.50, 'airplane_epoch1.png'),   # Epoch 1: Heavy filtering
        (5, 0.75, 'airplane_epoch5.png'),   # Epoch 5: Medium filtering
        (10, 0.95, 'airplane_epoch10.png'), # Epoch 10: Light filtering
        (20, 0.98, 'airplane_epoch20.png'), # Epoch 20: Nearly original (98%)
        (40, 1.00, 'airplane_epoch40.png'), # Epoch 40: Fully converged (100%)
    ]

    output_dir = Path('../public')
    output_dir.mkdir(exist_ok=True)

    for epoch, retain, filename in epoch_configs:
        print(f"\nProcessing epoch {epoch} (retain {retain*100}%)...")

        if retain >= 1.0:
            # No filtering, use original
            filtered = airplane.astype(np.float32) / 255.0
        else:
            # Apply filtering
            filtered = apply_frequency_filter(airplane, retain_energy=retain)

        # Create figure with rectangular aspect ratio to fit better
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.imshow(filtered)
        ax.axis('off')

        # Remove all whitespace/padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save
        output_path = output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        print(f"✓ Saved: {output_path}")

    # Also save the original
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.imshow(airplane)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = output_dir / 'airplane_original.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_path}")

    print("\n✅ All airplane epoch images generated successfully!")
    print("\nEpoch progression:")
    print("  Epoch 1  -> 50% energy retained (heavily filtered)")
    print("  Epoch 5  -> 75% energy retained (medium filtering)")
    print("  Epoch 10 -> 95% energy retained (light filtering)")
    print("  Epoch 20 -> 98% energy retained (nearly converged)")
    print("  Epoch 40 -> 100% original (fully converged)")

if __name__ == '__main__':
    print("=" * 70)
    print("Generating Clean Airplane Epoch Images")
    print("=" * 70)
    create_airplane_epoch_images()
