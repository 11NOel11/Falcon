#!/usr/bin/env python3
"""
Extract airplane images from different filter stages to represent epoch progression.
Maps filtering stages to training epochs for visualization.
"""

from PIL import Image
import sys

def extract_airplane_epochs():
    """Extract individual airplane images for each epoch."""

    # Load the full image
    img_path = '../public/fig_real_image_filtering.png'
    img = Image.open(img_path)

    width, height = img.size
    print(f"Original image size: {width}x{height}")

    # The image has 4 rows and 7 columns
    # We want row 0 (airplane)
    row_height = height // 4
    col_width = width // 7

    print(f"Each cell: {col_width}x{row_height}")

    # Row 0 is airplane (y=0 to y=row_height)
    airplane_row_y = 0

    # Define which columns to use for which epochs
    # Column layout: [Original, FFT, Retain95%, Retain75%, Retain50%, Removed95%, Removed50%]
    epoch_mappings = [
        (1, 4, 'airplane_epoch1.png'),   # Epoch 1: Retain 50% (heavily filtered, model confused)
        (5, 3, 'airplane_epoch5.png'),   # Epoch 5: Retain 75% (learning)
        (10, 2, 'airplane_epoch10.png'), # Epoch 10: Retain 95% (almost there)
        (20, 0, 'airplane_epoch20.png'), # Epoch 20: Original (learned)
        (40, 0, 'airplane_epoch40.png'), # Epoch 40: Original (mastered)
    ]

    for epoch, col_idx, filename in epoch_mappings:
        # Calculate crop box (left, upper, right, lower)
        left = col_idx * col_width
        upper = airplane_row_y
        right = left + col_width
        lower = airplane_row_y + row_height

        # Crop the region
        cropped = img.crop((left, upper, right, lower))

        # Manually crop to remove title/label whitespace
        # The actual image content is in the lower portion
        cell_height = cropped.height
        cell_width = cropped.width

        # Remove ~72% from top (title/whitespace) and keep bottom 20%
        # Just want the actual airplane image, no text
        content_top = int(cell_height * 0.72)
        content_bottom = int(cell_height * 0.98)
        # Add small horizontal margin
        content_left = int(cell_width * 0.05)
        content_right = int(cell_width * 0.95)

        cropped = cropped.crop((content_left, content_top, content_right, content_bottom))

        # Save
        output_path = f'../public/{filename}'
        cropped.save(output_path)
        print(f"✓ Saved epoch {epoch}: {output_path} (from column {col_idx}) - size: {cropped.size}")

    # Also save just the original airplane for reference
    left = 0
    upper = airplane_row_y
    right = col_width
    lower = airplane_row_y + row_height
    original = img.crop((left, upper, right, lower))

    # Apply same cropping
    cell_height = original.height
    cell_width = original.width
    content_top = int(cell_height * 0.72)
    content_bottom = int(cell_height * 0.98)
    content_left = int(cell_width * 0.05)
    content_right = int(cell_width * 0.95)
    original = original.crop((content_left, content_top, content_right, content_bottom))

    original.save('../public/airplane_original.png')
    print(f"✓ Saved original airplane: ../public/airplane_original.png - size: {original.size}")

    print("\n✅ All airplane epoch images extracted successfully!")
    print("\nMapping:")
    print("  Epoch 1  -> Heavily filtered (Retain 50%)")
    print("  Epoch 5  -> Medium filtered (Retain 75%)")
    print("  Epoch 10 -> Lightly filtered (Retain 95%)")
    print("  Epoch 20 -> Original (clear)")
    print("  Epoch 40 -> Original (clear)")

if __name__ == '__main__':
    extract_airplane_epochs()
