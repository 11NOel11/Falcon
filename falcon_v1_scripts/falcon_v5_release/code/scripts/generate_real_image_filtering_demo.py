"""
Generate visualization showing FALCON's frequency filtering on real CIFAR-10 images.
This demonstrates the actual effect of the optimizer on real training data.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def load_cifar10_samples():
    """Load a few representative CIFAR-10 images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    
    # Select diverse samples (one from each category)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    samples = []
    
    # Get one sample from each class
    for class_idx in range(10):
        # Find first image of this class
        for i, (img, label) in enumerate(dataset):
            if label == class_idx:
                samples.append((np.array(img), class_names[class_idx]))
                break
    
    return samples[:4]  # Use first 4 for visualization


def apply_frequency_filter(image, retain_energy=0.95):
    """
    Apply FALCON-style frequency filtering to an image.
    
    Args:
        image: numpy array (H, W, 3) in range [0, 255]
        retain_energy: fraction of energy to keep (0.95 = 95%)
    
    Returns:
        filtered_image, removed_components, mask, fft_magnitude
    """
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0
    
    # Process each channel separately
    filtered_channels = []
    removed_channels = []
    masks = []
    fft_mags = []
    
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
        fft_removed = fft_shifted * (1 - mask)
        
        # Inverse FFT
        filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
        removed = np.fft.ifft2(np.fft.ifftshift(fft_removed))
        
        filtered_channels.append(np.real(filtered))
        removed_channels.append(np.real(removed))
        masks.append(mask)
        fft_mags.append(np.log(magnitude + 1))
    
    # Stack channels
    filtered_image = np.stack(filtered_channels, axis=2)
    removed_image = np.stack(removed_channels, axis=2)
    combined_mask = np.mean(masks, axis=0)
    combined_fft = np.mean(fft_mags, axis=0)
    
    # Clip to valid range
    filtered_image = np.clip(filtered_image, 0, 1)
    
    # Normalize removed for visualization
    removed_image = np.abs(removed_image)
    removed_image = removed_image / (removed_image.max() + 1e-8)
    
    return filtered_image, removed_image, combined_mask, combined_fft


def create_real_image_filtering_demo():
    """Create comprehensive visualization of frequency filtering on real CIFAR-10 images."""
    print("Loading CIFAR-10 samples...")
    samples = load_cifar10_samples()
    
    retain_levels = [0.95, 0.75, 0.50]
    n_samples = len(samples)
    n_levels = len(retain_levels)
    
    # Create figure
    fig = plt.figure(figsize=(24, 5*n_samples))
    gs = fig.add_gridspec(n_samples, 7, hspace=0.4, wspace=0.3)
    
    print("Processing images...")
    for row_idx, (image, class_name) in enumerate(samples):
        # Column 0: Original image
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(image)
        ax.set_title(f'Original\n({class_name})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Column 1: FFT magnitude
        _, _, _, fft_mag = apply_frequency_filter(image, retain_energy=0.95)
        ax = fig.add_subplot(gs[row_idx, 1])
        im = ax.imshow(fft_mag, cmap='hot')
        ax.set_title('FFT Magnitude\n(Log Scale)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Columns 2-4: Different filtering levels (filtered result)
        for col_idx, retain in enumerate(retain_levels, start=2):
            filtered, _, mask, _ = apply_frequency_filter(image, retain_energy=retain)
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(filtered)
            
            # Count percentage of frequencies kept
            pct_kept = np.sum(mask > 0.5) / mask.size * 100
            ax.set_title(f'Retain {int(retain*100)}%\n({pct_kept:.1f}% freqs kept)', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Columns 5-6: Difference from original at 95% and 50%
        for col_idx, retain in enumerate([0.95, 0.50], start=5):
            filtered, removed, _, _ = apply_frequency_filter(image, retain_energy=retain)
            
            # Compute difference
            diff = np.abs(image.astype(float)/255.0 - filtered)
            diff = diff / (diff.max() + 1e-8)  # Normalize for visibility
            
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(diff)
            ax.set_title(f'Removed at {int(retain*100)}%\n(High-freq noise)', 
                        fontsize=11, fontweight='bold', color='red')
            ax.axis('off')
    
    # Add overall title
    fig.suptitle('FALCON v5: Frequency Filtering on Real CIFAR-10 Images\n' + 
                 'Demonstrating gradient smoothing through selective frequency retention',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = Path('paper_stuff/fig_real_image_filtering.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # Also save to results
    results_path = Path('results_v5_final/fig_real_image_filtering.png')
    results_path.parent.mkdir(exist_ok=True)
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_path}")
    
    plt.close()


def create_frequency_spectrum_comparison():
    """Create visualization comparing frequency spectra at different retain levels."""
    print("\nCreating frequency spectrum comparison...")
    samples = load_cifar10_samples()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    retain_levels = [0.95, 0.85, 0.75, 0.50]
    
    for row_idx in range(2):
        image, class_name = samples[row_idx]
        
        for col_idx, retain in enumerate(retain_levels):
            ax = axes[row_idx, col_idx]
            
            filtered, removed, mask, fft_mag = apply_frequency_filter(image, retain_energy=retain)
            
            # Show mask overlay on FFT
            masked_fft = fft_mag * mask
            
            im = ax.imshow(masked_fft, cmap='hot')
            ax.set_title(f'{class_name}\nRetain {int(retain*100)}% energy', 
                        fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle('Frequency Masks at Different Retention Levels\n' +
                 'Red = Kept frequencies, Black = Filtered out',
                 fontsize=14, fontweight='bold')
    
    output_path = Path('paper_stuff/fig_frequency_masks.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    results_path = Path('results_v5_final/fig_frequency_masks.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_path}")
    
    plt.close()


def create_progressive_filtering_demo():
    """Show progressive filtering from 99% to 30% energy retention."""
    print("\nCreating progressive filtering demonstration...")
    samples = load_cifar10_samples()
    image, class_name = samples[0]  # Use first sample
    
    retain_levels = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.30]
    
    fig, axes = plt.subplots(2, len(retain_levels), figsize=(24, 6))
    
    for col_idx, retain in enumerate(retain_levels):
        filtered, removed, mask, _ = apply_frequency_filter(image, retain_energy=retain)
        
        # Top row: Filtered image
        axes[0, col_idx].imshow(filtered)
        axes[0, col_idx].set_title(f'{int(retain*100)}%', fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Bottom row: What was removed
        axes[1, col_idx].imshow(removed, cmap='hot')
        axes[1, col_idx].set_title(f'Removed', fontsize=10)
        axes[1, col_idx].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Filtered Image', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.1, 0.5, 'High-Freq Noise', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', rotation=90, va='center')
    
    fig.suptitle(f'Progressive Frequency Filtering ({class_name})\n' +
                 'From conservative (99%) to aggressive (30%) filtering',
                 fontsize=14, fontweight='bold')
    
    output_path = Path('paper_stuff/fig_progressive_filtering.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    results_path = Path('results_v5_final/fig_progressive_filtering.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {results_path}")
    
    plt.close()


if __name__ == '__main__':
    print("=" * 80)
    print("Generating Real Image Filtering Demonstrations")
    print("=" * 80)
    
    create_real_image_filtering_demo()
    create_frequency_spectrum_comparison()
    create_progressive_filtering_demo()
    
    print("\n" + "=" * 80)
    print("✅ All real image filtering figures generated successfully!")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  1. fig_real_image_filtering.png - Main demo on 4 CIFAR-10 images")
    print("  2. fig_frequency_masks.png - Frequency masks at different levels")
    print("  3. fig_progressive_filtering.png - Progressive filtering from 99% to 30%")
    print("\n📁 Saved to: paper_stuff/ and results_v5_final/")
