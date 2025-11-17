# Realistic Eval-Time Noise Implementation

## Summary of Changes

I've updated `train.py` to use a more realistic noise injection that properly handles CIFAR-10 normalization.

## Changes Made

### ✅ 1. Added Helper Functions for Normalization

**`_maybe_denorm(x)`** - Detects if input is normalized and converts to pixel space [0,1]
- Uses CIFAR-10 statistics: 
  - MEAN = [0.4914, 0.4822, 0.4465]
  - STD = [0.2023, 0.1994, 0.2010]
- Detects normalization by checking if values are outside [0,1]
- Returns: (denormalized_x, stats, was_normalized)

**`_maybe_norm(x, stats)`** - Re-normalizes if needed
- Takes stats from `_maybe_denorm`
- Returns input unchanged if stats are None
- Otherwise applies: `(x - MEAN) / STD`

### ✅ 2. Replaced add_highfreq_noise() Function

New implementation with proper normalization handling:

```python
def add_highfreq_noise(x: torch.Tensor, sigma_img: float) -> torch.Tensor:
    """
    sigma_img: std of Gaussian noise in pixel space [0,1]. Recommended 0.02–0.06.
    """
    if sigma_img <= 0:
        return x
    # 1) Work in pixel space
    x_img, stats, was_normed = _maybe_denorm(x)
    # 2) Add Gaussian noise in [0,1]
    x_noisy = x_img + torch.randn_like(x_img) * sigma_img
    # 3) Gentle high-pass (depthwise Laplacian)
    lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    lap = lap.repeat(x_noisy.size(1), 1, 1, 1)
    hp = F.conv2d(x_noisy, lap, padding=1, groups=x_noisy.size(1))
    x_noisy = x_noisy + 0.08 * hp  # small blend; keep subtle
    # 4) Clamp and re-normalize if needed
    x_noisy = x_noisy.clamp(0.0, 1.0)
    return _maybe_norm(x_noisy, stats)
```

**Key improvements:**
- Automatically detects if input is normalized
- Works in pixel space [0,1] where noise std is meaningful
- Uses gentler high-pass blend (0.08 vs 0.25)
- Clamps to valid range before re-normalizing
- Returns data in same format as input

### ✅ 3. Updated CLI Help Text

Changed from:
```python
ap.add_argument("--test-highfreq-noise", type=float, default=0.0)
```

To:
```python
ap.add_argument("--test-highfreq-noise", type=float, default=0.0,
                help="Eval-only: add Gaussian + mild high-pass noise in image space [0,1]. Recommended 0.02–0.06.")
```

**Clarifies:**
- Noise is applied in image space [0,1]
- Recommended values: 0.02–0.06 (much more realistic than 0.15)
- Only used during evaluation

## Recommended Usage

### Previous Usage (Too Strong)
```bash
# Old recommendation was sigma=0.15 which is very harsh
python train.py --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.15
```

### New Recommended Usage (Realistic)
```bash
# Light noise (σ=0.02 in pixel space)
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.02

# Moderate noise (σ=0.04)
python train.py --optimizer muon --eval-only --load runs/M1_full_lr125/best.pt --test-highfreq-noise 0.04

# Heavier noise (σ=0.06)
python train.py --optimizer falcon --eval-only --load runs/F1_v3/best.pt --test-highfreq-noise 0.06
```

## Technical Details

### Why This is Better

1. **Normalized Input Handling**: CIFAR-10 typically uses normalized inputs. The old implementation added noise directly to normalized values, which doesn't correspond to meaningful pixel noise levels.

2. **Meaningful Noise Levels**: 
   - Old: σ=0.15 on normalized data (~0.75 in pixel space) - extremely harsh
   - New: σ=0.02-0.06 in pixel space - realistic sensor/compression noise

3. **Gentler High-Pass**: Reduced from 0.25 to 0.08 blend factor for more subtle high-frequency emphasis

4. **Automatic Detection**: Works whether data is normalized or not, no configuration needed

### Noise Scale Comparison

| Old σ (normalized) | New σ (pixel) | Equivalent Effect |
|--------------------|---------------|-------------------|
| 0.15               | ~0.75         | Extreme corruption |
| -                  | 0.06          | Heavy noise (realistic upper bound) |
| -                  | 0.04          | Moderate noise (typical) |
| -                  | 0.02          | Light noise (minimal) |

## Validation

✅ Imports correct (torch, F already present)
✅ Helper functions added before add_highfreq_noise
✅ add_highfreq_noise signature updated (sigma → sigma_img)
✅ validate() already calls add_highfreq_noise(x, noise_sigma) correctly
✅ CLI help updated with recommended values
✅ No syntax errors (only expected optional import warnings)

## Testing

```bash
# Test clean evaluation
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt

# Test with realistic light noise
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.02

# Test with realistic moderate noise
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.04

# Test with realistic heavy noise
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.06
```

---

**Status**: ✅ Complete and ready to use with realistic noise levels
**File**: train.py (429 lines)
**Implementation Date**: November 16, 2025
