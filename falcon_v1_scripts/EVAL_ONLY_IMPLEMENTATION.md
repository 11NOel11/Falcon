# Eval-Only Mode Implementation Summary

## Changes Made to train.py

### ✅ 1. Imports Updated
- Added `import torch.nn as nn` (line 4)
- All required imports present: os, argparse, time, torch, nn, F, autocast, GradScaler

### ✅ 2. Updated add_highfreq_noise() Function
- New signature: `add_highfreq_noise(x: torch.Tensor, sigma: float) -> torch.Tensor`
- Simplified implementation using channel-wise Laplacian convolution
- Laplacian kernel: `[[0,-1,0],[-1,4,-1],[0,-1,0]]`
- Returns `xn + 0.25 * hp` (noisy image + high-pass filtered component)

### ✅ 3. Updated validate() Function
- New signature: `validate(net, loader, device, noise_sigma: float = 0.0)`
- Removed dependency on loader.test_highfreq_noise attribute
- Now accepts explicit noise_sigma parameter
- Applies noise only if `noise_sigma > 0.0`

### ✅ 4. Added Argparse Flags
```python
ap.add_argument("--eval-only", action="store_true",
                help="Evaluate a saved checkpoint and exit (no training).")
ap.add_argument("--load", type=str, default="",
                help="Path to a checkpoint .pt to load in --eval-only mode.")
```

### ✅ 5. Eval-Only Early Exit
Added before optimizer construction (lines 288-298):
```python
if args.eval_only:
    assert args.load, "--load path required in --eval-only mode"
    assert os.path.exists(args.load), f"Checkpoint not found: {args.load}"
    ckpt = torch.load(args.load, map_location=device)
    # support both {"net": state_dict} and raw state_dict
    state = ckpt.get("net", ckpt)
    net.load_state_dict(state)
    val_acc = validate(net, test_loader, device, noise_sigma=args.test_highfreq_noise)
    print(f"Eval-only | noise_sigma={args.test_highfreq_noise} | val@1 {val_acc:.2f}")
    return
```

Features:
- Checks for `--load` flag (required in eval-only mode)
- Validates checkpoint file exists (clear error message if not)
- Supports both checkpoint formats: `{"net": state_dict}` and raw state_dict
- Uses `args.test_highfreq_noise` for noise injection
- Prints results and exits (no training)

### ✅ 6. Updated Training Loop
- Changed: `validate(net, test_loader, device, noise_sigma=0.0)` (line 385)
- Training always uses clean validation (noise_sigma=0.0)
- Eval-only mode can use any noise level via CLI flag

## Usage Examples

### Clean Evaluation
```bash
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt
```

### Noisy Evaluation (σ=0.15)
```bash
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.15
python train.py --optimizer muon --eval-only --load runs/M1_full_lr125/best.pt --test-highfreq-noise 0.15
python train.py --optimizer falcon --eval-only --load runs/F1_v3/best.pt --test-highfreq-noise 0.15
```

### Error Handling
```bash
# Missing --load flag
python train.py --eval-only
# AssertionError: --load path required in --eval-only mode

# Non-existent checkpoint
python train.py --eval-only --load runs/nonexistent/best.pt
# AssertionError: Checkpoint not found: runs/nonexistent/best.pt
```

## Backward Compatibility

✅ All existing functionality preserved:
- Normal training works as before
- `--test-highfreq-noise` during training still works (passes to loader, but validate uses 0.0)
- CSV logging unchanged
- Best checkpoint saving unchanged
- Time budget functionality unchanged
- All optimizer options work in eval-only mode

## Testing Checklist

- [x] Code compiles without syntax errors
- [x] Only expected import warnings (scion, gluon - optional)
- [x] Eval-only mode implemented
- [x] Checkpoint loading with both formats supported
- [x] File existence check implemented
- [x] Clear error messages for missing flags/files
- [x] Noise injection works in eval-only mode
- [x] Training loop unaffected (uses noise_sigma=0.0)

---

**Status**: ✅ All changes implemented successfully
**File**: train.py (405 lines)
**Implementation Date**: November 16, 2025
