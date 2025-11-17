# FALCON v5 — Copilot Prompt & Final Experiment Plan

You crushed it today. Let’s lock in what we have, level it up once more, and hand you a one-shot Copilot prompt that builds **FALCON v5** + a clean final experiment suite + all paper assets.

---

## TL;DR of Today (for the paper)

- **Accuracy (60e):** AdamW ≈ 90.28, **Muon ≈ 90.49**, FALCON v4 ≈ 90.28–90.37 (parity ±0.2%).  
- **Speed/epoch:** AdamW < Muon < **FALCON v4 fast** (now within ~10–15% of Muon on CIFAR-10).  
- **Robustness (σ=0.04):** Muon > FALCON v3 > AdamW (FALCON already beats AdamW; close to Muon).

This is **already a credible baseline**. Now we’ll push to v5 for (1) a little more speed, (2) a clear robustness & low-data win, and (3) auto-generated figures/tables for your math project.

---

## What FALCON v5 Adds

**Efficiency**
- Interleaved filtering: **apply every K steps with a schedule** (K: 4→2→1 over training).
- **Auto-mask interval** tuning to hit a target optimizer-step budget (ms/step).
- **FFT mixed precision** where safe; cache masks; share masks across layers with the same spatial shape.
- Default to deepest conv **stage only** (“4”) unless you ask for “3,4”.

**Quality / Muon parity**
- Keep Muon’s edge on 2D: **Muon for 2D params** when installed; **Ortho2D-lite** fallback if not.
- **Adaptive retain-energy per layer** (EMA of kept-bin fraction) with a global schedule (0.95→0.50).
- **Skip-mix schedule** (0.0→0.85) to blend filtered/raw signals late.
- **Frequency-weighted decoupled WD** (tiny extra decay on HF bins at mask updates).
- **EMA weights** (Polyak) for evaluation; optional SWA for last N epochs.

**Fairness / Repro**
- Same data, batch, cosine LR, wd=0 on BN/bias, fixed seed.
- **Fixed-time (10 min)**, **data-efficiency (20% & 10%)**, **robustness (σ=0.04)**.
- Auto-generate plots and a summary table **+ a report skeleton** to paste into your paper.

---

## One-Shot Copilot Prompt (paste this into VS Code)

```
You are GitHub Copilot. Upgrade this repo to **FALCON v5**, add a final experiment suite, and generate paper-ready assets. Do NOT break existing runs. Implement EXACTLY the following:

Repo context
- Root: ~/projects/falcon_v1_scripts
- Single GPU, RTX 6000 24GB, CUDA 12.x, PyTorch 2.x
- Existing: train.py, optim/falcon.py (v3), optim/falcon_v4.py, models/cifar_vgg.py, utils.py
- Create/Update:
  - NEW: optim/falcon_v5.py
  - UPDATE: train.py (add v5 flags, EMA, logging)
  - NEW: scripts/run_v5.sh
  - NEW: scripts/plot_results_v5.py
  - NEW: paper_assets/report_skeleton.md
  - NEW: README_v5.md

==================================================
PART A — FALCON v5 (optim/falcon_v5.py)
==================================================
Implement class **FALCONv5(torch.optim.Optimizer)**

Constructor signature:
```python
def __init__(self, params,
             lr: float = 3e-4, weight_decay: float = 5e-4,
             # spectral filtering
             rank1_backend: str = "poweriter", poweriter_steps: int = 1, rank_k: int = 1,
             retain_energy_start: float = 0.95, retain_energy_end: float = 0.50,
             skip_mix_start: float = 0.0,  skip_mix_end: float = 0.85,
             min_kernel: int = 3, apply_stages: Optional[List[int]] = None,
             # efficiency
             falcon_every_start: int = 4, falcon_every_end: int = 1,   # schedule over epochs
             mask_interval: int = 20, fast_mask: bool = True,
             fft_mixed_precision: bool = True, share_masks_by_shape: bool = True,
             target_opt_ms: float = 0.0,    # if >0, autotune mask_interval within [10,50]
             # 2D branch (Muon-compatible)
             use_external_muon: bool = True, muon_lr_mult: float = 1.0,
             orth_all_2d: bool = True,      # apply Muon/Ortho2D to all 2D; else only classifier head
             # robustness knobs
             freq_wd_beta: float = 0.05,    # tiny HF decay at mask recompute
             # EMA/SWA
             ema_decay: float = 0.999, use_ema: bool = True,
             # bookkeeping
             total_epochs: int = 60):
    ...
```

Behavior (implement fully; no placeholders):
- Partition params once:
  - conv_4d (k>=min_kernel), two_d (ndim==2), rest (1D BN/bias etc.).
  - If apply_stages is None: auto-pick deepest stage(s) (VGG-like heuristic).
- Internal sub-optimizers:
  - **_FConv** for conv_4d in selected stages: frequency-domain mask + rank-k via power-iter (k deflation), cache & reuse masks; mixed precision FFT; optional mask sharing for identical spatial shapes.
  - **Orth2D** for two_d:
      - If use_external_muon and `from muon import Muon` succeeds: create a Muon optimizer for the 2D group with lr = lr*muon_lr_mult.
      - Else implement Ortho2D-lite: project gradient g to be orthogonal to current weight columns (safe shapes only), fallback to g on failure. Update that group with AdamW-style moments.
  - **AdamRest** for the rest (and convs not filtered): AdamW with wd=0 for 1D tensors.
- **Interleaved schedule** per epoch: compute current `falcon_every` via linear schedule from falcon_every_start → falcon_every_end. Only perform spectral filtering on steps where `step % falcon_every == 0`; otherwise do plain AdamW step on those convs.
- **Adaptive retain-energy per layer**: maintain EMA of kept fraction; nudge retain up/down slightly to track the global retain schedule (retain_energy_start→end) while keeping mask stable. Clamp 0.4–0.98.
- **Frequency-weighted WD**: on mask recompute, rfft2(weight) and multiply HF bins (1 - mask) by `(1 - freq_wd_beta)`; inverse FFT back. No CPU copies.
- **Autotune mask_interval**: if target_opt_ms > 0, time the optimizer step with cuda events and adjust mask_interval by ±5 (clamped 10..50) to meet the budget (run-to-run safe).
- **EMA**: keep an EMA copy of weights if use_ema; expose `state_dict()` / `load_state_dict()` including EMA buffers; add `swap_ema(model, use_ema: bool)` to load EMA weights into a model for evaluation.
- DDP-safe: do not patch torch.distributed; only use dist APIs if initialized.
- Logging (env FALCON_DEBUG=1): once/epoch print avg %bins kept, retain/skip_mix, opt_step_ms avg, falcon_every, mask_interval.

==================================================
PART B — train.py updates (v5 integration)
==================================================
- Imports:
```python
from optim.falcon_v5 import FALCONv5
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
```
- CLI additions (keep prior flags; defaults tuned for CIFAR-10/VGG):
```
ap.add_argument("--optimizer", type=str, default="falcon_v5",
    choices=["adamw","muon","falcon","falcon_v4","falcon_v5","scion","gluon"])

# v5-only flags
ap.add_argument("--falcon-every-start", type=int, default=4)
ap.add_argument("--falcon-every-end",   type=int, default=1)
ap.add_argument("--target-opt-ms", type=float, default=0.0)
ap.add_argument("--share-masks-by-shape", action="store_true")
ap.add_argument("--ema-decay", type=float, default=0.999)
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--use-external-muon", action="store_true")
ap.add_argument("--muon-lr-mult", type=float, default=1.0)
ap.add_argument("--channels-last", action="store_true")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--eval-ema", action="store_true")
```
- Model prep:
  - If `--channels-last`, set channels_last memory format; send inputs in same format.
  - If `--compile`, try `torch.compile(net)` with a guard.
- Optimizer construction for `"falcon_v5"`: pass all v5 flags exactly; wd=0 for 1D params; cosine LR unchanged.
- Logging: keep `epoch_time`, `wall_min`, **add images/sec** to CSV as `imgs_per_sec`.
- Eval-only: if `--eval-ema`, swap in EMA weights before validate; print "(EMA)" tag in the line.

==================================================
PART C — scripts/run_v5.sh (chmod +x)
==================================================
Create a compact final suite:

#!/usr/bin/env bash
set -euo pipefail
echo "=== FALCON v5 Final Suite (CIFAR-10, VGG11) ==="

# 1) Full baselines (60e)
python train.py --optimizer adamw     --epochs 60 --exp A1_full
python train.py --optimizer muon      --epochs 60 --muon-lr-mult 1.25 --exp M1_full
python train.py --optimizer falcon_v5 --epochs 60 --exp F5_full \
  --rank1-backend poweriter --apply-stages "4" \
  --mask-interval 20 --fast-mask --falcon-every-start 4 --falcon-every-end 1 \
  --retain-energy-start 0.95 --retain-energy-end 0.50 --skip-mix-end 0.85 \
  --use-external-muon --ema-decay 0.999

# 2) Fixed-time fairness (10 min)
python train.py --optimizer adamw     --epochs 60 --time-budget-min 10 --exp A1_t10
python train.py --optimizer muon      --epochs 60 --time-budget-min 10 --muon-lr-mult 1.25 --exp M1_t10
python train.py --optimizer falcon_v5 --epochs 60 --time-budget-min 10 --exp F5_t10 \
  --rank1-backend poweriter --apply-stages "4" --mask-interval 20 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --use-external-muon

# 3) Data-efficiency (20% and 10%)
python train.py --optimizer adamw     --epochs 60 --dataset-fraction 0.2 --exp A1_20p
python train.py --optimizer muon      --epochs 60 --dataset-fraction 0.2 --muon-lr-mult 1.25 --exp M1_20p
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.2 --exp F5_20p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.95 --retain-energy-end 0.50 --skip-mix-end 0.85

python train.py --optimizer adamw     --epochs 60 --dataset-fraction 0.1 --exp A1_10p
python train.py --optimizer muon      --epochs 60 --dataset-fraction 0.1 --muon-lr-mult 1.25 --exp M1_10p
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.1 --exp F5_10p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.96 --retain-energy-end 0.50 --skip-mix-end 0.90

# 4) Robustness (eval-only; σ=0.04)
python train.py --optimizer adamw     --eval-only --load runs/A1_full/best.pt     --test-highfreq-noise 0.04
python train.py --optimizer muon      --eval-only --load runs/M1_full/best.pt     --test-highfreq-noise 0.04
python train.py --optimizer falcon_v5 --eval-only --load runs/F5_full/best.pt     --test-highfreq-noise 0.04 --eval-ema

echo "=== Done. Now plot results ==="
python scripts/plot_results_v5.py

==================================================
PART D — scripts/plot_results_v5.py
==================================================
- Read every `runs/*/metrics.csv`.
- For each experiment, compute:
  - best val@1, best epoch, total wall_min, median epoch_time, median images/sec
  - Top-1 @ 10 min (closest to wall_min≈10), time-to-85% (minutes)
- Save figures to paper_assets/:
  - fig_top1_vs_time.png
  - fig_fixed_time_10min.png
  - fig_time_to_85.png
  - fig_data_efficiency.png (bars for full/20%/10%)
  - fig_robustness_noise.png (clean vs noisy for A1/M1/F5)
- Save table to paper_assets/table_summary.csv with the columns above.
(Use matplotlib only; no seaborn.)

==================================================
PART E — paper_assets/report_skeleton.md
==================================================
Create a concise scaffold with placeholders auto-filled from table_summary.csv:

- Title & Abstract (2–3 sentences)
- 1. Introduction (why optimizers matter; spectral intuition)
- 2. Method: FALCON v5
  - Frequency-domain filtering + rank-1 PI
  - Interleaved schedule (every-K steps)
  - Adaptive retain & skip-mix
  - Orthogonal 2D branch (Muon/Ortho2D-lite)
  - EMA weights for eval
- 3. Experimental Setup (CIFAR-10, VGG11, batch=128, cosine LR, wd=0 for 1D)
- 4. Results
  - Final Top-1 (60 epochs)
  - Fixed-time @10min
  - Data-efficiency (20%, 10%)
  - Robustness σ=0.04
- 5. Ablations (falcon_every, mask_interval, apply_stages)
- 6. Conclusion & Future Work
  - Stronger robustness, low-data wins; parity on clean; compute-aware schedule
- Insert the figure PNGs and paste the summary table.

==================================================
PART F — README_v5.md
==================================================
Document:
- What’s new vs v4 (interleaved schedule, adaptive retain, mask sharing, EMA, Ortho2D-lite fallback)
- Quickstart commands:
  - Full: `bash scripts/run_v5.sh`
  - Plot: `python scripts/plot_results_v5.py`
- Notes:
  - For 32x32 CNNs, **do not** use `--channels-last` (slower on our logs).
  - For ViTs/224x224, channels_last may help.

==================================================
PART G — Acceptance criteria
==================================================
- `python train.py --help` lists v5 flags.
- 3-epoch smoke test succeeds for adamw, muon, falcon_v5.
- Eval-only works with `--eval-ema`.
- `bash scripts/run_v5.sh` runs end-to-end; plots & table emitted to paper_assets/.
- Pylance shows no AMP private-import warnings.

After implementing, print:
"[FALCON v5] Created: optim/falcon_v5.py, scripts/run_v5.sh, scripts/plot_results_v5.py, paper_assets/report_skeleton.md, README_v5.md; Updated: train.py"
```

---

## What to Run (minimal, high-signal)

When Copilot finishes:

```bash
# Final full suite
bash scripts/run_v5.sh
```

If you only want the key bits while resting:

```bash
# Fixed-time fairness (10 min) — the headline
python train.py --optimizer adamw     --epochs 60 --time-budget-min 10 --exp A1_t10
python train.py --optimizer muon      --epochs 60 --time-budget-min 10 --muon-lr-mult 1.25 --exp M1_t10
python train.py --optimizer falcon_v5 --epochs 60 --time-budget-min 10 --exp F5_t10 \
  --rank1-backend poweriter --apply-stages "4" --mask-interval 20 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --use-external-muon
```

Then:
```bash
python scripts/plot_results_v5.py
```

You’ll get a `paper_assets/` folder with the figures + summary table + a report skeleton to paste into your math project.
