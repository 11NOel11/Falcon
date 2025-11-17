#!/usr/bin/env bash
set -euo pipefail

echo "=== FALCON v4 Suite (CIFAR-10, VGG11) ==="
echo "This suite runs comprehensive experiments comparing AdamW, Muon, and FALCON v4"
echo ""

# ============================================
# Full Training (60 epochs)
# ============================================
echo "[1/11] Full training: AdamW baseline"
python train.py --optimizer adamw --epochs 60 --exp A1_full

echo "[2/11] Full training: Muon"
python train.py --optimizer muon --epochs 60 --muon-lr-mult 1.25 --exp M1_full

echo "[3/11] Full training: FALCON v4"
python train.py --optimizer falcon_v4 --epochs 60 --exp F4_full \
  --rank1-backend poweriter --poweriter-steps 1 --apply-stages "4" \
  --mask-interval 20 --fast-mask --falcon-every 2 --fft-mixed-precision \
  --skip-mix-start 0.0 --skip-mix-end 0.70 --retain-energy-start 0.90 --retain-energy-end 0.60 \
  --channels-last --orth-all-2d --use-external-muon

# ============================================
# Fixed-Time Fairness (10 min budget)
# ============================================
echo "[4/11] Fixed-time 10min: AdamW"
python train.py --optimizer adamw --epochs 60 --time-budget-min 10 --exp A1_t10

echo "[5/11] Fixed-time 10min: Muon"
python train.py --optimizer muon --epochs 60 --time-budget-min 10 --muon-lr-mult 1.25 --exp M1_t10

echo "[6/11] Fixed-time 10min: FALCON v4"
python train.py --optimizer falcon_v4 --epochs 60 --time-budget-min 10 --exp F4_t10 \
  --rank1-backend poweriter --apply-stages "4" --mask-interval 20 --fast-mask \
  --falcon-every 2 --fft-mixed-precision --channels-last --orth-all-2d --use-external-muon

# ============================================
# Data Efficiency (20% and 10% data)
# ============================================
echo "[7/11] Data efficiency 20%: AdamW"
python train.py --optimizer adamw --epochs 60 --dataset-fraction 0.2 --exp A1_20p

echo "[8/11] Data efficiency 20%: Muon"
python train.py --optimizer muon --epochs 60 --dataset-fraction 0.2 --muon-lr-mult 1.25 --exp M1_20p

echo "[9/11] Data efficiency 20%: FALCON v4"
python train.py --optimizer falcon_v4 --epochs 60 --dataset-fraction 0.2 --exp F4_20p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every 2 --retain-energy-start 0.95 --retain-energy-end 0.50 --skip-mix-end 0.85 \
  --channels-last --orth-all-2d --use-external-muon

echo "[10/11] Data efficiency 10%: AdamW"
python train.py --optimizer adamw --epochs 60 --dataset-fraction 0.1 --exp A1_10p

echo "[11/11] Data efficiency 10%: Muon"
python train.py --optimizer muon --epochs 60 --dataset-fraction 0.1 --muon-lr-mult 1.25 --exp M1_10p

echo "[12/11] Data efficiency 10%: FALCON v4"
python train.py --optimizer falcon_v4 --epochs 60 --dataset-fraction 0.1 --exp F4_10p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every 2 --retain-energy-start 0.95 --retain-energy-end 0.50 --skip-mix-end 0.85 \
  --channels-last --orth-all-2d --use-external-muon

# ============================================
# Robustness Evaluation (eval-only, noise σ=0.04)
# ============================================
echo ""
echo "=== Robustness Evaluation (σ=0.04 pixel-space noise) ==="

echo "Robustness eval: AdamW"
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.04

echo "Robustness eval: Muon"
python train.py --optimizer muon --eval-only --load runs/M1_full/best.pt --test-highfreq-noise 0.04

echo "Robustness eval: FALCON v4"
python train.py --optimizer falcon_v4 --eval-only --load runs/F4_full/best.pt --test-highfreq-noise 0.04

echo ""
echo "=== FALCON v4 Suite Complete ==="
echo "Results saved to runs/*/"
echo "Run 'python scripts/plot_results_v4.py' to generate figures and summary table"
