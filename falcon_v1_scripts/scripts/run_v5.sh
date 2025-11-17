#!/usr/bin/env bash
set -euo pipefail

# Email notification (set your email here)
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"  # Set via: export NOTIFY_EMAIL="your.email@example.com"

# Optional SMTP settings (for reliable delivery)
SMTP_SERVER="${SMTP_SERVER:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASSWORD="${SMTP_PASSWORD:-}"

echo "=== FALCON v5 Final Suite (CIFAR-10, VGG11) ==="
echo "Comprehensive experiments: full training, fixed-time, data-efficiency, robustness"
if [ -n "$NOTIFY_EMAIL" ]; then
    echo "📧 Email notifications enabled: $NOTIFY_EMAIL"
fi
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================
# 1) Full Training (60 epochs)
# ============================================
echo "[1/12] Full training: AdamW baseline"
python train.py --optimizer adamw --epochs 60 --exp A1_full

echo "[2/12] Full training: Muon"
python train.py --optimizer muon --epochs 60 --muon-lr-mult 1.25 --exp M1_full

echo "[3/12] Full training: FALCON v5"
python train.py --optimizer falcon_v5 --epochs 60 --exp F5_full \
  --rank1-backend poweriter --apply-stages "4" \
  --mask-interval 20 --fast-mask --falcon-every-start 4 --falcon-every-end 1 \
  --retain-energy-start 0.95 --retain-energy-end 0.50 --skip-mix-end 0.85 \
  --use-external-muon --ema-decay 0.999

# ============================================
# 2) Fixed-Time Fairness (10 minute budget)
# ============================================
echo "[4/12] Fixed-time 10min: AdamW"
python train.py --optimizer adamw --epochs 60 --time-budget-min 10 --exp A1_t10

echo "[5/12] Fixed-time 10min: Muon"
python train.py --optimizer muon --epochs 60 --time-budget-min 10 --muon-lr-mult 1.25 --exp M1_t10

echo "[6/12] Fixed-time 10min: FALCON v5"
python train.py --optimizer falcon_v5 --epochs 60 --time-budget-min 10 --exp F5_t10 \
  --rank1-backend poweriter --apply-stages "4" --mask-interval 20 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --use-external-muon

# ============================================
# 3) Data Efficiency (20% and 10% data)
# ============================================
echo "[7/12] Data efficiency 20%: AdamW"
python train.py --optimizer adamw --epochs 60 --dataset-fraction 0.2 --exp A1_20p

echo "[8/12] Data efficiency 20%: Muon"
python train.py --optimizer muon --epochs 60 --dataset-fraction 0.2 --muon-lr-mult 1.25 --exp M1_20p

echo "[9/12] Data efficiency 20%: FALCON v5"
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.2 --exp F5_20p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.95 --retain-energy-end 0.50 \
  --skip-mix-end 0.85

echo "[10/12] Data efficiency 10%: AdamW"
python train.py --optimizer adamw --epochs 60 --dataset-fraction 0.1 --exp A1_10p

echo "[11/12] Data efficiency 10%: Muon"
python train.py --optimizer muon --epochs 60 --dataset-fraction 0.1 --muon-lr-mult 1.25 --exp M1_10p

echo "[12/12] Data efficiency 10%: FALCON v5"
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.1 --exp F5_10p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.96 --retain-energy-end 0.50 \
  --skip-mix-end 0.90

# ============================================
# 4) Robustness Evaluation (eval-only, σ=0.04)
# ============================================
echo ""
echo "=== Robustness Evaluation (σ=0.04 pixel-space noise) ==="

echo "Robustness eval: AdamW"
python train.py --optimizer adamw --eval-only --load runs/A1_full/best.pt --test-highfreq-noise 0.04

echo "Robustness eval: Muon"
python train.py --optimizer muon --eval-only --load runs/M1_full/best.pt --test-highfreq-noise 0.04

echo "Robustness eval: FALCON v5 (with EMA)"
python train.py --optimizer falcon_v5 --eval-only --load runs/F5_full/best.pt --test-highfreq-noise 0.04 --eval-ema

echo ""
echo "=== FALCON v5 Suite Complete ==="
echo "Results saved to runs/*/"
echo ""
echo "Next step: Generate figures and summary table"
python scripts/plot_results_v5.py

echo ""
echo "Paper assets saved to paper_assets/"
echo "  - 5 publication-quality figures (PNG)"
echo "  - table_summary.csv"
echo "  - report_skeleton.md (ready to paste into your project)"
