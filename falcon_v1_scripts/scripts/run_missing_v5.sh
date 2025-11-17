#!/bin/bash
# Complete the missing FALCON v5 experiments
# Run this to finish the experiment suite

set -e  # Exit on error

echo "=========================================="
echo "FALCON v5 - Missing Experiments Runner"
echo "=========================================="
echo ""
echo "This will run the remaining experiments:"
echo "  - F5_t10 (FALCON v5 fixed-time 10min)"
echo "  - A1_20p, M1_20p, F5_20p (20% data)"
echo "  - A1_10p, M1_10p, F5_10p (10% data)"
echo ""
echo "Estimated time: ~2-3 hours"
echo ""

# ============================================
# 1) Fixed-Time: FALCON v5 (10 min budget)
# ============================================
echo "[1/7] Fixed-time (10 min): FALCON v5"
# ~18 epochs in 10 min at 32s/epoch
python train.py --optimizer falcon_v5 --epochs 18 --exp F5_t10 \
  --rank1-backend poweriter --apply-stages "4" --mask-interval 20 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --use-external-muon

# ============================================
# 2) Data Efficiency: 20% dataset
# ============================================
echo "[2/7] Data efficiency (20%): AdamW"
python train.py --optimizer adamw --epochs 60 --dataset-fraction 0.2 --exp A1_20p

echo "[3/7] Data efficiency (20%): Muon"
python train.py --optimizer muon --epochs 60 --dataset-fraction 0.2 --muon-lr-mult 1.25 --exp M1_20p

echo "[4/7] Data efficiency (20%): FALCON v5"
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.2 --exp F5_20p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.95 --retain-energy-end 0.50 \
  --skip-mix-end 0.85

# ============================================
# 3) Data Efficiency: 10% dataset
# ============================================
echo "[5/7] Data efficiency (10%): AdamW"
python train.py --optimizer adamw --epochs 100 --dataset-fraction 0.1 --exp A1_10p

echo "[6/7] Data efficiency (10%): Muon"
python train.py --optimizer muon --epochs 100 --dataset-fraction 0.1 --muon-lr-mult 1.25 --exp M1_10p

echo "[7/7] Data efficiency (10%): FALCON v5"
python train.py --optimizer falcon_v5 --epochs 100 --dataset-fraction 0.1 --exp F5_10p \
  --rank1-backend poweriter --apply-stages "3,4" --mask-interval 15 --fast-mask \
  --falcon-every-start 2 --falcon-every-end 1 --retain-energy-start 0.96 --retain-energy-end 0.50 \
  --skip-mix-end 0.90

echo ""
echo "=========================================="
echo "All missing experiments complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Generate visualizations: python scripts/plot_results_v5.py"
echo "  2. Check paper_assets/ for figures and tables"
echo ""
