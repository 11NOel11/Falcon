#!/bin/bash
# FALCON v2 Experimental Suite
# Run from repo root: bash scripts/run_v2.sh

set -e

echo "=== FALCON v2 Experimental Suite ==="
echo "Starting at: $(date)"

cd "$(dirname "$0")/.."

# PRIMARY EXPERIMENTS (Full 60 epochs)
echo ""
echo "=== PRIMARY: Full baselines (60 epochs) ==="

# A1: AdamW baseline
echo "Running A1_full (AdamW)..."
python train.py --optimizer adamw --epochs 60 --batch-size 128 --seed 42 --exp A1_full

# M1: Muon baselines (LR sweep for fairness)
echo "Running M1_full_lr100 (Muon, LR mult 1.0)..."
python train.py --optimizer muon --epochs 60 --batch-size 128 --seed 42 --muon-lr-mult 1.00 --exp M1_full_lr100

echo "Running M1_full_lr125 (Muon, LR mult 1.25)..."
python train.py --optimizer muon --epochs 60 --batch-size 128 --seed 42 --muon-lr-mult 1.25 --exp M1_full_lr125

# F1: FALCON v2 (fast settings)
echo "Running F1_fast (FALCON v2)..."
python train.py --optimizer falcon --epochs 60 --batch-size 128 --seed 42 --exp F1_fast \
  --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
  --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 --rank-k 1

# ROBUSTNESS EXPERIMENTS (High-freq noise evaluation)
echo ""
echo "=== ROBUSTNESS: High-freq noise (eval only) ==="

echo "Running A1_noise (AdamW + HF noise)..."
python train.py --optimizer adamw --epochs 60 --batch-size 128 --seed 42 --exp A1_noise --test-highfreq-noise 0.15

echo "Running F1_noise (FALCON + HF noise)..."
python train.py --optimizer falcon --epochs 60 --batch-size 128 --seed 42 --exp F1_noise --test-highfreq-noise 0.15 \
  --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
  --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 --rank-k 1

# DATA-EFFICIENCY EXPERIMENTS (20% training data)
echo ""
echo "=== DATA-EFFICIENCY: 20% train data ==="

echo "Running A1_20p (AdamW, 20% data)..."
python train.py --optimizer adamw --epochs 60 --batch-size 128 --seed 42 --exp A1_20p --dataset-fraction 0.2

echo "Running F1_20p (FALCON, 20% data)..."
python train.py --optimizer falcon --epochs 60 --batch-size 128 --seed 42 --exp F1_20p --dataset-fraction 0.2 \
  --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
  --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 --rank-k 1

# FIXED-TIME EXPERIMENTS (10-minute budget)
echo ""
echo "=== FIXED-TIME: 10-minute budget ==="

echo "Running A1_t10 (AdamW, 10min)..."
python train.py --optimizer adamw --epochs 60 --batch-size 128 --seed 42 --exp A1_t10 --time-budget-min 10

echo "Running M1_t10 (Muon, 10min)..."
python train.py --optimizer muon --epochs 60 --batch-size 128 --seed 42 --exp M1_t10 --time-budget-min 10 --muon-lr-mult 1.00

echo "Running F1_t10 (FALCON, 10min)..."
python train.py --optimizer falcon --epochs 60 --batch-size 128 --seed 42 --exp F1_t10 --time-budget-min 10 \
  --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
  --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 --rank-k 1

echo ""
echo "=== All experiments complete! ==="
echo "Finished at: $(date)"
echo ""
echo "Run plotting script:"
echo "  python scripts/plot_results.py"
