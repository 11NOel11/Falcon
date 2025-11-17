#!/usr/bin/env bash
set -euo pipefail

# FALCON v3 Comprehensive Experiment Suite
# Comparable to optimizer papers (AdamW/Muon)

echo "========================================="
echo " FALCON v3 Experiment Suite"
echo "========================================="

# === Full baselines (60 epochs, CIFAR-10 / VGG11) ===
echo ""
echo "[1/13] AdamW full training..."
python train.py --optimizer adamw --epochs 60 --lr 3e-4 --wd 5e-4 --exp A1_full --seed 0

echo ""
echo "[2/13] Muon full training (lr_mult=1.00)..."
python train.py --optimizer muon --epochs 60 --lr 3e-4 --wd 5e-4 --muon-lr-mult 1.00 --exp M1_full_lr100 --seed 0

echo ""
echo "[3/13] Muon full training (lr_mult=1.25)..."
python train.py --optimizer muon --epochs 60 --lr 3e-4 --wd 5e-4 --muon-lr-mult 1.25 --exp M1_full_lr125 --seed 0

# === FALCON v3 fast defaults ===
echo ""
echo "[4/13] FALCON v3 full training..."
python train.py --optimizer falcon --epochs 60 --lr 3e-4 --wd 5e-4 --exp F1_v3 --seed 0 \
    --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
    --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 \
    --rank-k 1 --late-rank-k-epoch 40 --freq-wd-beta 0.0

# === Robustness (eval noise only) ===
echo ""
echo "[5/13] AdamW with test noise..."
python train.py --optimizer adamw --epochs 60 --lr 3e-4 --wd 5e-4 --exp A1_noise --seed 0 \
    --test-highfreq-noise 0.15

echo ""
echo "[6/13] Muon with test noise..."
python train.py --optimizer muon --epochs 60 --lr 3e-4 --wd 5e-4 --muon-lr-mult 1.25 --exp M1_noise --seed 0 \
    --test-highfreq-noise 0.15

echo ""
echo "[7/13] FALCON v3 with test noise..."
python train.py --optimizer falcon --epochs 60 --lr 3e-4 --wd 5e-4 --exp F1_noise --seed 0 \
    --test-highfreq-noise 0.15 \
    --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
    --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 \
    --rank-k 1 --late-rank-k-epoch 40

# === Data-efficiency (20%) ===
echo ""
echo "[8/13] AdamW with 20% data..."
python train.py --optimizer adamw --epochs 60 --lr 3e-4 --wd 5e-4 --exp A1_20p --seed 0 \
    --dataset-fraction 0.2

echo ""
echo "[9/13] Muon with 20% data..."
python train.py --optimizer muon --epochs 60 --lr 3e-4 --wd 5e-4 --muon-lr-mult 1.25 --exp M1_20p --seed 0 \
    --dataset-fraction 0.2

echo ""
echo "[10/13] FALCON v3 with 20% data..."
python train.py --optimizer falcon --epochs 60 --lr 3e-4 --wd 5e-4 --exp F1_20p --seed 0 \
    --dataset-fraction 0.2 \
    --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
    --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 \
    --rank-k 1 --late-rank-k-epoch 40

# === Fixed-time fairness (10 minutes) ===
echo ""
echo "[11/13] AdamW with 10min budget..."
python train.py --optimizer adamw --epochs 60 --lr 3e-4 --wd 5e-4 --exp A1_t10 --seed 0 \
    --time-budget-min 10

echo ""
echo "[12/13] Muon with 10min budget..."
python train.py --optimizer muon --epochs 60 --lr 3e-4 --wd 5e-4 --muon-lr-mult 1.25 --exp M1_t10 --seed 0 \
    --time-budget-min 10

echo ""
echo "[13/13] FALCON v3 with 10min budget..."
python train.py --optimizer falcon --epochs 60 --lr 3e-4 --wd 5e-4 --exp F1_t10 --seed 0 \
    --time-budget-min 10 \
    --rank1-backend poweriter --poweriter-steps 1 --mask-interval 5 --apply-stages "3,4" \
    --skip-mix-start 0.0 --skip-mix-end 0.7 --retain-energy-start 0.90 --retain-energy-end 0.60 \
    --rank-k 1 --late-rank-k-epoch 40

echo ""
echo "========================================="
echo " All experiments completed!"
echo " Results saved in runs/*/"
echo " Run: python scripts/plot_results.py"
echo "========================================="
