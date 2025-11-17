# FALCON v1 (CIFAR-10) — Ready to Run

This repo contains:
- `optim/falcon.py` — FALCON optimizer (frequency mask + rank-1 SVD, AdamW-style update)
- `train.py` — Training script (CIFAR-10, small VGG for CIFAR, AMP enabled)
- `models/cifar_vgg.py` — VGG-like model that works on 32x32
- `utils.py` — helpers (accuracy, logging, seed)

## 0) Create folder in VS Code
```bash
mkdir -p ~/projects && cd ~/projects
# download the ZIP from ChatGPT and extract here into ./falcon_v1
```

## 1) Create & activate environment
Using conda (recommended):
```bash
conda create -n falcon python=3.10 -y
conda activate falcon
```

Or venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install PyTorch (CUDA build)
Find the exact command for your CUDA on the official site:
https://pytorch.org/get-started/locally/

**Example (CUDA 12.x wheels):**
```bash
# Replace cu12X with what the selector shows for you (e.g., cu121 / cu124 / cu128)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
> If the CUDA wheel index doesn't match your driver, you can still use the CUDA **runtime** wheels above (you don't need a local toolkit). If in doubt, try the site selector first.

## 3) Install the rest
```bash
pip install -r requirements.txt
# Optional: Muon baseline
pip install git+https://github.com/KellerJordan/Muon
```

## 4) Quick sanity (GPU visible)
```bash
python - <<'PY'
import torch; print('CUDA?', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())
PY
```

## 5) Run pilots (10 epochs) then main (60 epochs)

**Pilots:** (sanity + throughput)
```bash
python train.py --epochs 10 --optimizer adamw --exp A1-pilot
python train.py --epochs 10 --optimizer muon  --exp M1-pilot
python train.py --epochs 10 --optimizer falcon --exp F1-pilot
```

**Main runs (60 epochs):**
```bash
python train.py --epochs 60 --optimizer adamw --exp A1
python train.py --epochs 60 --optimizer muon  --exp M1
python train.py --epochs 60 --optimizer falcon --exp F1 --retain-energy-start 0.75 --retain-energy-end 0.50 --rank1-backend svd
```

## 6) Plot (optional quick look)
Use the CSV in `runs/<EXP>/metrics.csv` to plot Top-1 vs Time.

---

### Notes
- FALCON only processes conv kernels with k>=3 to keep overhead small.
- Rank-1 is done by batched SVD (clear and correct for v1). Use `--rank1-backend poweriter` if you want even faster (approximate) updates.
- AMP is enabled by default.
