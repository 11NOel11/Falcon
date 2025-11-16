# FALCON v5: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering

**Official Release Package**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Package Contents](#package-contents)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

FALCON v5 is a hybrid optimizer that integrates frequency-domain gradient filtering with orthogonal parameter updates for deep neural network training. This release contains:

- **2 CVPR-format research papers** (~10,400 words)
- **14 publication-quality figures** (300 dpi)
- **Complete source code** for training and evaluation
- **Full experimental results** from 12 experiments on CIFAR-10
- **Comprehensive documentation** for reproduction

### What is FALCON v5?

FALCON v5 combines:
1. **Frequency-domain gradient filtering** (FFT-based noise removal)
2. **Muon's orthogonal updates** (for 2D parameters)
3. **Adaptive scheduling** (dynamic retain-energy and interleaved filtering)
4. **EMA weight averaging** (for stable evaluation)
5. **Frequency-weighted weight decay** (targeting high-freq components)
6. **Mask sharing** (computational efficiency via spatial grouping)

---

## 🔬 Key Findings

### Accuracy
- **FALCON v5:** 90.33% on CIFAR-10 with VGG11
- **AdamW (baseline):** 90.28%
- **Muon:** 90.49% (best)

**Verdict:** ✅ Achieves parity with state-of-the-art methods

### Speed
- **FALCON v5:** 6.7s per epoch
- **AdamW:** 4.8s per epoch
- **Overhead:** +40% (primarily due to FFT operations)

**Verdict:** ⚠️ Significant computational cost

### Data Efficiency
- **20% data:** FALCON v5 79.89% vs AdamW 80.66% (-0.77%)
- **10% data:** FALCON v5 74.40% vs AdamW 75.43% (-1.03%)

**Verdict:** ❌ No advantage with limited data (hypothesis rejected)

### Overall Rating: 6.5/10
- **For Practitioners:** Stick with AdamW (faster, simpler, equally effective)
- **For Researchers:** Interesting ideas but needs further work
- **Scientific Value:** Honest negative results; demonstrates frequency-domain viability

---

## 📦 Package Contents

```
falcon_v5_release/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── paper/                       # Research papers (2)
│   ├── CVPR_PAPER_FALCON_V5.md
│   └── CVPR_PAPER_MUON_ANALYSIS.md
│
├── figures/                     # All figures (14 PNG files, 300 dpi)
│   ├── fig_top1_vs_time.png
│   ├── fig_architecture_comparison.png
│   ├── fig_real_image_filtering.png
│   └── ... (11 more)
│
├── results/                     # Experimental data (13 CSV files)
│   ├── table_summary.csv
│   ├── A1_full_metrics.csv
│   ├── M1_full_metrics.csv
│   ├── F5_full_metrics.csv
│   └── ... (9 more)
│
├── code/                        # Source code
│   ├── train.py                 # Main training script
│   ├── validate_v3.py           # Validation utilities
│   ├── utils.py                 # Helper functions
│   ├── requirements.txt         # Dependencies
│   │
│   ├── optim/
│   │   ├── falcon_v5.py         # FALCON v5 optimizer implementation
│   │   └── __init__.py
│   │
│   ├── models/
│   │   └── cifar_vgg.py         # VGG11 for CIFAR-10
│   │
│   └── scripts/
│       ├── generate_architecture_figures.py
│       ├── generate_real_image_filtering_demo.py
│       └── plot_results_v5.py
│
├── docs/                        # Documentation (multiple .md files)
│   ├── EXECUTIVE_SUMMARY.md
│   ├── QUICK_START_GUIDE.md
│   ├── COMPLETE_MATERIALS_INDEX.md
│   └── ... (more guides)
│
└── data/                        # (empty - download CIFAR-10 here)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd falcon_v5_release
pip install -r requirements.txt
```

### 2. Run Training
```bash
cd code

# Train with FALCON v5 (full 60 epochs)
python train.py --optimizer falcon_v5 --epochs 60 --exp test_falcon

# Train with AdamW (baseline)
python train.py --optimizer adamw --epochs 60 --exp test_adamw

# Train with Muon
python train.py --optimizer muon --epochs 60 --muon-lr-mult 1.25 --exp test_muon
```

### 3. View Results
```bash
# Results saved to: runs/<exp_name>/
# CSV metrics: runs/<exp_name>/metrics.csv
# Checkpoints: runs/<exp_name>/best.pth
```

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ GPU memory (for batch size 512)

### Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib seaborn pandas
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📖 Usage

### Basic Training

**FALCON v5 (recommended settings):**
```bash
python code/train.py \
    --optimizer falcon_v5 \
    --epochs 60 \
    --batch-size 512 \
    --lr 0.01 \
    --weight-decay 0.05 \
    --rank1-backend poweriter \
    --apply-stages "3,4" \
    --mask-interval 15 \
    --falcon-every-start 4 \
    --falcon-every-end 1 \
    --retain-energy-start 0.95 \
    --retain-energy-end 0.50 \
    --skip-mix-end 0.85 \
    --exp my_falcon_run
```

**AdamW (baseline):**
```bash
python code/train.py \
    --optimizer adamw \
    --epochs 60 \
    --batch-size 512 \
    --lr 0.01 \
    --weight-decay 0.05 \
    --exp my_adamw_run
```

**Muon:**
```bash
python code/train.py \
    --optimizer muon \
    --epochs 60 \
    --batch-size 512 \
    --lr 0.01 \
    --weight-decay 0.05 \
    --muon-lr-mult 1.25 \
    --exp my_muon_run
```

### Data Efficiency Experiments

**20% of training data:**
```bash
python code/train.py \
    --optimizer falcon_v5 \
    --epochs 60 \
    --dataset-fraction 0.2 \
    --exp falcon_20pct
```

**10% of training data:**
```bash
python code/train.py \
    --optimizer falcon_v5 \
    --epochs 100 \
    --dataset-fraction 0.1 \
    --exp falcon_10pct
```

### Generate Figures

**Architecture and mechanism figures:**
```bash
python code/scripts/generate_architecture_figures.py
# Output: paper_stuff/*.png (6 figures)
```

**Real image filtering demonstrations:**
```bash
python code/scripts/generate_real_image_filtering_demo.py
# Output: paper_stuff/*.png (3 figures)
```

**Results plots:**
```bash
python code/scripts/plot_results_v5.py
# Output: results_v5_final/*.png (5 figures)
```

---

## 📊 Results

### Full Training (60 epochs, 100% data)

| Optimizer | Best Accuracy | Time per Epoch | Total Time | Throughput |
|-----------|--------------|----------------|------------|------------|
| AdamW | 90.28% | 4.8s | 5.0 min | 10,382 img/s |
| Muon | **90.49%** | 5.3s | 5.4 min | 9,418 img/s |
| FALCON v5 | 90.33% | **6.7s** | 7.0 min | **7,486 img/s** |

### Convergence Speed

| Optimizer | Time to 85% Accuracy |
|-----------|---------------------|
| Muon | **1.18 min** |
| AdamW | 1.27 min |
| FALCON v5 | 1.35 min |

### Data Efficiency

**20% data (10k images, 60 epochs):**
- AdamW: 80.66%
- Muon: 80.78%
- FALCON v5: 79.89% ❌

**10% data (5k images, 100 epochs):**
- AdamW: 75.43%
- Muon: 75.37%
- FALCON v5: 74.40% ❌

### Computational Breakdown (FALCON v5)

| Component | Time | Percentage |
|-----------|------|------------|
| FFT Forward | 0.4s | 13% |
| Energy & Mask | 0.3s | 9% |
| Rank-k Approx | 0.5s | 16% |
| FFT Inverse | 0.4s | 13% |
| Muon Step | 0.5s | 16% |
| AdamW Step | 0.3s | 9% |
| EMA Update | 0.1s | 3% |
| Other | 0.7s | 21% |

**Key Insight:** FFT operations (forward + inverse) consume 26% of optimizer time.

---

## 🔧 Hyperparameters

### FALCON v5 Default Settings

```python
{
    # Basic
    'lr': 0.01,
    'weight_decay': 0.05,
    'batch_size': 512,
    'epochs': 60,
    
    # FALCON-specific
    'falcon_every_start': 4,      # Start: filter every 4 epochs
    'falcon_every_end': 1,        # End: filter every 1 epoch
    'retain_energy_start': 0.95,  # Start: keep 95% energy
    'retain_energy_end': 0.50,    # End: keep 50% energy
    'skip_mix_end': 0.85,         # Muon/AdamW blending
    'ema_decay': 0.999,           # EMA for weights
    'mask_interval': 15,          # Recompute masks every 15 epochs
    'apply_stages': '3,4',        # Apply to VGG stages 3-4
    'rank1_backend': 'poweriter', # Rank-k approximation
    'freq_wd_beta': 0.05,         # Frequency-weighted decay
}
```

### Sensitivity Analysis

See `docs/QUICK_START_GUIDE.md` for hyperparameter tuning guidance.

---

## 📚 Documentation

### Quick Access

- **5-minute overview:** `docs/QUICK_REFERENCE.md`
- **Getting started:** `docs/QUICK_START_GUIDE.md`
- **Complete details:** `docs/COMPLETE_MATERIALS_INDEX.md`
- **Package summary:** `docs/FINAL_PACKAGE_SUMMARY.md`

### Papers

- **Main paper:** `paper/CVPR_PAPER_FALCON_V5.md` (5,600 words)
- **Muon analysis:** `paper/CVPR_PAPER_MUON_ANALYSIS.md` (4,800 words)

### Figures

All 14 figures available in `figures/`:
- 5 results plots
- 6 architecture/mechanism diagrams
- 3 real image filtering demonstrations

---

## 🎓 Citation

If you use FALCON v5 in your research, please cite:

```bibtex
@article{falcon_v5_2024,
  title={FALCON v5: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}

@article{muon_analysis_2024,
  title={Muon Optimizer: An Exploratory Analysis on CIFAR-10},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 🤝 Contributing

This is a research release. For questions or issues:
1. Check documentation in `docs/`
2. Review papers in `paper/`
3. Open an issue on GitHub (when public)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- PyTorch team for excellent deep learning framework
- CIFAR-10 dataset creators
- Muon optimizer authors ([github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon))
- VGG architecture designers

---

## 📧 Contact

For questions about this release:
- Check `docs/QUICK_REFERENCE.md` for quick answers
- Read `docs/COMPLETE_MATERIALS_INDEX.md` for detailed catalog
- Review papers in `paper/` for technical details

---

## 📊 Project Statistics

- **Papers:** 2 (10,400 words)
- **Figures:** 14 (300 dpi, publication quality)
- **Code:** ~5,000 lines Python
- **Experiments:** 12 complete runs
- **Data Points:** ~15,000 (across all experiments)
- **Training Time:** ~3 hours total (all experiments)

---

## ⚠️ Known Limitations

1. **Computational Overhead:** 40% slower than AdamW
2. **Limited Scope:** Only tested on CIFAR-10 + VGG11
3. **Hyperparameter Complexity:** 20+ parameters require tuning
4. **No Data Efficiency Gain:** Performs worse with limited data
5. **Memory Overhead:** ~50% more memory than AdamW

See papers for detailed discussion of limitations.

---

## 🔮 Future Work

### Near-Term
- ImageNet validation (ResNet-50, EfficientNet)
- Transformer experiments (ViT, BERT)
- Custom CUDA kernels for FFT operations
- Automatic hyperparameter tuning

### Long-Term
- Theoretical convergence analysis
- Learnable frequency masks
- Hardware co-design (ASIC support)
- Scaling to larger models (GPT-scale)

---

## ✅ Reproducibility Checklist

- [x] Complete source code provided
- [x] Exact hyperparameters documented
- [x] Random seeds fixed (42)
- [x] Requirements.txt included
- [x] All experimental data included
- [x] Figure generation scripts provided
- [x] Comprehensive documentation
- [x] Honest limitations discussed

---

**Version:** 1.0 Final  
**Release Date:** November 2024  
**Status:** Complete and Ready for Publication

---

**🎉 Thank you for using FALCON v5!**

For the latest updates and more information, check the documentation in `docs/`.
