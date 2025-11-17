# FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid optimizer integrating frequency-domain gradient filtering with orthogonal parameter updates for deep neural network training.

## 📄 Paper

**FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering**  
*Noel Thomas, Mohamed bin Zayed University of Artificial Intelligence*

Full LaTeX paper: [`latex_paper/main.pdf`](latex_paper/main.pdf) (12 pages)

## 🎯 Key Results

- **Accuracy**: 90.33% on CIFAR-10/VGG11 (vs AdamW 90.28%, Muon 90.49%)
- **Innovation**: 6 integrated techniques (FFT filtering, adaptive energy, mask sharing, EMA, freq-weighted decay, hybrid optimization)
- **Trade-off**: 40% computational overhead vs AdamW

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with FALCON
python train.py --optimizer falcon_v5 --epochs 60

# Train with Muon
python train.py --optimizer muon --epochs 60 --muon-lr-mult 1.25

# Train with AdamW
python train.py --optimizer adamw --epochs 60
```

## 📁 Repository Structure

```
Falcon/
├── optim/falcon_v5.py      # FALCON optimizer
├── models/cifar_vgg.py     # VGG11 model
├── latex_paper/            # Full paper (LaTeX + PDF)
├── train.py                # Main training script
├── validate.py             # Quick validation
└── requirements.txt        # Dependencies
```

## 📊 Results Summary

### Full Training (CIFAR-10, VGG11, 60 epochs)

| Optimizer | Accuracy | Time/Epoch | Relative Speed |
|-----------|----------|------------|----------------|
| AdamW     | 90.28%   | 4.8s       | 1.00×         |
| Muon      | **90.49%** | 5.3s     | 0.91×         |
| FALCON    | 90.33%   | 6.7s       | 0.72×         |

### Convergence Speed

| Optimizer | Time to 85% | Epochs | Relative Speed |
|-----------|-------------|--------|----------------|
| Muon      | **1.18 min** | ~13   | 1.08× faster  |
| AdamW     | 1.27 min    | ~15    | baseline      |
| FALCON    | 1.35 min    | ~10    | 0.94×         |

### Key Findings

✅ **FALCON Strengths:**
- Competitive accuracy (90.33% vs AdamW 90.28%)
- Validates frequency-domain optimization
- Innovative 6-component design

❌ **FALCON Limitations:**
- 40% computational overhead
- Underperforms with limited data
- Not suitable for time-constrained scenarios

✅ **Muon Strengths:**
- +0.21% accuracy improvement over AdamW
- 7% faster convergence to 85%
- Only +10% overhead (acceptable)

## 🛠️ Usage

### Basic Training

```python
from optim.falcon_v5 import FALCONv5

optimizer = FALCONv5(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
    retain_energy_start=0.99,
    retain_energy_end=0.50,
    falcon_every_start=2,
    falcon_every_end=1
)
```

### Training Arguments

```bash
# Full training
python train.py --optimizer falcon_v5 --epochs 60 --exp falcon_full

# Data efficiency (20% data)
python train.py --optimizer falcon_v5 --epochs 60 --dataset-fraction 0.2 --exp falcon_20p

# Fixed time budget (10 minutes)
python train.py --optimizer falcon_v5 --max-time 600 --exp falcon_t10
```

## 🔬 Ablation Studies

| Configuration | Accuracy | Impact |
|--------------|----------|--------|
| Full FALCON | 90.33% | baseline |
| - No Muon | 89.42% | **-0.91%** |
| - No adaptive energy | 89.78% | -0.55% |
| - No mask sharing | 89.95% | -0.38% |
| - No EMA | 90.18% | -0.15% |

**Key Insight**: Muon integration is the most critical component

## 📚 Citation

```bibtex
@article{thomas2025falcon,
  title={FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering},
  author={Thomas, Noel},
  institution={Mohamed bin Zayed University of Artificial Intelligence},
  year={2025},
  url={https://github.com/11NOel11/Falcon}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team for the framework
- CIFAR-10 dataset creators
- Muon optimizer authors

## 📞 Contact

**Noel Thomas**  
Mohamed bin Zayed University of Artificial Intelligence  
noel.thomas@mbzuai.ac.ae  
[@11NOel11](https://github.com/11NOel11)

---

**Paper**: Full LaTeX source and PDF in [`latex_paper/`](latex_paper/)
