# Falcon Optimizer

**Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering**

![Falcon UI](https://img.shields.io/badge/Falcon-Optimizer-00F5FF?style=for-the-badge)
![Next.js](https://img.shields.io/badge/Next.js-14.0-black?style=for-the-badge&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?style=for-the-badge&logo=typescript)

**Live Demo**: [https://11noel11.github.io/Falcon/](https://11noel11.github.io/Falcon/)

## Overview

Falcon is a novel optimization algorithm that combines frequency domain analysis with low-rank matrix approximations for deep neural network training. This repository contains:

- **Interactive UI** (`falcon_ui/`) - Modern, 3D-enhanced visualization suite
- **Research Code** (`falcon_v5_release/`) - Implementation and experiments
- **Documentation** - Complete research papers and results

## Features

### Interactive Visualizations

- **3D Trajectory Viewer** - Explore optimizer paths across loss landscapes
- **Frequency Filter Explorer** - Real-time 2D FFT visualization
- **Training Dynamics** - Multi-metric tracking and adaptive scheduling
- **Experimental Results** - Complete CIFAR-10 analysis with figures

### Performance Highlights

- **90.33%** accuracy on CIFAR-10 with VGG11
- **7,486** images/second throughput
- **6.7s** per epoch training time
- **1.56 min** convergence time to 85% accuracy

## Quick Start

### View the UI

Visit [https://11noel11.github.io/Falcon/](https://11noel11.github.io/Falcon/) to explore the interactive visualizations.

### Run Locally

```bash
cd falcon_ui
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
cd falcon_ui
npm run build
```

Static files will be in `falcon_ui/out/`.

## Repository Structure

```
Falcon/
├── falcon_ui/              # Interactive visualization suite
│   ├── pages/              # Next.js pages
│   ├── components/         # Reusable UI components
│   ├── data/               # Experimental data
│   └── styles/             # Global styles and animations
├── falcon_v5_release/      # Research code and experiments
│   ├── code/               # Optimizer implementation
│   ├── figures/            # Generated plots
│   ├── results/            # Experimental results
│   └── docs/               # Documentation
└── README.md               # This file
```

## The Falcon Algorithm

Falcon combines three key innovations:

1. **Frequency Masking** - Energy-aware spectral filtering via 2D FFT
2. **Rank-1 Updates** - Low-rank approximations preserving essential gradients
3. **Orthogonal Projection** - Gram-Schmidt orthogonalization for decorrelated updates

## Citation

If you use this work in your research, please cite:

```bibtex
@article{thomas2025falcon,
  title={FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering},
  author={Thomas, Noel},
  institution={MBZUAI},
  year={2025}
}
```

## License

This project is provided for educational and research purposes.

## Acknowledgments

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [Plotly.js](https://plotly.com/javascript/) - Interactive visualizations
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

**"In the dance of gradients and frequencies, patterns emerge—each optimizer a unique choreography across the loss landscape."**
