# Falcon UI - Optimizer Visualization Suite

> **Where Frequency Meets Geometry**
> Experience learning as art, through trajectories, spectra and structure

![Falcon UI](https://img.shields.io/badge/Falcon-Optimizer-00F5FF?style=for-the-badge)
![Next.js](https://img.shields.io/badge/Next.js-14.0-black?style=for-the-badge&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue?style=for-the-badge&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=for-the-badge&logo=tailwind-css)

**🌐 Live Demo**: [https://11noel11.github.io/Falcon/](https://11noel11.github.io/Falcon/)

## 🚀 Overview

**Falcon UI** is an interactive visualization suite that brings optimization algorithms to life. Built with Next.js, React, and Tailwind CSS, it visualizes **real experimental results** from CIFAR-10 training experiments comparing AdamW, Muon, and Falcon optimizers.

### Real Results from Paper

Based on "FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering" (MBZUAI, 2025):

| Optimizer | Accuracy | Time/Epoch | Throughput | Convergence |
|-----------|----------|------------|------------|-------------|
| **AdamW** | 90.28% | 4.8s | 10,382 img/s | 1.27 min |
| **Muon** | 90.49% | 5.3s | 9,418 img/s | 1.18 min |
| **Falcon** | 90.33% | 6.7s | 7,486 img/s | 1.56 min |

*Training on CIFAR-10 with VGG11 (60 epochs, 50k images)*

### What is Falcon?

**Falcon** (Frequency-Aware Low-rank Conditioning Optimizer) is a novel optimization algorithm that combines:

- **Frequency Domain Analysis**: FFT-based gradient filtering to remove high-frequency noise
- **Low-Rank Approximations**: Rank-1 updates to preserve essential gradient directions
- **Orthogonal Projections**: Gram-Schmidt orthogonalization for decorrelated updates
- **Adaptive Scheduling**: Dynamic masking that evolves throughout training

## ✨ Features

### 🎯 Interactive Visualizations

1. **Trajectory Visualizer** (`/trajectory`)
   - 3D loss surface rendering with Plotly.js
   - Real-time optimizer path comparison (AdamW, Muon, Scion, Falcon)
   - Adjustable learning rates and iteration counts
   - Interactive tooltips with optimizer equations

2. **Frequency Filter Explorer** (`/filter`)
   - Interactive 7×7 filter drawing canvas
   - Real-time 2D FFT computation
   - Magnitude spectrum visualization
   - Energy-based frequency masking
   - Preset filters: Edge detection, Gaussian, Random

3. **Training Dynamics** (`/dynamics`)
   - Multi-metric visualization (Loss, Accuracy, Spectral Norms)
   - Falcon adaptive schedule tracking
   - Layer-wise spectral norm evolution
   - Interactive optimizer toggling

4. **Additional Components**
   - **SVD Explorer**: Interactive rank-k approximation demo
   - **Network Diagram**: Layer-wise optimizer strategy visualization

### 🎨 Design Highlights

- **Dark Theme**: Carefully crafted color palette (#0A0F24 background, neon accents)
- **Animated Elements**: SVG wave backgrounds, glowing sliders, hover effects
- **Custom Typography**: Inter for body text, Playfair Display for poetic captions
- **Responsive Design**: Adapts seamlessly to mobile, tablet, and desktop

## 📦 Installation

### Prerequisites

- Node.js 18+ and npm

### Local Development

```bash
# Clone the repository
git clone https://github.com/11NOel11/Falcon.git
cd Falcon

# Install dependencies
npm install

# Run development server
npm run dev

# Open browser to http://localhost:3000
```

### Build for Production

```bash
# Create optimized static export
npm run build

# Static files will be in ./out directory
```

### Deploy to GitHub Pages

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy**:
1. Push to `main` branch
2. GitHub Actions automatically builds and deploys
3. Site live at https://11noel11.github.io/Falcon/

## 🛠️ Project Structure

```
falcon_ui/
├── pages/
│   ├── index.tsx           # Landing page with hero section
│   ├── trajectory.tsx      # 3D trajectory visualizer
│   ├── filter.tsx          # FFT filter explorer
│   ├── dynamics.tsx        # Training dynamics charts
│   ├── _app.tsx            # App wrapper with global navbar
│   └── _document.tsx       # HTML document configuration
├── components/
│   ├── Navbar.tsx          # Navigation bar with scroll effects
│   ├── Hero.tsx            # Hero section with animated waves
│   ├── WaveBackground.tsx  # SVG animated sine waves
│   ├── Card.tsx            # Reusable card component
│   ├── Slider.tsx          # Custom slider with glow effect
│   ├── Toggle.tsx          # Checkbox toggle component
│   ├── SVDExplorer.tsx     # Interactive SVD demonstration
│   └── NetworkDiagram.tsx  # Layer strategy visualization
├── data/
│   ├── trajectories.json   # Pre-computed optimizer paths
│   └── dynamics.json       # Training metrics and schedules
├── utils/
│   ├── fft.ts              # FFT algorithms (1D, 2D, filtering)
│   └── svd.ts              # SVD, power iteration, matrix ops
├── styles/
│   └── globals.css         # Global styles and Tailwind imports
├── public/                 # Static assets
├── tailwind.config.js      # Tailwind configuration
├── tsconfig.json           # TypeScript configuration
├── next.config.js          # Next.js configuration
└── package.json            # Dependencies and scripts
```

## 🧮 The Mathematics

### Frequency Masking

Falcon applies 2D FFT to gradient matrices and filters frequency components based on energy:

```
Ĝ = FFT(∇L)
E = |Ĝ|²
Threshold = percentile(E, ρ)
Ĝ_filtered = mask(Ĝ, E > Threshold)
G_filtered = IFFT(Ĝ_filtered)
```

- **Retain Fraction (ρ)**: Starts at 0.95, decreases to 0.50 over training
- **Energy Schedule**: Adaptive masking removes more noise as training progresses

### Rank-1 Approximation

Power iteration finds the dominant singular value and vectors:

```
v ← random vector
for iter in 1..max_iterations:
    v ← normalize(A^T A v)
u ← normalize(A v)
σ ← ||A v||
```

The rank-1 update is:
```
θ ← θ - α · σ · u · v^T
```

### Orthogonal Projection

Gram-Schmidt orthogonalization ensures decorrelated updates:

```
for i in 1..n:
    v_i ← g_i
    for j in 1..(i-1):
        v_i ← v_i - <v_i, q_j> · q_j
    q_i ← normalize(v_i)
```

## 🎓 Educational Use

This project is designed for:

- **Researchers**: Understanding optimizer behavior in visual form
- **Students**: Learning about frequency analysis, SVD, and optimization
- **Engineers**: Comparing different optimization strategies
- **Educators**: Teaching machine learning concepts interactively

## 📚 References and Credits

### Core Concepts

- **Frequency Masking**: Inspired by signal processing techniques for noise reduction
- **Low-Rank Approximations**: Based on truncated SVD and matrix factorization theory
- **Orthogonal Methods**: Gram-Schmidt process for basis orthonormalization
- **Adaptive Schedules**: Curriculum learning and adaptive hyperparameter strategies

### Technologies

- **Next.js**: React framework for production-grade applications
- **Plotly.js**: Interactive graphing library for scientific visualization
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Type-safe JavaScript for robust code

### Optimizer Comparisons

- **AdamW**: Adaptive moment estimation with weight decay
- **Muon**: Momentum with orthogonalization
- **Scion**: Spectral conditioning with intelligent orthogonalization
- **Falcon**: Frequency-aware low-rank conditioning (this project's focus)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional visualizations (gradient flows, attention maps)
- Real-time training integration
- More optimizer algorithms
- Performance optimizations for larger matrices
- Interactive tutorials and guided tours

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

Special thanks to:

- The machine learning research community for developing these algorithms
- The open-source community for the amazing tools and libraries
- Anyone passionate about making complex mathematics accessible and beautiful

---

**Built with passion for the art of optimization**
*"In the dance of gradients and frequencies, patterns emerge—each optimizer a unique choreography across the loss landscape."*
