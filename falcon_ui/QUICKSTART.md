# Falcon UI - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
cd falcon_ui
npm install
```

This will install all required packages:
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Plotly.js for visualizations

### Step 2: Run the Development Server

```bash
npm run dev
```

The application will start at `http://localhost:3000`

### Step 3: Explore the Visualizations

Navigate through the pages:

1. **Home** (`/`) - Overview and interactive components
2. **Trajectory** (`/trajectory`) - 3D optimizer paths
3. **Filter** (`/filter`) - FFT frequency filtering
4. **Dynamics** (`/dynamics`) - Training metrics over time

## 🎮 Interactive Features

### Trajectory Page
- Toggle different optimizers on/off
- Adjust learning rate (0.001 - 0.1)
- Change iteration count (1-10)
- Rotate and zoom the 3D plot

### Filter Page
- Select preset filters (Edge, Gaussian, Random)
- Adjust retain fraction ρ (0.5 - 0.95)
- Enable rank-1 approximation toggle
- View real-time FFT magnitude spectrum

### Dynamics Page
- Switch between metrics (Loss, Accuracy, Spectral Norms, Schedule)
- Toggle optimizers in the legend
- Hover for detailed tooltips
- Compare optimizer performance

### Home Page Components
- **SVD Explorer**: Generate random 4×4 matrices and explore rank-k approximations
- **Network Diagram**: Hover over layers to see which optimizer strategy applies

## 🛠️ Development Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linter
npm run lint
```

## 📁 Key Files to Explore

- `pages/trajectory.tsx` - 3D visualization with Plotly
- `utils/fft.ts` - FFT implementation
- `utils/svd.ts` - SVD and matrix operations
- `data/trajectories.json` - Optimizer path data
- `data/dynamics.json` - Training metrics

## 🎨 Customization

### Change Colors

Edit `tailwind.config.js`:

```js
colors: {
  'falcon-bg': '#0A0F24',      // Background
  'falcon-card': '#1C2240',    // Card background
  'falcon-blue': '#4FACF7',    // Primary accent
  'falcon-pink': '#E87BF8',    // Secondary accent
  // ... add your own colors
}
```

### Add New Optimizer

1. Add data to `data/trajectories.json`
2. Update the `OptimizerKey` type in `pages/trajectory.tsx`
3. Add color to the color map

### Modify FFT Parameters

Edit filter presets in `pages/filter.tsx`:

```typescript
const PRESETS: Record<Preset, number[][]> = {
  yourPreset: [
    // 7x7 matrix
  ]
}
```

## 🐛 Troubleshooting

### Port 3000 Already in Use

```bash
# Use a different port
PORT=3001 npm run dev
```

### TypeScript Errors

Make sure all dependencies are installed:

```bash
rm -rf node_modules package-lock.json
npm install
```

### Plotly Not Rendering

Plotly uses dynamic imports to avoid SSR issues. If you see blank plots, check the browser console for errors.

## 📚 Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Plotly.js](https://plotly.com/javascript/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## 🎯 Next Steps

1. Explore all four pages and interact with controls
2. Review the source code to understand the implementation
3. Try modifying the data files to see your own visualizations
4. Customize colors and styling to match your preferences
5. Add new features or visualizations

**Happy Exploring!** 🚀
