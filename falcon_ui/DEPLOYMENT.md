# Falcon UI - GitHub Pages Deployment Guide

## 🚀 Automatic Deployment (Recommended)

The project is configured to automatically deploy to GitHub Pages when you push to the `main` branch.

### Initial Setup

1. **Push to GitHub**:
```bash
cd falcon_ui
git init
git add .
git commit -m "Initial commit: Falcon UI with real results"
git branch -M main
git remote add origin https://github.com/11NOel11/Falcon.git
git push -u origin main
```

2. **Enable GitHub Pages**:
   - Go to your repository: https://github.com/11NOel11/Falcon
   - Click **Settings** → **Pages**
   - Under "Build and deployment":
     - Source: **GitHub Actions**
   - Save

3. **Wait for Deployment**:
   - Go to **Actions** tab
   - Watch the "Deploy Falcon UI to GitHub Pages" workflow
   - Once complete, your site will be live at: **https://11noel11.github.io/Falcon/**

### Subsequent Deployments

Just push to `main`:
```bash
git add .
git commit -m "Update with new features"
git push
```

The site will automatically rebuild and deploy!

---

## 📝 Manual Deployment (Alternative)

If you prefer manual deployment:

### 1. Build the Static Site

```bash
npm run build
```

This creates an `out/` directory with static files.

### 2. Deploy to GitHub Pages (Manual)

```bash
# Install gh-pages package
npm install --save-dev gh-pages

# Add to package.json scripts:
# "deploy": "gh-pages -d out"

# Deploy
npm run deploy
```

---

## 🔧 Configuration Details

### next.config.js

```javascript
basePath: process.env.NODE_ENV === 'production' ? '/Falcon' : '',
assetPrefix: process.env.NODE_ENV === 'production' ? '/Falcon/' : '',
output: 'export',
```

- **basePath**: Must match your repository name (`/Falcon`)
- **output: 'export'**: Generates static HTML/CSS/JS
- **images.unoptimized**: Required for static export

### Local Development

For local development (without basePath):

```bash
# Development mode (no basePath)
npm run dev

# Production build locally
NODE_ENV=development npm run build
npm start
```

---

##real 🌐 Accessing Your Site

- **Live URL**: https://11noel11.github.io/Falcon/
- **Repository**: https://github.com/11NOel11/Falcon

### Pages:
- `/` - Landing page with hero and interactive components
- `/trajectory` - 3D optimizer trajectory visualization
- `/filter` - FFT frequency filter explorer
- `/dynamics` - Training dynamics with real CIFAR-10 results

---

## 📊 Real Data Integration

The UI now uses **real experimental results** from:
- `results/A1_full_metrics.csv` - AdamW results
- `results/M1_full_metrics.csv` - Muon results
- `results/F5_full_metrics.csv` - Falcon results

### Key Results Displayed:
- **Final Accuracies**: AdamW (90.28%), Muon (90.49%), Falcon (90.33%)
- **Training Time**: 60 epochs of CIFAR-10 training
- **Convergence Speed**: Time to reach 85% accuracy
- **Throughput**: Images processed per second

---

## 🐛 Troubleshooting

### Build Fails

If build fails:
```bash
# Clear cache
rm -rf .next node_modules
npm install
npm run build
```

### 404 on GitHub Pages

- Check basePath matches repository name
- Ensure `.nojekyll` file exists in `public/`
- Verify GitHub Actions workflow completed successfully

### Assets Not Loading

- Check `assetPrefix` in `next.config.js`
- Ensure all asset paths are relative (no leading `/`)
- Images must use `next/image` with `unoptimized: true`

---

## 📦 Project Structure

```
falcon_ui/
├── .github/workflows/
│   └── deploy.yml          # Auto-deployment workflow
├── data/
│   ├── dynamics.json       # Real training metrics
│   └── trajectories.json   # Optimizer info
├── pages/                  # Next.js pages
├── components/             # React components
├── public/
│   └── .nojekyll          # GitHub Pages config
├── next.config.js         # GitHub Pages configuration
└── package.json           # Dependencies
```

---

## ✅ Deployment Checklist

- [ ] Real data converted from CSV to JSON
- [ ] next.config.js configured with basePath
- [ ] GitHub Actions workflow created
- [ ] .nojekyll file added
- [ ] Repository pushed to GitHub
- [ ] GitHub Pages enabled in Settings
- [ ] Workflow completed successfully
- [ ] Site accessible at https://11noel11.github.io/Falcon/

---

## 🔄 Updating the Site

1. Make changes to code
2. Test locally: `npm run dev`
3. Commit and push:
   ```bash
   git add .
   git commit -m "Your changes"
   git push
   ```
4. GitHub Actions automatically builds and deploys!
5. Check https://11noel11.github.io/Falcon/ (may take 1-2 minutes)

---

## 📚 Additional Resources

- [Next.js Static Export](https://nextjs.org/docs/pages/building-your-application/deploying/static-exports)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions](https://docs.github.com/en/actions)

**Your Falcon UI is ready to soar! 🦅**
