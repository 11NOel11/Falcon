# Falcon UI - Installation Guide

## ✅ Current Status

**Node.js and npm are already installed on this machine!**

- ✅ Node.js v20.19.5 installed via nvm
- ✅ npm v10.8.2 installed
- ✅ All 747 dependencies installed
- ✅ Project builds successfully
- ✅ Zero vulnerabilities

## 🚀 Quick Start (3 Commands)

```bash
cd /home/noel.thomas/projects/falcon_ui

# Option 1: Use the startup script
./start.sh

# Option 2: Use npm directly
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && npm run dev
```

Then open your browser to: **http://localhost:3000**

## 📋 Available Scripts

| Script | Command | Description |
|--------|---------|-------------|
| **Development** | `./start.sh` | Start dev server with hot reload |
| **Build** | `./build.sh` | Build optimized production bundle |
| **Production** | `./run-production.sh` | Run production server |
| **Setup** | `./setup.sh` | Reinstall dependencies |

## 🔧 Using npm Directly

If you prefer using npm commands:

```bash
# First, load nvm in your terminal
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Then use npm commands
npm run dev     # Development server (localhost:3000)
npm run build   # Build for production
npm start       # Run production server
npm run lint    # Run ESLint
```

## 🌐 Accessing the Application

Once started, the app will be available at:

```
http://localhost:3000        - Local machine
http://127.0.0.1:3000       - Alternative local
http://<your-ip>:3000       - Network access (if firewall allows)
```

### Pages Available

- `/` - Landing page with hero and interactive components
- `/trajectory` - 3D trajectory visualizer
- `/filter` - FFT frequency filter explorer
- `/dynamics` - Training dynamics charts

## 💡 Tips

### Persistent nvm Loading

Add this to your `~/.bashrc` to automatically load nvm in every terminal:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
```

Then reload: `source ~/.bashrc`

### Port Already in Use?

If port 3000 is taken:

```bash
PORT=3001 npm run dev
```

### Clear Cache

If you encounter issues:

```bash
rm -rf .next
rm -rf node_modules
npm install
npm run dev
```

## 🐛 Troubleshooting

### "npm: command not found"

```bash
# Load nvm first
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Verify it works
npm --version
```

### TypeScript Errors

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build Errors

```bash
# Clear Next.js cache
rm -rf .next
npm run build
```

### Module Not Found

```bash
# Ensure all dependencies are installed
npm install
```

## 📦 Installing on Another Machine

If you want to run this on a different machine:

### Option 1: With Node.js Already Installed

```bash
cd falcon_ui
npm install
npm run dev
```

### Option 2: Fresh Installation

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Reload terminal or run:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install Node.js
nvm install 20
nvm use 20

# Install dependencies
cd falcon_ui
npm install

# Start the app
npm run dev
```

## 🌍 Deploying to Production

### Deploy to Vercel (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel
```

### Deploy to Other Platforms

The built files are in the `.next` folder after running `npm run build`.

Compatible with:
- Vercel
- Netlify
- AWS Amplify
- Docker
- Any Node.js hosting

## 📊 System Requirements

- **RAM**: 2GB minimum, 4GB recommended
- **Disk**: 500MB for node_modules
- **CPU**: Any modern processor
- **OS**: Linux, macOS, Windows (WSL)

## 🔐 Security

- All dependencies audited: **0 vulnerabilities**
- Regular updates recommended: `npm audit` and `npm update`

## 📚 Next Steps

1. Start the dev server: `./start.sh`
2. Open http://localhost:3000
3. Explore the visualizations
4. Check out the code in `pages/` and `components/`
5. Read the documentation in `README.md`

**Happy coding!** 🚀
