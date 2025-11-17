# Falcon UI - Dependencies

## System Requirements

- **Node.js**: v20.19.5 (installed via nvm)
- **npm**: v10.8.2
- **Platform**: Linux, macOS, or Windows

## Node.js Installation (This Machine)

Node.js was installed using nvm (Node Version Manager):

```bash
# nvm is already installed at: ~/.nvm
# To activate in new terminal sessions:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Check versions
node --version  # v20.19.5
npm --version   # v10.8.2
```

## NPM Dependencies

### Production Dependencies (16 packages)

| Package | Version | Purpose |
|---------|---------|---------|
| **next** | ^14.0.4 | React framework for production |
| **react** | ^18.2.0 | UI library |
| **react-dom** | ^18.2.0 | React DOM rendering |
| **plotly.js** | ^2.27.1 | Interactive graphing library |
| **react-plotly.js** | ^2.6.0 | React wrapper for Plotly |

### Development Dependencies (10 packages)

| Package | Version | Purpose |
|---------|---------|---------|
| **typescript** | ^5.3.3 | TypeScript compiler |
| **@types/node** | ^20.10.5 | Node.js type definitions |
| **@types/react** | ^18.2.45 | React type definitions |
| **@types/react-dom** | ^18.2.18 | React DOM type definitions |
| **@types/plotly.js** | ^2.12.29 | Plotly type definitions |
| **@types/react-plotly.js** | ^2.6.3 | React Plotly type definitions |
| **tailwindcss** | ^3.4.0 | CSS framework |
| **postcss** | ^8.4.32 | CSS transformer |
| **autoprefixer** | ^10.4.16 | PostCSS plugin |
| **eslint** | ^9.39.1 | Code linter |
| **eslint-config-next** | ^16.0.3 | Next.js ESLint config |

## Total Package Count

- **Direct dependencies**: 16
- **Total installed (including sub-dependencies)**: 747 packages
- **Zero vulnerabilities** ✅

## Installation Commands

### First Time Setup

```bash
# Install all dependencies
npm install

# Or use the setup script
./setup.sh
```

### Install Individual Dependencies

```bash
# Production dependencies
npm install next react react-dom plotly.js react-plotly.js

# Development dependencies
npm install --save-dev typescript @types/node @types/react @types/react-dom \
  @types/plotly.js @types/react-plotly.js tailwindcss postcss autoprefixer \
  eslint eslint-config-next
```

## Package Lock

The exact versions of all 747 packages are locked in:
- `package-lock.json` (NPM lock file)

This ensures consistent installations across different machines.

## Equivalent to Python's requirements.txt

While Python uses `requirements.txt`, Node.js uses:
- **package.json** - Lists direct dependencies
- **package-lock.json** - Locks all dependency versions

To export all installed packages (similar to `pip freeze`):

```bash
npm list --depth=0  # Direct dependencies only
npm list           # All dependencies (very long!)
```

## Build Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **Next.js Compiler** | 14.2.33 | Compiles and bundles the app |
| **SWC** | Built-in | Fast Rust-based compiler |
| **Tailwind JIT** | 3.4.0 | Just-in-time CSS compilation |

## Runtime Information

- **First Load JS**: ~87.5 kB (shared across all pages)
- **Page Sizes**: 3.5-4.7 kB per page
- **Build Time**: ~60 seconds
- **Pages Generated**: 6 static pages

## Update Dependencies

```bash
# Check for outdated packages
npm outdated

# Update to latest compatible versions
npm update

# Update to latest versions (may break)
npm install package@latest
```

## Browser Compatibility

Falcon UI supports all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Additional Notes

- All dependencies are open source
- Zero security vulnerabilities
- Production build is optimized and minified
- Static generation for optimal performance
