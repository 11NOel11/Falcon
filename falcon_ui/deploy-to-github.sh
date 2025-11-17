#!/bin/bash

echo "🚀 Falcon UI - GitHub Deployment Script"
echo "========================================"
echo ""

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M main
fi

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/11NOel11/Falcon.git
else
    echo "✅ GitHub remote already configured"
fi

echo ""
echo "🔨 Testing build..."
NODE_ENV=production npm run build

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed! Please fix errors before deploying."
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""
echo "📝 Staging files for commit..."
git add .

echo ""
echo "💬 Creating commit..."
git commit -m "Deploy: Falcon UI with real CIFAR-10 results

- Real training data from paper experiments
- AdamW: 90.28% accuracy, 4.8s/epoch
- Muon: 90.49% accuracy, 5.3s/epoch
- Falcon: 90.33% accuracy, 6.7s/epoch
- Configured for GitHub Pages deployment
- Auto-deploy via GitHub Actions"

echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "📍 Next Steps:"
    echo "  1. Go to: https://github.com/11NOel11/Falcon/settings/pages"
    echo "  2. Under 'Build and deployment', select Source: GitHub Actions"
    echo "  3. Wait for deployment (check Actions tab)"
    echo "  4. Visit: https://11noel11.github.io/Falcon/"
    echo ""
    echo "🎉 Your Falcon UI will be live in ~2 minutes!"
else
    echo ""
    echo "❌ Push failed. You may need to:"
    echo "  1. Configure git credentials: git config --global user.name 'Your Name'"
    echo "  2. Set up GitHub authentication (SSH key or token)"
    echo "  3. Check repository permissions"
fi
