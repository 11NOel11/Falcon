#!/bin/bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Build the project
echo "🔨 Building Falcon UI for production..."
npm run build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "🚀 Start the production server with: ./run-production.sh"
else
    echo ""
    echo "❌ Build failed. Check the errors above."
fi
