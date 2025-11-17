#!/bin/bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

echo "📦 Installing Falcon UI dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Available commands:"
    echo "  ./start.sh           - Start development server"
    echo "  ./build.sh           - Build for production"
    echo "  ./run-production.sh  - Run production server"
    echo ""
    echo "Or use npm directly:"
    echo "  npm run dev   - Development server"
    echo "  npm run build - Build for production"
    echo "  npm start     - Production server"
else
    echo ""
    echo "❌ Setup failed. Check the errors above."
fi
