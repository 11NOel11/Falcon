#!/bin/bash

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Start the production server
echo "🚀 Starting Falcon UI production server..."
echo "📍 Open your browser to: http://localhost:3000"
echo ""
npm start
