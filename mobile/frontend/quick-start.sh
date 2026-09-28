#!/bin/bash

# Color Aware Leaf – Frontend Quick Start

echo "🍃 Color Aware Leaf – Frontend Setup"
echo "===================================="
echo ""

# Check if in frontend directory
if [ ! -f "package.json" ]; then
  echo "❌ Error: Please run this script from the frontend directory"
  exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
  echo "❌ npm install failed"
  exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📱 Next steps:"
echo ""
echo "  1. Start the dev server:"
echo "     npm start"
echo ""
echo "  2. Choose your platform:"
echo "     - Press 'i' for iOS simulator"
echo "     - Press 'a' for Android emulator"
echo "     - Press 'w' for web browser"
echo ""
echo "  3. Or scan the QR code with Expo Go on your phone"
echo ""
echo "📝 Configuration:"
echo "  - Backend URL: Edit constants/api.ts or set EXPO_PUBLIC_API_URL"
echo ""
echo "🔗 Useful links:"
echo "  - Expo Docs: https://docs.expo.dev"
echo "  - Routing: https://docs.expo.dev/routing/introduction/"
echo ""
