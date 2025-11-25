#!/bin/bash

# Quick Start Script for Object Detection Web UI

echo "=========================================="
echo "Object Detection Web UI"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Launch the web application
echo "🚀 Starting web server..."
echo ""
echo "✨ The web interface will be available at:"
echo "   http://localhost:7860"
echo "   http://0.0.0.0:7860"
echo ""
echo "📱 To access from other devices on your network:"
echo "   Find your IP: hostname -I"
echo "   Access at: http://YOUR_IP:7860"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

python app.py
