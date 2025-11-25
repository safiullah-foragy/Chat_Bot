#!/bin/bash

# Setup script for Object Detection Model

echo "=========================================="
echo "Object Detection Model Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Download COCO dataset: python dataset.py --download"
echo "   (Warning: This will download ~20GB of data)"
echo "3. Train the model: python train.py --epochs 10"
echo "4. Make predictions: python predict.py --image path/to/image.jpg"
echo ""
echo "Or use the pretrained model directly:"
echo "python predict.py --image path/to/image.jpg"
echo ""
