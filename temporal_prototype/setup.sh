#!/bin/bash
# Automated setup script for Kaggle/Colab
# Just run: bash setup.sh

echo "========================================="
echo "TEMPORAL Setup - Kaggle/Colab Ready"
echo "========================================="

# Install dependencies
echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy matplotlib seaborn scipy tqdm datasets transformers

echo "✅ Dependencies installed!"

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints logs outputs/plots

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  python run_all.py    # Train, evaluate, and visualize everything"
echo "  python train.py      # Just training"
echo ""
