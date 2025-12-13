#!/bin/bash
# ============================================
# Vast.ai Training Script for RNN Doodle Classifier
# ============================================
# This script trains the model using Hugging Face Datasets
# No manual data download required!
#
# Usage on vast.ai:
#   1. Clone your repo
#   2. cd Neural_Networks_Fundamentals/3.RNN
#   3. chmod +x vast_train.sh && ./vast_train.sh
# ============================================

set -e  # Exit on error

echo "🚀 Starting RNN Doodle Classifier Training on Vast.ai"
echo "=================================================="

# Check for GPU
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Clear any existing cache (to rebuild with new data)
echo "🗑️  Clearing old cache..."
rm -rf archive/seq_cache_v1

# Train the model (uses HF Datasets by default)
echo "🏋️ Starting training..."
python training-doodle.py

# Test the model
echo "📊 Evaluating model..."
python test_model.py --per_class

echo "=================================================="
echo "✅ Training complete!"
echo "📁 Model saved to: archive/rnn_animals_best.pt"
echo ""
echo "To download the model to your local machine:"
echo "  scp vast-instance:~/Neural_Networks_Fundamentals/3.RNN/archive/rnn_animals_best.pt ."
