#!/bin/bash

# Fast Training script for Signal Orchestration Project
# Optimized for quick testing and demonstration

set -e  # Exit on any error

# Default values
CONFIG="conf/config_fast.yaml"
DEVICE="cpu"
EPOCHS="5"
BATCH_SIZE="128"

echo "🚀 Starting FAST training..."
echo "Configuration: $CONFIG"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

# Check if configuration file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file '$CONFIG' not found!"
    exit 1
fi

# Check if processed data exists
if [ ! -d "processed" ]; then
    echo "Error: Processed data directory not found!"
    echo "Please ensure the processed/ directory contains the HDF5 files."
    exit 1
fi

# Create necessary directories
mkdir -p logs checkpoints

# Run fast training
echo "🔥 Running fast training..."
source venv_new/bin/activate && export PYTHONPATH=$PWD:$PYTHONPATH && python src/train.py \
    --config "$CONFIG" \
    --device "$DEVICE"

echo "✅ Fast training completed!"
echo "📊 Check logs/ directory for training logs"
echo "💾 Check checkpoints/ directory for saved models"


