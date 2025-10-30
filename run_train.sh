#!/bin/bash

# Training script for Signal Orchestration Project
# Multitask Neural Network for Wireless Signal Classification and SNR Estimation

set -e  # Exit on any error

# Default values
CONFIG="conf/config.yaml"
DEVICE="auto"
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config PATH    Path to configuration file (default: conf/config.yaml)"
            echo "  --device DEVICE  Device to use (auto, cpu, cuda, mps) (default: auto)"
            echo "  --resume PATH    Path to checkpoint to resume from"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting training..."
echo "Configuration: $CONFIG"
echo "Device: $DEVICE"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi

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

# Check if required HDF5 files exist
for file in "sdr_wifi_train.h5" "sdr_wifi_val.h5" "sdr_wifi_test.h5"; do
    if [ ! -f "processed/$file" ]; then
        echo "Error: Required file 'processed/$file' not found!"
        exit 1
    fi
done

# Create necessary directories
mkdir -p logs checkpoints

# Run training
python src/train.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    ${RESUME:+--resume "$RESUME"}

echo "Training completed!"
echo "Check logs/ directory for training logs and TensorBoard files."
echo "Check checkpoints/ directory for saved models."


