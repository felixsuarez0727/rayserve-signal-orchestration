#!/bin/bash

# Ray Serve deployment script for Signal Orchestration Project
# Multitask Neural Network for Wireless Signal Classification and SNR Estimation

set -e  # Exit on any error

# Default values
CONFIG="conf/config.yaml"
HOST="0.0.0.0"
PORT="8000"
NUM_REPLICAS="2"
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-replicas)
            NUM_REPLICAS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config PATH      Path to configuration file (default: conf/config.yaml)"
            echo "  --host HOST        Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT        Port to bind to (default: 8000)"
            echo "  --num-replicas N   Number of model replicas (default: 2)"
            echo "  --checkpoint PATH  Path to model checkpoint"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting Ray Serve deployment..."
echo "Configuration: $CONFIG"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Replicas: $NUM_REPLICAS"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi

# Check if configuration file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file '$CONFIG' not found!"
    exit 1
fi

# Check if checkpoint exists (if provided)
if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT' not found!"
    exit 1
fi

# Create necessary directories
mkdir -p logs

# Start Ray Serve
echo "Starting Ray Serve..."
python src/serve/deploy.py \
    --config "$CONFIG" \
    --host "$HOST" \
    --port "$PORT" \
    --num-replicas "$NUM_REPLICAS" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}

echo "Ray Serve deployment completed!"
echo "API available at: http://$HOST:$PORT"
echo "Endpoints:"
echo "  POST /infer      - Main inference endpoint"
echo "  POST /spectrum    - Spectrum analysis endpoint"
echo "  POST /feedback    - Feedback endpoint"
echo "  GET  /health      - Health check endpoint"


