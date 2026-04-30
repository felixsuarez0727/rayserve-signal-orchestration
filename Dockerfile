# Dockerfile for Signal Orchestration Project
# Multitask Neural Network for Wireless Signal Classification and SNR Estimation

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid massive CUDA downloads in container
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining project dependencies
RUN grep -vE '^(torch|torchvision)' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/logs/run artifacts/checkpoints/main processed data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Ray Serve
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "src/serve/deploy.py", "--host", "0.0.0.0", "--port", "8000"]


