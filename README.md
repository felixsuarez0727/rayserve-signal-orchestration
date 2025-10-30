# 🧠 Signal Orchestration Project

**Multitask Neural Network for Wireless Signal Classification and SNR Estimation with Ray Serve**

A comprehensive research project implementing a multitask neural network for wireless signal classification and opportunistic spectrum sensing, orchestrated using Ray Serve for distributed inference.

## 🔬 Scientific Description

This project addresses the challenge of **multitask learning in wireless signal processing** by implementing a shared feature extractor with dual heads for:

1. **Signal Classification**: Identifying signal types (WiFi, LTE, BLE, Noise)
2. **SNR Estimation**: Predicting Signal-to-Noise Ratio for signal quality assessment
3. **Opportunistic Spectrum Sensing**: Analyzing Power Spectral Density (PSD) and channel occupancy for WiFi signals

The architecture leverages **Ray Serve** for distributed inference, enabling scalable and parallel processing of multiple tasks while maintaining low latency and high throughput.

### Key Contributions

- **Multitask Learning**: Shared feature extractor reduces computational overhead while maintaining task-specific performance
- **Ray Serve Orchestration**: Distributed inference with automatic scaling and load balancing
- **Opportunistic Spectrum Sensing**: Real-time PSD analysis for WiFi signals
- **End-to-End Pipeline**: From raw I/Q samples to actionable insights

## 📊 Dataset

The project uses the **SDR WiFi Dataset** from Northeastern University (Daniel Uvaydov, 2021), containing:

- **Signal Types**: WiFi, LTE, BLE, and Noise
- **Format**: I/Q samples (complex signals)
- **Length**: 128 samples per signal
- **Splits**: 70% train, 15% validation, 15% test
- **Total Samples**: ~1,758 training samples

### Dataset Structure

```
processed/
├── sdr_wifi_train.h5  (70% - 1,758 samples)
├── sdr_wifi_val.h5    (15% - ~377 samples)
└── sdr_wifi_test.h5   (15% - ~377 samples)
```

Each HDF5 file contains:
- `X`: I/Q signals of shape `(N, 128, 2)`
- `y`: One-hot encoded labels of shape `(N, 4)`

## 🏗️ Architecture

### Multitask Neural Network

```mermaid
flowchart TD
    A[Input (2,L)] --> B[Feature Extractor]
    B --> C{Shared Features}
    C --> D[Classification Head]
    C --> E[SNR Head]
    D --> F[Softmax]
    E --> G[Scalar SNR]
    F -->|WiFi| H[Opportunistic Spectrum Sensing]
```

### Loss Function

The total loss combines classification and SNR estimation:

```
L_total = L_classification + α × L_snr
```

Where `α` is the SNR loss weight (default: 0.5).

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Ray 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd signal_orchestration
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset**:
   ```bash
   ls processed/
   # Should show: sdr_wifi_train.h5, sdr_wifi_val.h5, sdr_wifi_test.h5
   ```

### Training

```bash
# Basic training
./run_train.sh

# With custom configuration
./run_train.sh --config conf/config.yaml --device cuda

# Resume from checkpoint
./run_train.sh --resume checkpoints/best_checkpoint.pth
```

### Evaluation

```bash
# Evaluate on test set
python src/evaluate.py --checkpoint checkpoints/best_checkpoint.pth --split test

# Evaluate on all splits
python src/evaluate.py --checkpoint checkpoints/best_checkpoint.pth --split all
```

### Ray Serve Deployment

```bash
# Start Ray Serve
./run_serve.sh

# With custom settings
./run_serve.sh --host 0.0.0.0 --port 8000 --num-replicas 4
```

## 🔧 Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Model Configuration
model:
  input_channels: 2
  signal_length: 128
  num_classes: 4
  loss_weights:
    classification_weight: 1.0
    snr_weight: 0.5

# Training Configuration
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping:
    patience: 15

# Ray Serve Configuration
ray_serve:
  host: "0.0.0.0"
  port: 8000
  num_replicas: 2
```

## 🌐 API Endpoints

Once Ray Serve is deployed, the following endpoints are available:

### 1. Inference Endpoint

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [[0.1, 0.2], [0.3, 0.4], ...]
  }'
```

**Response**:
```json
{
  "predictions": {
    "predicted_class": 0,
    "predicted_class_name": "wifi",
    "class_probabilities": {
      "wifi": 0.85,
      "lte": 0.10,
      "ble": 0.03,
      "noise": 0.02
    },
    "snr_estimate": 18.5,
    "confidence": 0.85
  },
  "spectrum_analysis": {
    "peak_frequency": 2.4e9,
    "occupancy_ratio": 0.75,
    "bandwidth": 20e6
  }
}
```

### 2. Spectrum Analysis Endpoint

```bash
curl -X POST http://localhost:8000/spectrum \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [[0.1, 0.2], [0.3, 0.4], ...]
  }'
```

### 3. Feedback Endpoint

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "signal_001",
    "corrected_class": "wifi",
    "corrected_snr": 20.5
  }'
```

### 4. Health Check

```bash
curl http://localhost:8000/health
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t signal-orchestration .
```

### Run Container

```bash
# Basic run
docker run -p 8000:8000 signal-orchestration

# With volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/processed:/app/processed \
  -v $(pwd)/checkpoints:/app/checkpoints \
  signal-orchestration
```

### Docker Compose

```yaml
version: '3.8'
services:
  signal-orchestration:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./processed:/app/processed
      - ./checkpoints:/app/checkpoints
    environment:
      - PYTHONPATH=/app
```

## 📈 Performance Metrics

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro and weighted F1 scores
- **Precision/Recall**: Per-class and macro averages
- **Confusion Matrix**: Detailed classification results

### SNR Estimation Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **R²**: Coefficient of determination
- **Correlation**: Pearson correlation coefficient

### Spectrum Sensing Metrics

- **Peak Frequency**: Detected signal frequency
- **Occupancy Ratio**: Channel utilization percentage
- **Bandwidth**: Estimated signal bandwidth
- **SNR Estimate**: Signal-to-noise ratio

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_h5_loader.py
python -m pytest tests/test_infer_mock.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📊 Monitoring and Logging

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# View at http://localhost:6006
```

### Logs

- **Training logs**: `logs/training.log`
- **Evaluation results**: `logs/evaluation_results.json`
- **Model checkpoints**: `checkpoints/`

## 🔬 Research Applications

This project enables research in:

1. **Multitask Learning**: Shared representation learning for signal processing
2. **Spectrum Sensing**: Opportunistic spectrum analysis for cognitive radio
3. **Distributed Inference**: Ray Serve for scalable signal processing
4. **Active Learning**: Feedback loop for model improvement

## 🛠️ Extending the Project

### Adding New Signal Types

1. Update `class_names` in `conf/config.yaml`
2. Retrain the model with new data
3. Update the spectrum sensing logic if needed

### Adding New Tasks

1. Create new head in `src/models/multitask_net.py`
2. Update loss function in `MultitaskLoss`
3. Modify training loop in `src/train.py`

### Custom Spectrum Analysis

1. Extend `SpectrumAnalyzer` in `src/opportunistic_sensing/psd.py`
2. Add new metrics to the analysis
3. Update API endpoints in `src/serve/app.py`

## 📚 References

1. Uvaydov, D. (2021). SDR WiFi Dataset. Northeastern University.
2. Ray Serve Documentation: https://docs.ray.io/en/latest/serve/
3. PyTorch Multitask Learning: https://pytorch.org/tutorials/

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request




