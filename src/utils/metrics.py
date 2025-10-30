"""
Optimized metrics and evaluation utilities for multitask learning
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime


class MetricsCalculator:
    """
    Optimized calculator for various metrics in multitask learning.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            class_names: List of class names for better reporting
        """
        self.class_names = class_names or [f"Class_{i}" for i in range(4)]
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.snr_predictions = []
        self.snr_targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        snr_predictions: torch.Tensor,
        snr_targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Classification predictions
            targets: Classification targets
            snr_predictions: SNR predictions
            snr_targets: SNR targets
            probabilities: Classification probabilities (optional)
        """
        # Convert to numpy and store
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.snr_predictions.extend(snr_predictions.cpu().numpy())
        self.snr_targets.extend(snr_targets.cpu().numpy())
    
    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics."""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Convert one-hot to class indices if needed
        if targets.ndim > 1 and targets.shape[1] > 1:
            targets = np.argmax(targets, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0)
        }
        
        return metrics
    
    def compute_snr_metrics(self) -> Dict[str, float]:
        """Compute SNR regression metrics."""
        snr_pred = np.array(self.snr_predictions)
        snr_true = np.array(self.snr_targets)
        
        metrics = {
            'snr_mae': mean_absolute_error(snr_true, snr_pred),
            'snr_mse': mean_squared_error(snr_true, snr_pred),
            'snr_r2': r2_score(snr_true, snr_pred)
        }
        
        return metrics
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        classification_metrics = self.compute_classification_metrics()
        snr_metrics = self.compute_snr_metrics()
        
        return {**classification_metrics, **snr_metrics}


def compute_loss_metrics(
    losses: Dict[str, torch.Tensor],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute metrics from loss dictionary.
    
    Args:
        losses: Dictionary of losses
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of loss metrics
    """
    metrics = {}
    for key, loss in losses.items():
        metrics[f"{prefix}{key}"] = loss.item()
    
    return metrics


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_metrics(
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
        additional_info: Additional information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    metrics_converted = _convert_numpy_types(metrics)
    
    # Prepare data to save
    data = {
        'metrics': metrics_converted,
        'timestamp': datetime.now().isoformat()
    }
    
    if additional_info:
        data['additional_info'] = additional_info
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_metrics(load_path: str) -> Dict:
    """
    Load metrics from file.
    
    Args:
        load_path: Path to load metrics from
        
    Returns:
        Dictionary of metrics and metadata
    """
    with open(load_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test metrics calculator
    calculator = MetricsCalculator(['wifi', 'lte', 'ble', 'noise'])
    
    # Generate dummy data
    batch_size = 100
    num_classes = 4
    
    predictions = torch.randint(0, num_classes, (batch_size,))
    targets = torch.randint(0, num_classes, (batch_size,))
    snr_predictions = torch.randn(batch_size) * 10 + 20  # SNR around 20 dB
    snr_targets = torch.randn(batch_size) * 10 + 20
    
    # Update metrics
    calculator.update(predictions, targets, snr_predictions, snr_targets)
    
    # Compute metrics
    classification_metrics = calculator.compute_classification_metrics()
    snr_metrics = calculator.compute_snr_metrics()
    all_metrics = calculator.compute_all_metrics()
    
    print("Classification Metrics:")
    for key, value in classification_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nSNR Metrics:")
    for key, value in snr_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test saving metrics
    save_metrics(all_metrics, "test_metrics.json")
    loaded_metrics = load_metrics("test_metrics.json")
    print(f"\nMetrics saved and loaded successfully: {len(loaded_metrics['metrics'])} metrics")
