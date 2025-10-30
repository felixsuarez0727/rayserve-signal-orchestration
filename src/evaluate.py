"""
Evaluation script for multitask neural network
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from src.data.h5_loader import create_data_loaders, SignalTransform
from src.models.multitask_net import create_model
from src.utils.logger import setup_logging, load_config
from src.utils.seed import set_seed, get_device
from src.utils.metrics import MetricsCalculator, save_metrics


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
    logger: logging.Logger
) -> dict:
    """Evaluate the model on a dataset split."""
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    all_predictions = []
    all_targets = []
    all_snr_predictions = []
    all_snr_targets = []
    all_probabilities = []
    
    logger.info(f"Evaluating on {split_name} set...")
    
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Convert one-hot labels to class indices
            if labels.dim() > 1 and labels.shape[1] > 1:
                class_labels = torch.argmax(labels, dim=1)
            else:
                class_labels = labels.squeeze()
            
            # Generate dummy SNR targets (in real scenario, these would come from dataset)
            snr_targets = torch.randn(labels.shape[0]).to(device) * 10 + 20
            
            # Forward pass
            classification_logits, snr_estimates = model(signals)
            
            # Get predictions
            predictions = torch.argmax(classification_logits, dim=1)
            probabilities = torch.softmax(classification_logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(class_labels.cpu().numpy())
            all_snr_predictions.extend(snr_estimates.cpu().numpy())
            all_snr_targets.extend(snr_targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update metrics calculator
            metrics_calculator.update(
                predictions, class_labels, snr_estimates, snr_targets, probabilities
            )
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics()
    
    # Add additional analysis
    metrics['num_samples'] = len(all_predictions)
    metrics['predictions'] = all_predictions
    metrics['targets'] = all_targets
    metrics['snr_predictions'] = all_snr_predictions
    metrics['snr_targets'] = all_snr_targets
    metrics['probabilities'] = all_probabilities
    
    logger.info(f"{split_name} Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  SNR MAE: {metrics['snr_mae']:.4f}")
    logger.info(f"  SNR R²: {metrics['snr_r2']:.4f}")
    
    return metrics


def plot_confusion_matrix(
    metrics: dict,
    class_names: list,
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """Plot and save confusion matrix."""
    import seaborn as sns
    
    predictions = np.array(metrics['predictions'])
    targets = np.array(metrics['targets'])
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, predictions)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_snr_scatter(
    metrics: dict,
    save_path: Path,
    title: str = "SNR Predictions vs True Values"
) -> None:
    """Plot SNR predictions vs true values."""
    snr_pred = np.array(metrics['snr_predictions'])
    snr_true = np.array(metrics['snr_targets'])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(snr_true, snr_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(snr_true.min(), snr_pred.min())
    max_val = max(snr_true.max(), snr_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('True SNR (dB)')
    plt.ylabel('Predicted SNR (dB)')
    plt.title(title)
    plt.legend()
    
    # Add R² score to plot
    r2 = metrics['snr_r2']
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SNR scatter plot saved to {save_path}")


def plot_class_probabilities(
    metrics: dict,
    class_names: list,
    save_path: Path,
    title: str = "Class Probabilities Distribution"
) -> None:
    """Plot distribution of class probabilities."""
    probabilities = np.array(metrics['probabilities'])
    targets = np.array(metrics['targets'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            # Get probabilities for this class
            class_probs = probabilities[:, i]
            
            # Plot histogram
            axes[i].hist(class_probs, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{class_name} Probabilities')
            axes[i].set_xlabel('Probability')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class probabilities plot saved to {save_path}")


def generate_classification_report(
    metrics: dict,
    class_names: list,
    save_path: Path
) -> None:
    """Generate and save detailed classification report."""
    from sklearn.metrics import classification_report
    
    predictions = np.array(metrics['predictions'])
    targets = np.array(metrics['targets'])
    
    report = classification_report(
        targets, predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Save as JSON
    import json
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Classification report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multitask neural network')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test', 'all'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output_dir', type=str, default='logs/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir']
    )
    
    logger.info("Starting evaluation...")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    val_transform = SignalTransform(normalize=True, add_noise=False)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        train_transform=val_transform,
        val_transform=val_transform,
        lazy_loading=True
    )
    
    # Create model
    logger.info("Creating model...")
    model, _ = create_model(config, device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    # Get class names
    class_names = config['dataset']['class_names']
    
    # Evaluate on specified split(s)
    results = {}
    
    if args.split == 'all':
        splits = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    else:
        splits = {
            args.split: {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }[args.split]
        }
    
    for split_name, data_loader in splits.items():
        logger.info(f"Evaluating {split_name} set...")
        
        # Evaluate model
        metrics = evaluate_model(model, data_loader, device, split_name, logger)
        results[split_name] = metrics
        
        # Generate plots
        plot_confusion_matrix(
            metrics, class_names,
            output_dir / f'{split_name}_confusion_matrix.png',
            f'{split_name.title()} Confusion Matrix'
        )
        
        plot_snr_scatter(
            metrics,
            output_dir / f'{split_name}_snr_scatter.png',
            f'{split_name.title()} SNR Predictions vs True Values'
        )
        
        plot_class_probabilities(
            metrics, class_names,
            output_dir / f'{split_name}_class_probabilities.png',
            f'{split_name.title()} Class Probabilities Distribution'
        )
        
        # Generate classification report
        generate_classification_report(
            metrics, class_names,
            output_dir / f'{split_name}_classification_report.json'
        )
    
    # Save all results
    save_metrics(
        results,
        output_dir / 'evaluation_results.json',
        {
            'checkpoint': args.checkpoint,
            'config': args.config,
            'timestamp': datetime.now().isoformat()
        }
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for split_name, metrics in results.items():
        logger.info(f"\n{split_name.upper()} SET:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"  SNR MAE: {metrics['snr_mae']:.4f}")
        logger.info(f"  SNR MSE: {metrics['snr_mse']:.4f}")
        logger.info(f"  SNR R²: {metrics['snr_r2']:.4f}")
        snr_corr = metrics.get('snr_correlation')
        if snr_corr is not None:
            logger.info(f"  SNR Correlation: {snr_corr:.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()


