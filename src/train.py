"""
Training script for multitask neural network
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import Dict

# Import our modules
from src.data.h5_loader import create_data_loaders, SignalTransform
from src.models.multitask_net import create_model, MultitaskLoss
from src.utils.logger import setup_logging, load_config
from src.utils.seed import set_seed, get_device
from src.utils.metrics import MetricsCalculator, save_metrics


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultitaskLoss,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics_calculator = MetricsCalculator()
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (signals, labels, snr_targets) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)
        snr_targets = snr_targets.to(device).squeeze()  # Ensure 1D tensor
        
        # Convert one-hot labels to class indices
        if labels.dim() > 1 and labels.shape[1] > 1:
            class_labels = torch.argmax(labels, dim=1)
        else:
            class_labels = labels.squeeze()
        
        optimizer.zero_grad()
        
        # Forward pass
        classification_logits, snr_estimates = model(signals)
        
        # Compute loss
        losses = loss_fn(classification_logits, snr_estimates, class_labels, snr_targets)
        
        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            predictions = torch.argmax(classification_logits, dim=1)
            probabilities = torch.softmax(classification_logits, dim=1)
            metrics_calculator.update(
                predictions, class_labels, snr_estimates, snr_targets, probabilities
            )
        
        total_loss += losses['total_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'cls_loss': f"{losses['classification_loss'].item():.4f}",
            'snr_loss': f"{losses['snr_loss'].item():.4f}"
        })
    
    # Compute epoch metrics
    epoch_metrics = metrics_calculator.compute_all_metrics()
    epoch_metrics['avg_loss'] = total_loss / num_batches
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: MultitaskLoss,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for signals, labels, snr_targets in tqdm(val_loader, desc=f"Validation {epoch}"):
            signals = signals.to(device)
            labels = labels.to(device)
            snr_targets = snr_targets.to(device).squeeze()  # Ensure 1D tensor
            
            # Convert one-hot labels to class indices
            if labels.dim() > 1 and labels.shape[1] > 1:
                class_labels = torch.argmax(labels, dim=1)
            else:
                class_labels = labels.squeeze()
            
            # Forward pass
            classification_logits, snr_estimates = model(signals)
            
            # Compute loss
            losses = loss_fn(classification_logits, snr_estimates, class_labels, snr_targets)
            
            # Update metrics
            predictions = torch.argmax(classification_logits, dim=1)
            probabilities = torch.softmax(classification_logits, dim=1)
            metrics_calculator.update(
                predictions, class_labels, snr_estimates, snr_targets, probabilities
            )
            
            total_loss += losses['total_loss'].item()
    
    # Compute epoch metrics
    epoch_metrics = metrics_calculator.compute_all_metrics()
    epoch_metrics['avg_loss'] = total_loss / num_batches
    
    return epoch_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Train multitask neural network')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir'],
        log_file=config['logging']['log_file']
    )
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {args.config}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(args.device)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_transform = SignalTransform(
        normalize=True, 
        add_noise=True, 
        time_shift=True, 
        freq_shift=True, 
        amplitude_scale=True
    )
    val_transform = SignalTransform(normalize=True, add_noise=False, time_shift=False, freq_shift=False, amplitude_scale=False)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        train_transform=train_transform,
        val_transform=val_transform,
        lazy_loading=True
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Calculate class weights for imbalanced dataset BEFORE creating model
    logger.info("Calculating class weights for imbalanced dataset...")
    num_classes = config['model']['num_classes']
    class_counts = torch.zeros(num_classes)
    for signals, labels, snr_targets in train_loader:
        if labels.dim() > 1 and labels.shape[1] > 1:
            class_indices = torch.argmax(labels, dim=1)
        else:
            class_indices = labels.squeeze().long()
        # Ensure indices are within valid range
        class_indices = torch.clamp(class_indices, 0, num_classes - 1)
        for idx in class_indices:
            class_counts[idx] += 1
    
    logger.info(f"Class counts: {class_counts.tolist()}")
    
    # Create model (loss_fn will be created after)
    logger.info("Creating model...")
    model, _ = create_model(config, device)
    
    # Compute weights (inverse frequency, balanced)
    total_samples = class_counts.sum()
    class_weights = total_samples / (config['model']['num_classes'] * class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * config['model']['num_classes']  # Normalize
    class_weights = class_weights.to(device)
    
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create loss function with class weights
    from src.models.multitask_net import MultitaskLoss
    loss_config = config['model'].get('loss_weights', {})
    loss_fn = MultitaskLoss(
        classification_weight=loss_config.get('classification_weight', 1.0),
        snr_weight=loss_config.get('snr_weight', 0.5),
        classification_loss_fn=nn.CrossEntropyLoss(weight=class_weights)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler with warmup
    if config['training']['scheduler']['type'] == 'cosine':
        warmup_epochs = config['training']['scheduler'].get('warmup_epochs', 5)
        if warmup_epochs > 0:
            # Cosine annealing with warmup
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (config['training']['num_epochs'] - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs']
            )
    else:
        scheduler = None
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    # Set up TensorBoard
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, Path(args.resume))
    
    # Training loop
    logger.info("Starting training loop...")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch+1, logger
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch+1, logger
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_metrics['avg_loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['avg_loss'], epoch)
        writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
        writer.add_scalar('SNR_MAE/Train', train_metrics['snr_mae'], epoch)
        writer.add_scalar('SNR_MAE/Validation', val_metrics['snr_mae'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['avg_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['avg_loss']
        
        if config['training']['checkpoint']['save_best'] and is_best:
            save_checkpoint(model, optimizer, epoch, val_metrics['avg_loss'],
                          Path(config['training']['checkpoint']['checkpoint_dir']), True)
        
        if config['training']['checkpoint']['save_last']:
            save_checkpoint(model, optimizer, epoch, val_metrics['avg_loss'],
                          Path(config['training']['checkpoint']['checkpoint_dir']), False)
        
        # Early stopping
        if early_stopping(val_metrics['avg_loss']):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final metrics
    final_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1
    }
    
    save_metrics(
        final_metrics,
        Path(config['logging']['log_dir']) / 'final_metrics.json'
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    writer.close()
    
    # Automatically run evaluation after training completes
    logger.info("=" * 60)
    logger.info("Starting automatic evaluation...")
    logger.info("=" * 60)
    
    # Import evaluation functions
    from src.evaluate import evaluate_model, plot_confusion_matrix, plot_snr_scatter, plot_class_probabilities, generate_classification_report
    
    # Determine best checkpoint path
    checkpoint_dir = Path(config['training']['checkpoint']['checkpoint_dir'])
    best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pth'
    
    if not best_checkpoint_path.exists():
        # Fallback to last checkpoint if best doesn't exist
        best_checkpoint_path = checkpoint_dir / 'last_checkpoint.pth'
        logger.warning(f"Best checkpoint not found, using last checkpoint: {best_checkpoint_path}")
    
    if not best_checkpoint_path.exists():
        logger.error(f"No checkpoint found at {best_checkpoint_path}. Skipping evaluation.")
        return
    
    # Load the best model
    logger.info(f"Loading best checkpoint: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    # Get class names
    class_names = config['dataset']['class_names']
    
    # Create output directory for evaluation
    output_dir = Path(config['logging']['log_dir']) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, 'test', logger)
    
    # Generate plots and reports
    logger.info("Generating evaluation plots and reports...")
    plot_confusion_matrix(
        test_metrics, class_names,
        output_dir / 'test_confusion_matrix.png',
        'Test Confusion Matrix'
    )
    
    plot_snr_scatter(
        test_metrics,
        output_dir / 'test_snr_scatter.png',
        'Test SNR Predictions vs True Values'
    )
    
    plot_class_probabilities(
        test_metrics, class_names,
        output_dir / 'test_class_probabilities.png',
        'Test Class Probabilities Distribution'
    )
    
    generate_classification_report(
        test_metrics, class_names,
        output_dir / 'test_classification_report.json'
    )
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("")
    logger.info("TEST SET:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
    logger.info(f"  SNR MAE: {test_metrics['snr_mae']:.4f}")
    logger.info(f"  SNR MSE: {test_metrics['snr_mse']:.4f}")
    logger.info(f"  SNR R²: {test_metrics['snr_r2']:.4f}")
    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

