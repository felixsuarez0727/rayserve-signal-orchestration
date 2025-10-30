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
    
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)
        
        # Convert one-hot labels to class indices
        if labels.dim() > 1 and labels.shape[1] > 1:
            class_labels = torch.argmax(labels, dim=1)
        else:
            class_labels = labels.squeeze()
        
        # Generate dummy SNR targets (in real scenario, these would come from dataset)
        snr_targets = torch.randn(labels.shape[0]).to(device) * 10 + 20  # SNR around 20 dB
        
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
        for signals, labels in tqdm(val_loader, desc=f"Validation {epoch}"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            # Convert one-hot labels to class indices
            if labels.dim() > 1 and labels.shape[1] > 1:
                class_labels = torch.argmax(labels, dim=1)
            else:
                class_labels = labels.squeeze()
            
            # Generate dummy SNR targets
            snr_targets = torch.randn(labels.shape[0]).to(device) * 10 + 20
            
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
    train_transform = SignalTransform(normalize=True, add_noise=True)
    val_transform = SignalTransform(normalize=True, add_noise=False)
    
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
    
    # Create model and loss function
    logger.info("Creating model...")
    model, loss_fn = create_model(config, device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    if config['training']['scheduler']['type'] == 'cosine':
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


if __name__ == "__main__":
    main()

