"""
Multitask Neural Network for Wireless Signal Classification and SNR Estimation
Shared feature extractor with dual heads for classification and SNR regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor using 1D convolutions for signal processing.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        conv_layers: list = [32, 64, 128, 256],
        kernel_sizes: list = [7, 5, 3, 3],
        dropout_rate: float = 0.3,
        pool_sizes: list = [2, 2, 2, 2]
    ):
        """
        Initialize the feature extractor.
        
        Args:
            input_channels: Number of input channels (I and Q)
            conv_layers: List of number of filters for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rate: Dropout rate for regularization
            pool_sizes: List of pool sizes for each layer
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.pool_sizes = pool_sizes
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(conv_layers, kernel_sizes, pool_sizes)
        ):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout_rate)
            )
            self.conv_blocks.append(conv_block)
            in_channels = out_channels
        
        # Calculate the output size after convolutions
        self.feature_size = self._calculate_feature_size()
        
        logger.info(f"Feature extractor initialized with {len(conv_layers)} conv layers")
        logger.info(f"Feature size: {self.feature_size}")
    
    def _calculate_feature_size(self) -> int:
        """Calculate the output feature size after all conv layers."""
        # Start with a dummy input to calculate the output size
        dummy_input = torch.randn(1, self.input_channels, 128)  # Assuming 128 samples
        with torch.no_grad():
            x = dummy_input
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Flattened feature tensor
        """
        # Transpose to (batch_size, channels, sequence_length) if needed
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
        
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        return x


class ClassificationHead(nn.Module):
    """
    Classification head for signal type identification.
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int = 4,
        hidden_dims: list = [512, 256],
        dropout_rate: float = 0.5
    ):
        """
        Initialize the classification head.
        
        Args:
            input_size: Size of input features
            num_classes: Number of signal classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build fully connected layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        logger.info(f"Classification head initialized: {input_size} -> {hidden_dims} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Classification logits
        """
        return self.classifier(x)


class SNRHead(nn.Module):
    """
    SNR estimation head for signal quality assessment.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_dims: list = [256, 128],
        dropout_rate: float = 0.3
    ):
        """
        Initialize the SNR head.
        
        Args:
            input_size: Size of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build fully connected layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single value for SNR)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.regressor = nn.Sequential(*layers)
        
        logger.info(f"SNR head initialized: {input_size} -> {hidden_dims} -> 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SNR head.
        
        Args:
            x: Input feature tensor
            
        Returns:
            SNR estimate
        """
        return self.regressor(x).squeeze(-1)


class MultitaskSignalNet(nn.Module):
    """
    Multitask neural network for wireless signal classification and SNR estimation.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        signal_length: int = 128,
        num_classes: int = 4,
        feature_extractor_config: Optional[Dict] = None,
        classification_head_config: Optional[Dict] = None,
        snr_head_config: Optional[Dict] = None
    ):
        """
        Initialize the multitask network.
        
        Args:
            input_channels: Number of input channels (I and Q)
            signal_length: Length of input signal
            num_classes: Number of signal classes
            feature_extractor_config: Configuration for feature extractor
            classification_head_config: Configuration for classification head
            snr_head_config: Configuration for SNR head
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.num_classes = num_classes
        
        # Default configurations
        default_fe_config = {
            'conv_layers': [32, 64, 128, 256],
            'kernel_sizes': [7, 5, 3, 3],
            'dropout_rate': 0.3,
            'pool_sizes': [2, 2, 2, 2]
        }
        
        default_cls_config = {
            'hidden_dims': [512, 256],
            'dropout_rate': 0.5
        }
        
        default_snr_config = {
            'hidden_dims': [256, 128],
            'dropout_rate': 0.3
        }
        
        # Merge configurations
        fe_config = {**default_fe_config, **(feature_extractor_config or {})}
        cls_config = {**default_cls_config, **(classification_head_config or {})}
        snr_config = {**default_snr_config, **(snr_head_config or {})}
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            **fe_config
        )
        
        self.classification_head = ClassificationHead(
            input_size=self.feature_extractor.feature_size,
            num_classes=num_classes,
            **cls_config
        )
        
        self.snr_head = SNRHead(
            input_size=self.feature_extractor.feature_size,
            **snr_config
        )
        
        logger.info("MultitaskSignalNet initialized successfully")
        logger.info(f"Input shape: ({input_channels}, {signal_length})")
        logger.info(f"Number of classes: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Tuple of (classification_logits, snr_estimates)
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Get outputs from both heads
        classification_logits = self.classification_head(features)
        snr_estimates = self.snr_head(features)
        
        return classification_logits, snr_estimates
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with probabilities and SNR estimates.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions, probabilities, and SNR estimates
        """
        self.eval()
        with torch.no_grad():
            classification_logits, snr_estimates = self.forward(x)
            
            # Get class probabilities
            class_probs = F.softmax(classification_logits, dim=1)
            predicted_classes = torch.argmax(class_probs, dim=1)
            
            return {
                'predictions': predicted_classes,
                'probabilities': class_probs,
                'snr_estimates': snr_estimates,
                'logits': classification_logits
            }
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature maps from the feature extractor.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps before flattening
        """
        # Transpose if needed
        if x.dim() == 3 and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
        
        # Apply convolutional blocks
        for conv_block in self.feature_extractor.conv_blocks:
            x = conv_block(x)
        
        return x


class MultitaskLoss(nn.Module):
    """
    Combined loss function for multitask learning.
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        snr_weight: float = 0.5,
        classification_loss_fn: Optional[nn.Module] = None,
        snr_loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize the multitask loss.
        
        Args:
            classification_weight: Weight for classification loss
            snr_weight: Weight for SNR loss (alpha parameter)
            classification_loss_fn: Loss function for classification
            snr_loss_fn: Loss function for SNR regression
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.snr_weight = snr_weight
        
        # Default loss functions
        self.classification_loss_fn = classification_loss_fn or nn.CrossEntropyLoss()
        self.snr_loss_fn = snr_loss_fn or nn.MSELoss()
        
        logger.info(f"MultitaskLoss initialized with weights: cls={classification_weight}, snr={snr_weight}")
    
    def forward(
        self,
        classification_logits: torch.Tensor,
        snr_estimates: torch.Tensor,
        classification_targets: torch.Tensor,
        snr_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.
        
        Args:
            classification_logits: Classification logits
            snr_estimates: SNR estimates
            classification_targets: Classification targets
            snr_targets: SNR targets
            
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        classification_loss = self.classification_loss_fn(classification_logits, classification_targets)
        snr_loss = self.snr_loss_fn(snr_estimates, snr_targets.float())
        
        # Compute weighted total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.snr_weight * snr_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'snr_loss': snr_loss
        }


def create_model(
    config: Dict,
    device: torch.device = torch.device('cpu')
) -> Tuple[MultitaskSignalNet, MultitaskLoss]:
    """
    Create the multitask model and loss function from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to place the model on
        
    Returns:
        Tuple of (model, loss_function)
    """
    # Extract model configuration
    model_config = config['model']
    
    # Create model
    model = MultitaskSignalNet(
        input_channels=model_config['input_channels'],
        signal_length=model_config['signal_length'],
        num_classes=model_config['num_classes'],
        feature_extractor_config=model_config.get('feature_extractor', {}),
        classification_head_config=model_config.get('classification_head', {}),
        snr_head_config=model_config.get('snr_head', {})
    )
    
    # Create loss function
    loss_config = model_config.get('loss_weights', {})
    loss_fn = MultitaskLoss(
        classification_weight=loss_config.get('classification_weight', 1.0),
        snr_weight=loss_config.get('snr_weight', 0.5)
    )
    
    # Move to device
    model = model.to(device)
    
    logger.info(f"Model created and moved to {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, loss_fn


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a dummy configuration
    config = {
        'model': {
            'input_channels': 2,
            'signal_length': 128,
            'num_classes': 4,
            'feature_extractor': {
                'conv_layers': [32, 64, 128],
                'kernel_sizes': [7, 5, 3],
                'dropout_rate': 0.3,
                'pool_sizes': [2, 2, 2]
            },
            'classification_head': {
                'hidden_dims': [256, 128],
                'dropout_rate': 0.5
            },
            'snr_head': {
                'hidden_dims': [128, 64],
                'dropout_rate': 0.3
            },
            'loss_weights': {
                'classification_weight': 1.0,
                'snr_weight': 0.5
            }
        }
    }
    
    # Create model and loss
    model, loss_fn = create_model(config, device)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2, 128).to(device)
    dummy_labels = torch.randint(0, 4, (batch_size,)).to(device)
    dummy_snr = torch.randn(batch_size).to(device)
    
    # Forward pass
    classification_logits, snr_estimates = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Classification logits shape: {classification_logits.shape}")
    print(f"SNR estimates shape: {snr_estimates.shape}")
    
    # Compute loss
    losses = loss_fn(classification_logits, snr_estimates, dummy_labels, dummy_snr)
    print(f"Total loss: {losses['total_loss']:.4f}")
    print(f"Classification loss: {losses['classification_loss']:.4f}")
    print(f"SNR loss: {losses['snr_loss']:.4f}")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"Predictions: {predictions['predictions']}")
    print(f"Probabilities shape: {predictions['probabilities'].shape}")
    print(f"SNR estimates: {predictions['snr_estimates']}")

