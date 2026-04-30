"""
Ray Serve application for multitask neural network inference
"""

import ray
from ray import serve
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import yaml
from datetime import datetime

# Import our modules
from src.models.multitask_net import create_model
from src.downstream.link_adaptation import LinkAdaptationModule
from src.orchestrator.pipeline import SignalOrchestrator
from src.utils.logger import load_config
from src.utils.seed import get_device

logger = logging.getLogger(__name__)


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}
)
class MultitaskSignalModel:
    """
    Ray Serve deployment for multitask signal classification and SNR estimation.
    """
    
    def __init__(self, config_path: str = "conf/config.yaml"):
        """
        Initialize the model deployment.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Get device
        self.device = get_device(self.config['hardware']['device'])
        
        # Create model
        self.model, _ = create_model(self.config, self.device)
        
        # Load checkpoint if specified
        checkpoint_path = self.config.get('model', {}).get('checkpoint_path')
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        
        self.model.eval()
        
        # Get class names
        self.class_names = self.config['dataset']['class_names']
        
        # Build orchestration downstream module
        downstream_cfg = self.config.get('downstream', {})
        self.orchestrator = SignalOrchestrator(
            downstream_module=LinkAdaptationModule(
                min_confidence=downstream_cfg.get('min_confidence', 0.60),
                snr_offset_db=downstream_cfg.get('snr_offset_db', 0.0),
            )
        )
        
        logger.info("MultitaskSignalModel initialized successfully")
    
    def preprocess_signal(self, signal_data: Union[List, np.ndarray]) -> torch.Tensor:
        """
        Preprocess input signal data.
        
        Args:
            signal_data: Input signal data (I/Q samples)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to numpy array if needed
        if isinstance(signal_data, list):
            signal_data = np.array(signal_data)
        
        # Ensure correct shape
        if signal_data.ndim == 2:
            # Shape is (N, 2) - I/Q samples
            if signal_data.shape[1] != 2:
                raise ValueError(f"Expected 2 channels (I/Q), got {signal_data.shape[1]}")
        elif signal_data.ndim == 3:
            # Shape is (batch_size, N, 2)
            if signal_data.shape[2] != 2:
                raise ValueError(f"Expected 2 channels (I/Q), got {signal_data.shape[2]}")
        else:
            raise ValueError(f"Expected 2D or 3D array, got {signal_data.ndim}D")
        
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal_data).float()
        
        # Add batch dimension if needed
        if signal_tensor.dim() == 2:
            signal_tensor = signal_tensor.unsqueeze(0)
        
        # Transpose to (batch_size, channels, sequence_length)
        if signal_tensor.shape[1] != 2:
            signal_tensor = signal_tensor.transpose(1, 2)
        
        return signal_tensor.to(self.device)
    
    def postprocess_predictions(
        self,
        classification_logits: torch.Tensor,
        snr_estimates: torch.Tensor
    ) -> Dict:
        """
        Postprocess model predictions.
        
        Args:
            classification_logits: Classification logits
            snr_estimates: SNR estimates
            
        Returns:
            Dictionary with processed predictions
        """
        # Get class probabilities and predictions
        probabilities = torch.softmax(classification_logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Convert to numpy
        predicted_classes = predicted_classes.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        snr_estimates = snr_estimates.cpu().numpy()
        
        # Create results
        results = []
        for i in range(len(predicted_classes)):
            result = {
                'predicted_class': int(predicted_classes[i]),
                'predicted_class_name': self.class_names[predicted_classes[i]],
                'class_probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities[i])
                },
                'snr_estimate': float(snr_estimates[i]),
                'confidence': float(probabilities[i][predicted_classes[i]])
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    async def infer(self, request: Dict) -> Dict:
        """
        Main inference endpoint.
        
        Args:
            request: Dictionary containing 'signal' key with I/Q data
            
        Returns:
            Dictionary with predictions and metrics
        """
        try:
            # Extract signal data
            signal_data = request['signal']
            
            # Preprocess
            signal_tensor = self.preprocess_signal(signal_data)
            
            # Inference
            with torch.no_grad():
                classification_logits, snr_estimates = self.model(signal_tensor)
            
            # Postprocess
            predictions = self.postprocess_predictions(classification_logits, snr_estimates)
            
            # Add metadata
            result = {
                'predictions': predictions,
                'orchestration': self.orchestrator.route(predictions),
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'input_shape': signal_tensor.shape,
                    'device': str(self.device)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def spectrum(self, request: Dict) -> Dict:
        """
        Compatibility endpoint: returns downstream decision details.
        
        Args:
            request: Dictionary containing 'signal' key with I/Q data
            
        Returns:
            Dictionary with spectrum analysis results
        """
        try:
            # Reuse inference endpoint and expose only orchestration decision
            infer_result = await self.infer(request)
            if 'error' in infer_result:
                return infer_result
            
            return {
                'orchestration': infer_result.get('orchestration', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Spectrum analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def feedback(self, request: Dict) -> Dict:
        """
        Feedback endpoint for active learning.
        
        Args:
            request: Dictionary with corrected labels
            
        Returns:
            Confirmation response
        """
        try:
            # Extract feedback data
            signal_id = request.get('signal_id')
            corrected_class = request.get('corrected_class')
            corrected_snr = request.get('corrected_snr')
            
            # Log feedback (in real implementation, save to database)
            logger.info(f"Received feedback for signal {signal_id}: "
                       f"class={corrected_class}, snr={corrected_snr}")
            
            return {
                'status': 'success',
                'message': 'Feedback received and logged',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def health(self) -> Dict:
        """
        Health check endpoint.
        
        Returns:
            Health status
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'device': str(self.device),
                'class_names': self.class_names
            }
        }


# Create the Ray Serve application
app = MultitaskSignalModel.bind()


if __name__ == "__main__":
    # This is for testing the deployment locally
    import asyncio
    
    async def test_deployment():
        # Initialize Ray
        ray.init()
        
        # Deploy the model
        handle = serve.run(app)
        
        # Test inference
        test_signal = np.random.randn(128, 2)  # Random I/Q signal
        
        result = await handle.infer.remote({'signal': test_signal.tolist()})
        print("Inference result:", result)
        
        # Test spectrum analysis
        spectrum_result = await handle.spectrum.remote({'signal': test_signal.tolist()})
        print("Spectrum analysis result:", spectrum_result)
        
        # Test health check
        health_result = await handle.health.remote()
        print("Health check result:", health_result)
        
        # Shutdown
        serve.shutdown()
        ray.shutdown()
    
    asyncio.run(test_deployment())


