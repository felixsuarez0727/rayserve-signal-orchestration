"""
Random seed utilities for reproducible experiments
"""

import random
import numpy as np
import torch
import os
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    print(f"Using device: {device_obj}")
    
    return device_obj


if __name__ == "__main__":
    # Test seed setting
    set_seed(42)
    
    # Test device detection
    device = get_device("auto")
    
    # Test reproducibility
    print("Testing reproducibility...")
    
    # Generate some random numbers
    python_random = random.random()
    numpy_random = np.random.random()
    torch_random = torch.rand(1).item()
    
    print(f"Python random: {python_random}")
    print(f"NumPy random: {numpy_random}")
    print(f"PyTorch random: {torch_random}")
    
    # Reset seed and generate again
    set_seed(42)
    
    python_random2 = random.random()
    numpy_random2 = np.random.random()
    torch_random2 = torch.rand(1).item()
    
    print(f"Python random (2nd): {python_random2}")
    print(f"NumPy random (2nd): {numpy_random2}")
    print(f"PyTorch random (2nd): {torch_random2}")
    
    print(f"Python reproducible: {python_random == python_random2}")
    print(f"NumPy reproducible: {numpy_random == numpy_random2}")
    print(f"PyTorch reproducible: {torch_random == torch_random2}")

