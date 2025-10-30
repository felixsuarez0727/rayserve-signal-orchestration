"""
Unit tests for HDF5 data loader
"""

import unittest
import numpy as np
import h5py
import torch
from pathlib import Path
import tempfile
import shutil

from src.data.h5_loader import SDRWiFiDataset, SignalTransform, create_data_loaders


class TestSDRWiFiDataset(unittest.TestCase):
    """Test cases for SDRWiFiDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = Path(self.temp_dir)
        
        # Create synthetic test data
        self.num_samples = 100
        self.signal_length = 128
        self.num_classes = 4
        
        # Generate synthetic I/Q signals
        self.signals = np.random.randn(self.num_samples, self.signal_length, 2)
        
        # Generate one-hot labels
        self.labels = np.zeros((self.num_samples, self.num_classes))
        for i in range(self.num_samples):
            class_idx = i % self.num_classes
            self.labels[i, class_idx] = 1
        
        # Create HDF5 file
        self.h5_file = self.temp_dir / "test_data.h5"
        with h5py.File(self.h5_file, 'w') as f:
            f.create_dataset('X', data=self.signals)
            f.create_dataset('y', data=self.labels)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = SDRWiFiDataset(self.h5_file)
        
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.get_signal_shape(), (self.signal_length, 2))
        self.assertEqual(dataset.get_num_classes(), self.num_classes)
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        dataset = SDRWiFiDataset(self.h5_file)
        
        # Test loading a single sample
        signal, label = dataset[0]
        
        self.assertIsInstance(signal, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(signal.shape, (self.signal_length, 2))
        self.assertEqual(label.shape, (self.num_classes,))
    
    def test_dataset_with_transform(self):
        """Test dataset with transform."""
        transform = SignalTransform(normalize=True, add_noise=True)
        dataset = SDRWiFiDataset(self.h5_file, transform=transform)
        
        signal, label = dataset[0]
        
        # Check that signal is normalized (mean close to 0, std close to 1)
        signal_mean = signal.mean(dim=-1, keepdim=True)
        signal_std = signal.std(dim=-1, keepdim=True)
        
        self.assertLess(signal_mean.abs().max(), 0.1)
        self.assertLess((signal_std - 1.0).abs().max(), 0.1)
    
    def test_dataset_lazy_loading(self):
        """Test lazy loading functionality."""
        # Test with lazy loading
        dataset_lazy = SDRWiFiDataset(self.h5_file, lazy_loading=True)
        
        # Test with eager loading
        dataset_eager = SDRWiFiDataset(self.h5_file, lazy_loading=False)
        
        # Both should return the same data
        signal_lazy, label_lazy = dataset_lazy[0]
        signal_eager, label_eager = dataset_eager[0]
        
        self.assertTrue(torch.allclose(signal_lazy, signal_eager))
        self.assertTrue(torch.allclose(label_lazy, label_eager))
    
    def test_dataset_with_nested_structure(self):
        """Test dataset with nested HDF5 structure."""
        # Create nested HDF5 file
        nested_h5_file = self.temp_dir / "nested_data.h5"
        with h5py.File(nested_h5_file, 'w') as f:
            group = f.create_group('train')
            group.create_dataset('X', data=self.signals)
            group.create_dataset('y', data=self.labels)
        
        dataset = SDRWiFiDataset(nested_h5_file)
        
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.get_signal_shape(), (self.signal_length, 2))
        self.assertEqual(dataset.get_num_classes(), self.num_classes)


class TestSignalTransform(unittest.TestCase):
    """Test cases for SignalTransform class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.signal_length = 128
        self.test_signal = torch.randn(2, self.signal_length)
    
    def test_normalize_transform(self):
        """Test normalization transform."""
        transform = SignalTransform(normalize=True, add_noise=False)
        transformed_signal = transform(self.test_signal)
        
        # Check normalization
        signal_mean = transformed_signal.mean(dim=-1, keepdim=True)
        signal_std = transformed_signal.std(dim=-1, keepdim=True)
        
        self.assertLess(signal_mean.abs().max(), 0.1)
        self.assertLess((signal_std - 1.0).abs().max(), 0.1)
    
    def test_noise_transform(self):
        """Test noise addition transform."""
        transform = SignalTransform(normalize=False, add_noise=True, noise_std=0.1)
        transformed_signal = transform(self.test_signal)
        
        # Check that noise was added (signal should be different)
        self.assertFalse(torch.allclose(self.test_signal, transformed_signal))
    
    def test_combined_transform(self):
        """Test combined normalization and noise addition."""
        transform = SignalTransform(normalize=True, add_noise=True, noise_std=0.1)
        transformed_signal = transform(self.test_signal)
        
        # Check that signal was transformed
        self.assertFalse(torch.allclose(self.test_signal, transformed_signal))
        
        # Check normalization
        signal_mean = transformed_signal.mean(dim=-1, keepdim=True)
        signal_std = transformed_signal.std(dim=-1, keepdim=True)
        
        self.assertLess(signal_mean.abs().max(), 0.2)  # Allow some tolerance due to noise
        self.assertLess((signal_std - 1.0).abs().max(), 0.2)


class TestDataLoaders(unittest.TestCase):
    """Test cases for data loader creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir = Path(self.temp_dir)
        
        # Create synthetic test data
        self.num_samples = 50
        self.signal_length = 128
        self.num_classes = 4
        
        # Generate synthetic I/Q signals
        self.signals = np.random.randn(self.num_samples, self.signal_length, 2)
        
        # Generate one-hot labels
        self.labels = np.zeros((self.num_samples, self.num_classes))
        for i in range(self.num_samples):
            class_idx = i % self.num_classes
            self.labels[i, class_idx] = 1
        
        # Create HDF5 files for train, val, test
        for split in ['train', 'val', 'test']:
            h5_file = self.temp_dir / f"sdr_wifi_{split}.h5"
            with h5py.File(h5_file, 'w') as f:
                f.create_dataset('X', data=self.signals)
                f.create_dataset('y', data=self.labels)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_data_loaders(self):
        """Test data loader creation."""
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=self.temp_dir,
            batch_size=16,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False,
            lazy_loading=True
        )
        
        # Test that loaders were created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test that loaders have correct number of samples
        self.assertEqual(len(train_loader.dataset), self.num_samples)
        self.assertEqual(len(val_loader.dataset), self.num_samples)
        self.assertEqual(len(test_loader.dataset), self.num_samples)
        
        # Test loading a batch
        for signals, labels in train_loader:
            self.assertEqual(signals.shape[0], 16)  # Batch size
            self.assertEqual(signals.shape[1], self.signal_length)  # Signal length
            self.assertEqual(signals.shape[2], 2)   # I/Q channels
            self.assertEqual(labels.shape[0], 16)   # Batch size
            self.assertEqual(labels.shape[1], self.num_classes)  # Number of classes
            break  # Only test first batch


if __name__ == '__main__':
    unittest.main()
