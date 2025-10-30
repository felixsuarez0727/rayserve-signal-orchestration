"""
Unit tests for inference functionality
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from src.models.multitask_net import MultitaskSignalNet, MultitaskLoss
from src.opportunistic_sensing.psd import SpectrumAnalyzer, analyze_wifi_signal


class TestMultitaskSignalNet(unittest.TestCase):
    """Test cases for MultitaskSignalNet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.input_channels = 2
        self.signal_length = 128
        self.num_classes = 4
        
        # Create dummy input
        self.dummy_input = torch.randn(self.batch_size, self.input_channels, self.signal_length)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        self.dummy_snr = torch.randn(self.batch_size) * 10 + 20  # SNR around 20 dB
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MultitaskSignalNet(
            input_channels=self.input_channels,
            signal_length=self.signal_length,
            num_classes=self.num_classes
        )
        
        self.assertIsNotNone(model.feature_extractor)
        self.assertIsNotNone(model.classification_head)
        self.assertIsNotNone(model.snr_head)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = MultitaskSignalNet(
            input_channels=self.input_channels,
            signal_length=self.signal_length,
            num_classes=self.num_classes
        )
        
        # Forward pass
        classification_logits, snr_estimates = model(self.dummy_input)
        
        # Check output shapes
        self.assertEqual(classification_logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(snr_estimates.shape, (self.batch_size,))
    
    def test_model_predict(self):
        """Test model prediction method."""
        model = MultitaskSignalNet(
            input_channels=self.input_channels,
            signal_length=self.signal_length,
            num_classes=self.num_classes
        )
        
        # Make predictions
        predictions = model.predict(self.dummy_input)
        
        # Check prediction keys
        expected_keys = ['predictions', 'probabilities', 'snr_estimates', 'logits']
        for key in expected_keys:
            self.assertIn(key, predictions)
        
        # Check prediction shapes
        self.assertEqual(predictions['predictions'].shape, (self.batch_size,))
        self.assertEqual(predictions['probabilities'].shape, (self.batch_size, self.num_classes))
        self.assertEqual(predictions['snr_estimates'].shape, (self.batch_size,))
        self.assertEqual(predictions['logits'].shape, (self.batch_size, self.num_classes))
    
    def test_model_with_different_input_shapes(self):
        """Test model with different input shapes."""
        model = MultitaskSignalNet(
            input_channels=self.input_channels,
            signal_length=self.signal_length,
            num_classes=self.num_classes
        )
        
        # Test with transposed input (N, 2) -> (2, N)
        transposed_input = self.dummy_input.transpose(1, 2)
        classification_logits, snr_estimates = model(transposed_input)
        
        self.assertEqual(classification_logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(snr_estimates.shape, (self.batch_size,))


class TestMultitaskLoss(unittest.TestCase):
    """Test cases for MultitaskLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 4
        
        # Create dummy data
        self.classification_logits = torch.randn(self.batch_size, self.num_classes)
        self.snr_estimates = torch.randn(self.batch_size)
        self.classification_targets = torch.randint(0, self.num_classes, (self.batch_size,))
        self.snr_targets = torch.randn(self.batch_size) * 10 + 20
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = MultitaskLoss(
            classification_weight=1.0,
            snr_weight=0.5
        )
        
        self.assertEqual(loss_fn.classification_weight, 1.0)
        self.assertEqual(loss_fn.snr_weight, 0.5)
    
    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = MultitaskLoss(
            classification_weight=1.0,
            snr_weight=0.5
        )
        
        losses = loss_fn(
            self.classification_logits,
            self.snr_estimates,
            self.classification_targets,
            self.snr_targets
        )
        
        # Check loss keys
        expected_keys = ['total_loss', 'classification_loss', 'snr_loss']
        for key in expected_keys:
            self.assertIn(key, losses)
        
        # Check that losses are positive
        for loss in losses.values():
            self.assertGreater(loss.item(), 0)
    
    def test_loss_weights(self):
        """Test loss weight effects."""
        # Test with different weights
        loss_fn1 = MultitaskLoss(classification_weight=1.0, snr_weight=0.5)
        loss_fn2 = MultitaskLoss(classification_weight=1.0, snr_weight=1.0)
        
        losses1 = loss_fn1(
            self.classification_logits,
            self.snr_estimates,
            self.classification_targets,
            self.snr_targets
        )
        
        losses2 = loss_fn2(
            self.classification_logits,
            self.snr_estimates,
            self.classification_targets,
            self.snr_targets
        )
        
        # Total loss should be different due to different SNR weights
        self.assertNotEqual(losses1['total_loss'].item(), losses2['total_loss'].item())
        
        # Classification loss should be the same
        self.assertEqual(losses1['classification_loss'].item(), losses2['classification_loss'].item())


class TestSpectrumAnalyzer(unittest.TestCase):
    """Test cases for SpectrumAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 20e6  # 20 MHz sampling rate
        self.signal_length = 128
        
        # Create synthetic I/Q signal
        t = np.linspace(0, self.signal_length / self.fs, self.signal_length)
        f1 = 2.4e9  # 2.4 GHz
        
        i_samples = np.cos(2 * np.pi * f1 * t)
        q_samples = np.sin(2 * np.pi * f1 * t)
        
        # Add noise
        noise_std = 0.1
        i_samples += np.random.normal(0, noise_std, len(i_samples))
        q_samples += np.random.normal(0, noise_std, len(q_samples))
        
        self.iq_signal = np.column_stack([i_samples, q_samples])
        
        self.analyzer = SpectrumAnalyzer(fs=self.fs)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.fs, self.fs)
        self.assertIsNotNone(self.analyzer.nperseg)
        self.assertIsNotNone(self.analyzer.noverlap)
        self.assertIsNotNone(self.analyzer.nfft)
    
    def test_iq_to_complex_conversion(self):
        """Test I/Q to complex conversion."""
        complex_signal = self.analyzer.iq_to_complex(self.iq_signal)
        
        self.assertEqual(len(complex_signal), self.signal_length)
        self.assertTrue(np.iscomplexobj(complex_signal))
    
    def test_psd_computation_welch(self):
        """Test PSD computation using Welch's method."""
        frequencies, psd = self.analyzer.compute_psd_welch(
            self.analyzer.iq_to_complex(self.iq_signal)
        )
        
        self.assertEqual(len(frequencies), len(psd))
        self.assertTrue(np.all(np.isfinite(psd)))
    
    def test_psd_computation_periodogram(self):
        """Test PSD computation using periodogram method."""
        frequencies, psd = self.analyzer.compute_psd_periodogram(
            self.analyzer.iq_to_complex(self.iq_signal)
        )
        
        self.assertEqual(len(frequencies), len(psd))
        self.assertTrue(np.all(np.isfinite(psd)))
    
    def test_peak_frequency_detection(self):
        """Test peak frequency detection."""
        complex_signal = self.analyzer.iq_to_complex(self.iq_signal)
        frequencies, psd = self.analyzer.compute_psd_welch(complex_signal)
        
        peak_freq, peak_power = self.analyzer.find_peak_frequency(frequencies, psd)
        
        self.assertIsInstance(peak_freq, float)
        self.assertIsInstance(peak_power, float)
        self.assertTrue(np.isfinite(peak_freq))
        self.assertTrue(np.isfinite(peak_power))
    
    def test_occupancy_ratio_computation(self):
        """Test occupancy ratio computation."""
        complex_signal = self.analyzer.iq_to_complex(self.iq_signal)
        frequencies, psd = self.analyzer.compute_psd_welch(complex_signal)
        
        occupancy_ratio = self.analyzer.compute_occupancy_ratio(psd)
        
        self.assertIsInstance(occupancy_ratio, float)
        self.assertGreaterEqual(occupancy_ratio, 0.0)
        self.assertLessEqual(occupancy_ratio, 1.0)
    
    def test_bandwidth_estimation(self):
        """Test bandwidth estimation."""
        complex_signal = self.analyzer.iq_to_complex(self.iq_signal)
        frequencies, psd = self.analyzer.compute_psd_welch(complex_signal)
        
        bandwidth = self.analyzer.estimate_bandwidth(frequencies, psd)
        
        self.assertIsInstance(bandwidth, float)
        self.assertGreaterEqual(bandwidth, 0.0)
    
    def test_snr_estimation(self):
        """Test SNR estimation."""
        complex_signal = self.analyzer.iq_to_complex(self.iq_signal)
        frequencies, psd = self.analyzer.compute_psd_welch(complex_signal)
        
        snr = self.analyzer.estimate_snr(psd)
        
        self.assertIsInstance(snr, float)
        self.assertTrue(np.isfinite(snr))
    
    def test_complete_spectrum_analysis(self):
        """Test complete spectrum analysis."""
        metrics = self.analyzer.analyze_spectrum(self.iq_signal)
        
        # Check that all metrics are present
        expected_attributes = [
            'psd', 'frequencies', 'peak_frequency', 'peak_power',
            'occupancy_ratio', 'average_power', 'bandwidth', 'snr_estimate'
        ]
        
        for attr in expected_attributes:
            self.assertTrue(hasattr(metrics, attr))
        
        # Check that values are finite
        for attr in expected_attributes:
            value = getattr(metrics, attr)
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value))


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 20e6
        self.signal_length = 128
        
        # Create synthetic I/Q signal
        t = np.linspace(0, self.signal_length / self.fs, self.signal_length)
        f1 = 2.4e9  # 2.4 GHz
        
        i_samples = np.cos(2 * np.pi * f1 * t)
        q_samples = np.sin(2 * np.pi * f1 * t)
        
        # Add noise
        noise_std = 0.1
        i_samples += np.random.normal(0, noise_std, len(i_samples))
        q_samples += np.random.normal(0, noise_std, len(q_samples))
        
        self.iq_signal = np.column_stack([i_samples, q_samples])
    
    def test_analyze_wifi_signal(self):
        """Test analyze_wifi_signal convenience function."""
        result = analyze_wifi_signal(self.iq_signal, fs=self.fs)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that required keys are present
        expected_keys = [
            'peak_frequency', 'peak_power', 'occupancy_ratio',
            'average_power', 'bandwidth', 'snr_estimate',
            'psd_frequencies', 'psd_values'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that values are finite
        for key in expected_keys:
            value = result[key]
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value))
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, (int, float)):
                        self.assertTrue(np.isfinite(v))


if __name__ == '__main__':
    unittest.main()


