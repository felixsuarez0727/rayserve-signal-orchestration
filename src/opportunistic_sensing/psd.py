"""
Opportunistic Spectrum Sensing Module for WiFi Signals
Calculates Power Spectral Density (PSD) and channel occupancy metrics
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpectrumMetrics:
    """Data class for spectrum sensing metrics."""
    psd: np.ndarray
    frequencies: np.ndarray
    peak_frequency: float
    peak_power: float
    occupancy_ratio: float
    average_power: float
    bandwidth: float
    snr_estimate: float


class SpectrumAnalyzer:
    """
    Analyzer for opportunistic spectrum sensing of WiFi signals.
    """
    
    def __init__(
        self,
        fs: float = 20e6,  # Sampling frequency in Hz
        nperseg: int = 64,
        noverlap: int = 32,
        nfft: int = 128,
        threshold_db: float = -80
    ):
        """
        Initialize the spectrum analyzer.
        
        Args:
            fs: Sampling frequency in Hz
            nperseg: Length of each segment for Welch's method
            noverlap: Number of points to overlap between segments
            nfft: Length of FFT used
            threshold_db: Threshold in dB for occupancy detection
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.threshold_db = threshold_db
        
        logger.info(f"SpectrumAnalyzer initialized with fs={fs/1e6:.1f} MHz")
    
    def iq_to_complex(self, iq_signal: np.ndarray) -> np.ndarray:
        """
        Convert I/Q samples to complex signal.
        
        Args:
            iq_signal: I/Q signal of shape (N, 2) or (2, N)
            
        Returns:
            Complex signal
        """
        if iq_signal.shape[0] == 2:
            # Shape is (2, N)
            i_samples = iq_signal[0, :]
            q_samples = iq_signal[1, :]
        else:
            # Shape is (N, 2)
            i_samples = iq_signal[:, 0]
            q_samples = iq_signal[:, 1]
        
        return i_samples + 1j * q_samples
    
    def compute_psd_welch(
        self,
        signal_data: np.ndarray,
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            signal_data: Complex signal data
            window: Window function for Welch's method
            
        Returns:
            Tuple of (frequencies, psd)
        """
        frequencies, psd = signal.welch(
            signal_data,
            fs=self.fs,
            window=window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            return_onesided=False
        )
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-12)  # Add small value to avoid log(0)
        
        return frequencies, psd_db
    
    def compute_psd_periodogram(
        self,
        signal_data: np.ndarray,
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using periodogram method.
        
        Args:
            signal_data: Complex signal data
            window: Window function
            
        Returns:
            Tuple of (frequencies, psd)
        """
        frequencies, psd = signal.periodogram(
            signal_data,
            fs=self.fs,
            window=window,
            nfft=self.nfft,
            return_onesided=False
        )
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-12)
        
        return frequencies, psd_db
    
    def find_peak_frequency(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        min_prominence: float = 3.0
    ) -> Tuple[float, float]:
        """
        Find the peak frequency and its power.
        
        Args:
            frequencies: Frequency array
            psd: Power spectral density in dB
            min_prominence: Minimum prominence for peak detection
            
        Returns:
            Tuple of (peak_frequency, peak_power)
        """
        # Find peaks
        peaks, properties = signal.find_peaks(
            psd,
            prominence=min_prominence,
            height=np.max(psd) - 20  # At least 20 dB below max
        )
        
        if len(peaks) == 0:
            # No peaks found, return the frequency with maximum power
            max_idx = np.argmax(psd)
            return frequencies[max_idx], psd[max_idx]
        
        # Return the highest peak
        peak_idx = peaks[np.argmax(psd[peaks])]
        return frequencies[peak_idx], psd[peak_idx]
    
    def compute_occupancy_ratio(
        self,
        psd: np.ndarray,
        threshold_db: Optional[float] = None
    ) -> float:
        """
        Compute channel occupancy ratio.
        
        Args:
            psd: Power spectral density in dB
            threshold_db: Threshold for occupancy detection
            
        Returns:
            Occupancy ratio (0-1)
        """
        if threshold_db is None:
            threshold_db = self.threshold_db
        
        # Count frequencies above threshold
        occupied_bins = np.sum(psd > threshold_db)
        total_bins = len(psd)
        
        return occupied_bins / total_bins
    
    def estimate_bandwidth(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        threshold_db: Optional[float] = None
    ) -> float:
        """
        Estimate signal bandwidth.
        
        Args:
            frequencies: Frequency array
            psd: Power spectral density in dB
            threshold_db: Threshold for bandwidth calculation
            
        Returns:
            Bandwidth in Hz
        """
        if threshold_db is None:
            threshold_db = self.threshold_db
        
        # Find frequencies above threshold
        above_threshold = psd > threshold_db
        
        if not np.any(above_threshold):
            return 0.0
        
        # Find the first and last frequencies above threshold
        occupied_indices = np.where(above_threshold)[0]
        first_idx = occupied_indices[0]
        last_idx = occupied_indices[-1]
        
        bandwidth = frequencies[last_idx] - frequencies[first_idx]
        
        return bandwidth
    
    def estimate_snr(
        self,
        psd: np.ndarray,
        noise_floor_percentile: float = 10.0
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio from PSD.
        
        Args:
            psd: Power spectral density in dB
            noise_floor_percentile: Percentile to use as noise floor
            
        Returns:
            SNR estimate in dB
        """
        # Estimate noise floor from lower percentile
        noise_floor = np.percentile(psd, noise_floor_percentile)
        
        # Signal power is the maximum
        signal_power = np.max(psd)
        
        # SNR is the difference
        snr = signal_power - noise_floor
        
        return snr
    
    def analyze_spectrum(
        self,
        iq_signal: np.ndarray,
        method: str = 'welch'
    ) -> SpectrumMetrics:
        """
        Perform complete spectrum analysis.
        
        Args:
            iq_signal: I/Q signal data
            method: PSD computation method ('welch' or 'periodogram')
            
        Returns:
            SpectrumMetrics object with all computed metrics
        """
        # Convert to complex signal
        complex_signal = self.iq_to_complex(iq_signal)
        
        # Compute PSD
        if method == 'welch':
            frequencies, psd = self.compute_psd_welch(complex_signal)
        elif method == 'periodogram':
            frequencies, psd = self.compute_psd_periodogram(complex_signal)
        else:
            raise ValueError(f"Unknown PSD method: {method}")
        
        # Find peak frequency and power
        peak_freq, peak_power = self.find_peak_frequency(frequencies, psd)
        
        # Compute occupancy ratio
        occupancy_ratio = self.compute_occupancy_ratio(psd)
        
        # Compute average power
        average_power = np.mean(psd)
        
        # Estimate bandwidth
        bandwidth = self.estimate_bandwidth(frequencies, psd)
        
        # Estimate SNR
        snr_estimate = self.estimate_snr(psd)
        
        return SpectrumMetrics(
            psd=psd,
            frequencies=frequencies,
            peak_frequency=peak_freq,
            peak_power=peak_power,
            occupancy_ratio=occupancy_ratio,
            average_power=average_power,
            bandwidth=bandwidth,
            snr_estimate=snr_estimate
        )
    
    def detect_wifi_channels(
        self,
        frequencies: np.ndarray,
        psd: np.ndarray,
        wifi_channels: Optional[list] = None
    ) -> Dict[str, Dict]:
        """
        Detect WiFi channels and their characteristics.
        
        Args:
            frequencies: Frequency array
            psd: Power spectral density in dB
            wifi_channels: List of WiFi channel center frequencies (Hz)
            
        Returns:
            Dictionary with channel information
        """
        if wifi_channels is None:
            # Common WiFi channels (2.4 GHz band)
            wifi_channels = [
                2412e6, 2417e6, 2422e6, 2427e6, 2432e6, 2437e6, 2442e6,
                2447e6, 2452e6, 2457e6, 2462e6, 2467e6, 2472e6
            ]
        
        channel_info = {}
        
        for i, channel_freq in enumerate(wifi_channels):
            # Find frequency bins around this channel
            channel_bandwidth = 20e6  # 20 MHz for WiFi
            freq_mask = np.abs(frequencies - channel_freq) <= channel_bandwidth / 2
            
            if np.any(freq_mask):
                channel_psd = psd[freq_mask]
                channel_freqs = frequencies[freq_mask]
                
                channel_info[f"channel_{i+1}"] = {
                    'center_frequency': channel_freq,
                    'max_power': np.max(channel_psd),
                    'average_power': np.mean(channel_psd),
                    'bandwidth': channel_bandwidth,
                    'occupied': np.max(channel_psd) > self.threshold_db
                }
        
        return channel_info


def analyze_wifi_signal(
    iq_signal: np.ndarray,
    fs: float = 20e6,
    method: str = 'welch'
) -> Dict:
    """
    Convenience function to analyze WiFi signal spectrum.
    
    Args:
        iq_signal: I/Q signal data
        fs: Sampling frequency in Hz
        method: PSD computation method
        
    Returns:
        Dictionary with spectrum analysis results
    """
    analyzer = SpectrumAnalyzer(fs=fs)
    metrics = analyzer.analyze_spectrum(iq_signal, method=method)
    
    # Convert to dictionary for JSON serialization
    result = {
        'peak_frequency': float(metrics.peak_frequency),
        'peak_power': float(metrics.peak_power),
        'occupancy_ratio': float(metrics.occupancy_ratio),
        'average_power': float(metrics.average_power),
        'bandwidth': float(metrics.bandwidth),
        'snr_estimate': float(metrics.snr_estimate),
        'psd_frequencies': metrics.frequencies.tolist(),
        'psd_values': metrics.psd.tolist()
    }
    
    return result


if __name__ == "__main__":
    # Test the spectrum analyzer
    import matplotlib.pyplot as plt
    
    # Generate a test WiFi-like signal
    fs = 20e6  # 20 MHz sampling rate
    duration = 0.001  # 1 ms
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create a signal with multiple frequency components
    f1 = 2.4e9  # 2.4 GHz
    f2 = 2.42e9  # 2.42 GHz
    
    # Generate I/Q samples
    i_samples = np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)
    q_samples = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Add noise
    noise_std = 0.1
    i_samples += np.random.normal(0, noise_std, len(i_samples))
    q_samples += np.random.normal(0, noise_std, len(q_samples))
    
    # Combine I/Q samples
    iq_signal = np.column_stack([i_samples, q_samples])
    
    # Analyze spectrum
    analyzer = SpectrumAnalyzer(fs=fs)
    metrics = analyzer.analyze_spectrum(iq_signal)
    
    print("Spectrum Analysis Results:")
    print(f"Peak Frequency: {metrics.peak_frequency/1e6:.2f} MHz")
    print(f"Peak Power: {metrics.peak_power:.2f} dB")
    print(f"Occupancy Ratio: {metrics.occupancy_ratio:.3f}")
    print(f"Average Power: {metrics.average_power:.2f} dB")
    print(f"Bandwidth: {metrics.bandwidth/1e6:.2f} MHz")
    print(f"SNR Estimate: {metrics.snr_estimate:.2f} dB")
    
    # Plot spectrum
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(metrics.frequencies/1e6, metrics.psd)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Power Spectral Density')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t*1e6, i_samples, label='I')
    plt.plot(t*1e6, q_samples, label='Q')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.title('I/Q Signal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test convenience function
    result = analyze_wifi_signal(iq_signal, fs=fs)
    print(f"\nConvenience function result keys: {list(result.keys())}")


