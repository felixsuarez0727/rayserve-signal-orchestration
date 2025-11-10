"""
HDF5 dataset loader for SDR WiFi multitask training.

- Lazy loading with h5py
- Auto-detect keys: signals (iq/samples/signals/X), labels (y/labels/targets/classes), snr (snr)
- Supports group nesting (e.g., train/val/test)
- Returns DataLoaders with transforms
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SignalTransform:
    def __init__(
        self, 
        normalize: bool = True, 
        add_noise: bool = False, 
        noise_std: float = 0.01,
        time_shift: bool = True,
        freq_shift: bool = True,
        amplitude_scale: bool = True
    ):
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.freq_shift = freq_shift
        self.amplitude_scale = amplitude_scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x expected shape: (L, 2) or (2, L)
        if x.ndim != 2:
            raise ValueError("Signal must be 2D (I/Q vs time)")

        # Ensure shape (L, 2)
        if x.shape[0] == 2 and x.shape[1] != 2:
            x = x.T

        # Time shift augmentation (circular shift)
        if self.time_shift:
            max_shift = int(x.shape[0] * 0.1)  # Up to 10% shift
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                x = np.roll(x, shift, axis=0)

        # Frequency shift (applied as phase rotation in time domain)
        if self.freq_shift:
            freq_shift_std = 0.05  # 5% frequency shift
            phase_shift = np.random.normal(0.0, freq_shift_std) * 2 * np.pi
            t = np.arange(x.shape[0])
            rotation = np.exp(1j * phase_shift * t / x.shape[0])
            x_complex = x[:, 0] + 1j * x[:, 1]
            x_complex = x_complex * rotation
            x = np.stack([x_complex.real, x_complex.imag], axis=1)

        # Amplitude scaling
        if self.amplitude_scale:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        if self.normalize:
            std = np.std(x, axis=0) + 1e-8
            x = (x - np.mean(x, axis=0)) / std

        if self.add_noise:
            x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(x.dtype)

        return x


class SDRWiFiDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        group: Optional[str] = None,
        transform: Optional[SignalTransform] = None,
        one_hot_num_classes: Optional[int] = None,
    ) -> None:
        self.h5_path = str(h5_path)
        self._group_name = group
        self.transform = transform
        self.one_hot_num_classes = one_hot_num_classes

        with h5py.File(self.h5_path, 'r') as f:
            self.keys = self._detect_keys(f)
            self.num_samples = self._get_num_samples(f)

    def _detect_keys(self, h5_file: h5py.File) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        signal_patterns = ['iq', 'samples', 'signals', 'X']
        label_patterns = ['y', 'labels', 'targets', 'classes']
        snr_patterns = ['snr']

        all_keys = list(h5_file.keys())

        if self._group_name is None and len(all_keys) == 1 and isinstance(h5_file[all_keys[0]], h5py.Group):
            self._group_name = all_keys[0]

        group_obj = h5_file[self._group_name] if self._group_name else h5_file
        inner_keys = list(group_obj.keys())

        for pattern in signal_patterns:
            if pattern in inner_keys:
                keys['signals'] = pattern
                break
        for pattern in label_patterns:
            if pattern in inner_keys:
                keys['labels'] = pattern
                break
        for pattern in snr_patterns:
            if pattern in inner_keys:
                keys['snr'] = pattern
                break

        if 'signals' not in keys:
            raise KeyError("Unable to detect signals dataset in HDF5")
        if 'labels' not in keys:
            raise KeyError("Unable to detect labels dataset in HDF5")
        # snr is optional but recommended
        return keys

    def _get_num_samples(self, h5_file: h5py.File) -> int:
        group_obj = h5_file[self._group_name] if self._group_name else h5_file
        return group_obj[self.keys['signals']].shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, 'r') as f:
            group_obj = f[self._group_name] if self._group_name else f
            x = group_obj[self.keys['signals']][idx]
            y = group_obj[self.keys['labels']][idx]
            
            # Load SNR if available
            snr = None
            if 'snr' in self.keys:
                snr = group_obj[self.keys['snr']][idx]

            # Ensure (L, 2)
            x = np.asarray(x)
            if x.ndim == 3:
                x = x.squeeze()
            if x.shape[0] == 2 and x.shape[1] != 2:
                x = x.T

            if self.transform:
                x = self.transform(x)

            # Convert multi-hot labels to binary: WiFi (sum>0) vs Noise (sum=0)
            y = np.asarray(y)
            
            # If label is multi-hot (4 dimensions), convert to binary
            if y.ndim == 1 and len(y) == 4:
                # Multi-hot label: convert to binary (WiFi=1 if any class is 1, Noise=0 if all zeros)
                label_sum = np.sum(y)
                y_binary_val = 1 if label_sum > 0 else 0  # WiFi=1 if sum>0, Noise=0 if sum=0
            elif y.ndim == 0:
                # Single scalar value
                y_binary_val = int(y)
            elif y.ndim == 1 and len(y) == 1:
                # Already a single value
                y_binary_val = int(y[0])
            else:
                # Fallback: assume it's already binary
                y_binary_val = int(y[0]) if len(y) > 0 else 0
            
            # Convert to one-hot if requested (for binary classification: [Noise, WiFi])
            if self.one_hot_num_classes:
                oh = np.zeros(self.one_hot_num_classes, dtype=np.float32)
                oh[y_binary_val] = 1.0
                y = oh
            else:
                y = np.array([y_binary_val], dtype=np.float32)

            x_t = torch.from_numpy(x.astype(np.float32))  # (L, 2)
            y_t = torch.from_numpy(np.asarray(y).astype(np.float32))
            
            # Handle SNR: return real value if available, else dummy
            if snr is not None:
                snr_t = torch.from_numpy(np.asarray(snr).astype(np.float32))
                if snr_t.ndim == 0:
                    snr_t = snr_t.unsqueeze(0)
            else:
                # Fallback: generate dummy SNR (should not happen if dataset has SNR)
                snr_t = torch.tensor([20.0], dtype=torch.float32)
            
            return x_t, y_t, snr_t


def create_data_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_transform: Optional[SignalTransform] = None,
    val_transform: Optional[SignalTransform] = None,
    lazy_loading: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test. Assumes standard file names.
    """
    data_dir_p = Path(data_dir)
    train_path = data_dir_p / 'sdr_wifi_train.h5'
    val_path = data_dir_p / 'sdr_wifi_val.h5'
    test_path = data_dir_p / 'sdr_wifi_test.h5'

    # Attempt to infer group (train/val/test) automatically if needed
    def infer_group(h5_path: Path) -> Optional[str]:
        with h5py.File(h5_path, 'r') as f:
            ks = list(f.keys())
            if len(ks) == 1 and isinstance(f[ks[0]], h5py.Group):
                return ks[0]
        return None

    train_ds = SDRWiFiDataset(
        str(train_path), group=infer_group(train_path), transform=train_transform, one_hot_num_classes=2
    )
    val_ds = SDRWiFiDataset(
        str(val_path), group=infer_group(val_path), transform=val_transform, one_hot_num_classes=2
    )
    test_ds = SDRWiFiDataset(
        str(test_path), group=infer_group(test_path), transform=val_transform, one_hot_num_classes=2
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader




