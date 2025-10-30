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
    def __init__(self, normalize: bool = True, add_noise: bool = False, noise_std: float = 0.01):
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x expected shape: (L, 2) or (2, L)
        if x.ndim != 2:
            raise ValueError("Signal must be 2D (I/Q vs time)")

        # Ensure shape (L, 2)
        if x.shape[0] == 2 and x.shape[1] != 2:
            x = x.T

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, 'r') as f:
            group_obj = f[self._group_name] if self._group_name else f
            x = group_obj[self.keys['signals']][idx]
            y = group_obj[self.keys['labels']][idx]

            # Ensure (L, 2)
            x = np.asarray(x)
            if x.ndim == 3:
                x = x.squeeze()
            if x.shape[0] == 2 and x.shape[1] != 2:
                x = x.T

            if self.transform:
                x = self.transform(x)

            # one-hot if requested
            if self.one_hot_num_classes and y.ndim == 0:
                oh = np.zeros(self.one_hot_num_classes, dtype=np.float32)
                oh[int(y)] = 1.0
                y = oh

            x_t = torch.from_numpy(x.astype(np.float32))  # (L, 2)
            y_t = torch.from_numpy(np.asarray(y).astype(np.float32))
            return x_t, y_t


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
        str(train_path), group=infer_group(train_path), transform=train_transform, one_hot_num_classes=4
    )
    val_ds = SDRWiFiDataset(
        str(val_path), group=infer_group(val_path), transform=val_transform, one_hot_num_classes=4
    )
    test_ds = SDRWiFiDataset(
        str(test_path), group=infer_group(test_path), transform=val_transform, one_hot_num_classes=4
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


