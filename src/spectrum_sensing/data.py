"""
HDF5 loader for deep-learning spectrum sensing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.h5_loader import SignalTransform


class SpectrumSensingH5Dataset(Dataset):
    """
    Expects:
    - X: (N, L, 2)
    - y: (N, 4) multi-label occupancy
    - snr: (N,) optional
    """

    def __init__(self, h5_path: str, transform: SignalTransform | None = None) -> None:
        self.h5_path = str(h5_path)
        self.transform = transform
        with h5py.File(self.h5_path, "r") as f:
            if "X" not in f or "y" not in f:
                raise KeyError(f"{self.h5_path} must contain 'X' and 'y'")
            self.num_samples = int(f["X"].shape[0])
            self.has_snr = "snr" in f

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            x = np.asarray(f["X"][idx], dtype=np.float32)
            y = np.asarray(f["y"][idx], dtype=np.float32)
            snr = np.asarray(f["snr"][idx], dtype=np.float32) if self.has_snr else np.array(0.0, dtype=np.float32)
        if self.transform:
            x = self.transform(x)
        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        snr_t = torch.tensor(float(snr), dtype=torch.float32)
        return x_t, y_t, snr_t


def create_spectrum_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_transform: SignalTransform | None,
    val_transform: SignalTransform | None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base = Path(data_dir)
    train_ds = SpectrumSensingH5Dataset(str(base / "spectrum_train.h5"), transform=train_transform)
    val_ds = SpectrumSensingH5Dataset(str(base / "spectrum_val.h5"), transform=val_transform)
    test_ds = SpectrumSensingH5Dataset(str(base / "spectrum_test.h5"), transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
