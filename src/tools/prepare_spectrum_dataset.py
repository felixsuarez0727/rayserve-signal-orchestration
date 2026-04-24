"""
Prepare HDF5 dataset for deep-learning spectrum sensing from .bin files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


@dataclass
class SplitBuffers:
    x: List[np.ndarray]
    y: List[np.ndarray]
    snr: List[np.ndarray]


def parse_label_from_name(file_name: str) -> np.ndarray:
    stem = Path(file_name).stem
    bits = stem.split("_")[0]
    if len(bits) != 4 or any(ch not in "01" for ch in bits):
        raise ValueError(f"Unexpected file naming pattern: {file_name}")
    return np.array([int(b) for b in bits], dtype=np.float32)


def infer_split(file_name: str) -> str:
    # Keep day2 for test to preserve day-based separation.
    if "_day2" in file_name:
        return "test"
    # day1 is split between train/val by deterministic hash.
    return "train"


def split_day1_train_val(
    x: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    val_ratio: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = x.shape[0]
    if n == 0:
        empty_x = np.empty((0, 128, 2), dtype=np.float32)
        empty_y = np.empty((0, 4), dtype=np.float32)
        empty_snr = np.empty((0,), dtype=np.float32)
        return (empty_x, empty_y, empty_snr), (empty_x, empty_y, empty_snr)
    idx = np.arange(n)
    # deterministic split
    val_mask = (idx % 100) < int(val_ratio * 100)
    train_mask = ~val_mask
    return (
        x[train_mask],
        y[train_mask],
        snr[train_mask],
    ), (
        x[val_mask],
        y[val_mask],
        snr[val_mask],
    )


def read_iq_pairs(bin_path: Path) -> np.ndarray:
    """
    Read binary as interleaved float32 I/Q pairs, fallback to int16.
    """
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size >= 2 and np.isfinite(raw).all():
        usable = (raw.size // 2) * 2
        x = raw[:usable].reshape(-1, 2)
        # fallback heuristic when values are almost all zeros
        if np.std(x) > 1e-8:
            return x.astype(np.float32)
    raw_i16 = np.fromfile(bin_path, dtype=np.int16).astype(np.float32)
    usable = (raw_i16.size // 2) * 2
    x = raw_i16[:usable].reshape(-1, 2)
    # Normalize int16 range.
    return (x / 32768.0).astype(np.float32)


def estimate_snr_per_chunk(chunks: np.ndarray) -> np.ndarray:
    """
    Fast SNR estimate from instantaneous power statistics.
    """
    power = np.sum(chunks * chunks, axis=2)  # (N, L)
    signal_power = np.mean(power, axis=1)
    noise_floor = np.percentile(power, 10.0, axis=1)
    snr = 10.0 * np.log10((signal_power + 1e-12) / (noise_floor + 1e-12))
    return snr.astype(np.float32)


def to_chunks(iq: np.ndarray, signal_length: int, stride: int, max_chunks: int | None) -> np.ndarray:
    n = iq.shape[0]
    if n < signal_length:
        return np.empty((0, signal_length, 2), dtype=np.float32)
    starts = np.arange(0, n - signal_length + 1, stride, dtype=np.int64)
    if max_chunks is not None:
        starts = starts[:max_chunks]
    chunks = np.stack([iq[s : s + signal_length] for s in starts], axis=0)
    return chunks.astype(np.float32)


def save_h5(path: Path, x: np.ndarray, y: np.ndarray, snr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=x, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")
        f.create_dataset("snr", data=snr, compression="gzip")


def summarize(name: str, x: np.ndarray, snr: np.ndarray) -> Dict[str, float]:
    if x.shape[0] == 0:
        return {"split": name, "samples": 0}
    return {
        "split": name,
        "samples": int(x.shape[0]),
        "snr_min": float(np.min(snr)),
        "snr_max": float(np.max(snr)),
        "snr_mean": float(np.mean(snr)),
        "snr_p5": float(np.percentile(snr, 5)),
        "snr_p95": float(np.percentile(snr, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare spectrum sensing dataset from .bin files")
    parser.add_argument("--input_dir", type=str, default="neu_bz61g073z")
    parser.add_argument("--output_dir", type=str, default="processed_spectrum")
    parser.add_argument("--signal_length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--val_ratio_day1", type=float, default=0.15)
    parser.add_argument("--max_chunks_per_file", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    day1_x, day1_y, day1_snr = [], [], []
    test_x, test_y, test_snr = [], [], []

    for bin_path in sorted(input_dir.glob("*.bin")):
        label = parse_label_from_name(bin_path.name)
        split = infer_split(bin_path.name)
        iq = read_iq_pairs(bin_path)
        chunks = to_chunks(iq, args.signal_length, args.stride, args.max_chunks_per_file)
        if chunks.shape[0] == 0:
            continue
        snr = estimate_snr_per_chunk(chunks)
        labels = np.repeat(label[None, :], chunks.shape[0], axis=0)
        if split == "test":
            test_x.append(chunks)
            test_y.append(labels)
            test_snr.append(snr)
        else:
            day1_x.append(chunks)
            day1_y.append(labels)
            day1_snr.append(snr)

    if not day1_x or not test_x:
        raise RuntimeError("No samples found. Verify bin format and paths.")

    day1_x_arr = np.concatenate(day1_x, axis=0)
    day1_y_arr = np.concatenate(day1_y, axis=0)
    day1_snr_arr = np.concatenate(day1_snr, axis=0)
    test_x_arr = np.concatenate(test_x, axis=0)
    test_y_arr = np.concatenate(test_y, axis=0)
    test_snr_arr = np.concatenate(test_snr, axis=0)

    (train_x, train_y, train_snr), (val_x, val_y, val_snr) = split_day1_train_val(
        day1_x_arr, day1_y_arr, day1_snr_arr, args.val_ratio_day1
    )

    save_h5(output_dir / "spectrum_train.h5", train_x, train_y, train_snr)
    save_h5(output_dir / "spectrum_val.h5", val_x, val_y, val_snr)
    save_h5(output_dir / "spectrum_test.h5", test_x_arr, test_y_arr, test_snr_arr)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "signal_length": args.signal_length,
        "stride": args.stride,
        "splits": [
            summarize("train", train_x, train_snr),
            summarize("val", val_x, val_snr),
            summarize("test", test_x_arr, test_snr_arr),
        ],
    }
    all_snr = np.concatenate([train_snr, val_snr, test_snr_arr], axis=0)
    summary["global_snr"] = {
        "min": float(np.min(all_snr)),
        "max": float(np.max(all_snr)),
        "mean": float(np.mean(all_snr)),
        "p5": float(np.percentile(all_snr, 5)),
        "p95": float(np.percentile(all_snr, 95)),
    }

    out_summary = output_dir / "spectrum_dataset_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()
