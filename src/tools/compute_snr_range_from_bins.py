"""
Compute SNR range directly from binary dataset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def detect_dtype(file_path: Path) -> np.dtype:
    """
    Heuristic detection between float32 interleaved IQ and int16 interleaved IQ.
    """
    preview = np.fromfile(file_path, dtype=np.float32, count=4096)
    if preview.size >= 2 and np.isfinite(preview).all() and np.std(preview) > 1e-8:
        return np.float32
    return np.int16


def update_stats(acc: Dict[str, float], values: np.ndarray) -> None:
    if values.size == 0:
        return
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    acc["min"] = min(acc["min"], v_min)
    acc["max"] = max(acc["max"], v_max)
    acc["sum"] += float(np.sum(values))
    acc["count"] += int(values.size)


def compute_file_snr(file_path: Path, signal_length: int, samples_per_block: int) -> np.ndarray:
    dtype = detect_dtype(file_path)
    data = np.memmap(file_path, dtype=dtype, mode="r")
    if dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)

    total_pairs = (data.size // 2)
    total_samples = total_pairs // signal_length
    if total_samples <= 0:
        return np.empty((0,), dtype=np.float32)

    snr_values = []
    offset_pairs = 0
    remaining_samples = total_samples

    while remaining_samples > 0:
        block_samples = min(samples_per_block, remaining_samples)
        pairs_needed = block_samples * signal_length
        raw = data[offset_pairs * 2 : (offset_pairs + pairs_needed) * 2]
        iq = np.asarray(raw, dtype=np.float32).reshape(block_samples, signal_length, 2)
        power = np.sum(iq * iq, axis=2)
        signal_power = np.mean(power, axis=1)
        noise_floor = np.percentile(power, 10.0, axis=1)
        snr = 10.0 * np.log10((signal_power + 1e-12) / (noise_floor + 1e-12))
        snr_values.append(snr.astype(np.float32))

        offset_pairs += pairs_needed
        remaining_samples -= block_samples

    return np.concatenate(snr_values, axis=0) if snr_values else np.empty((0,), dtype=np.float32)


def init_stats() -> Dict[str, float]:
    return {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}


def finalize(stats: Dict[str, float]) -> Dict[str, float]:
    if stats["count"] == 0:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["sum"] / stats["count"],
        "count": stats["count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SNR range from raw .bin files")
    parser.add_argument("--input_dir", type=str, default="neu_bz61g073z")
    parser.add_argument("--signal_length", type=int, default=128)
    parser.add_argument("--samples_per_block", type=int, default=50000)
    parser.add_argument("--output", type=str, default="logs/snr_range_from_bins.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = sorted(input_dir.glob("*.bin"))
    if not files:
        raise RuntimeError(f"No .bin files found in {input_dir}")

    global_stats = init_stats()
    day1_stats = init_stats()
    day2_stats = init_stats()
    per_file = {}

    for fp in files:
        snr = compute_file_snr(fp, args.signal_length, args.samples_per_block)
        file_stats = {
            "min": float(np.min(snr)) if snr.size else None,
            "max": float(np.max(snr)) if snr.size else None,
            "mean": float(np.mean(snr)) if snr.size else None,
            "count": int(snr.size),
        }
        per_file[fp.name] = file_stats
        update_stats(global_stats, snr)
        if "_day1" in fp.name:
            update_stats(day1_stats, snr)
        elif "_day2" in fp.name:
            update_stats(day2_stats, snr)

    result = {
        "input_dir": str(input_dir),
        "signal_length": args.signal_length,
        "global": finalize(global_stats),
        "day1": finalize(day1_stats),
        "day2": finalize(day2_stats),
        "per_file": per_file,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
