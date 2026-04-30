"""
Grid-search tuning for downstream link adaptation thresholds.

Runs on real test data and reports the best policy configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

from src.downstream.link_adaptation import LinkAdaptationModule
from src.orchestrator.pipeline import SignalOrchestrator
from src.models.multitask_net import create_model
from src.utils.logger import load_config
from src.utils.seed import get_device


MCS_TO_INDEX = {f"MCS{i}": i for i in range(10)}


def normalize_batch(xb: np.ndarray) -> np.ndarray:
    mean = np.mean(xb, axis=1, keepdims=True)
    std = np.std(xb, axis=1, keepdims=True) + 1e-8
    return (xb - mean) / std


def collect_predictions(
    model: torch.nn.Module,
    class_names: List[str],
    h5_path: Path,
    device: torch.device,
    batch_size: int,
) -> List[Dict]:
    records: List[Dict] = []
    with h5py.File(h5_path, "r") as f:
        x = f["X"]
        n = x.shape[0]
        for i in range(0, n, batch_size):
            xb = np.asarray(x[i : i + batch_size], dtype=np.float32)
            xb = normalize_batch(xb)
            xt = torch.from_numpy(xb).to(device)
            with torch.no_grad():
                logits, snr = model(xt)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
            for j in range(pred.shape[0]):
                cls_idx = int(pred[j].item())
                conf = float(probs[j, cls_idx].item())
                snr_est = float(snr[j].item())
                records.append(
                    {
                        "predicted_class_name": class_names[cls_idx],
                        "confidence": conf,
                        "snr_estimate": snr_est,
                    }
                )
    return records


def entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values() if v > 0]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def evaluate_policy(records: List[Dict], min_confidence: float, snr_offset_db: float) -> Dict:
    orch = SignalOrchestrator(
        LinkAdaptationModule(min_confidence=min_confidence, snr_offset_db=snr_offset_db)
    )

    status_counts = Counter()
    mcs_counts = Counter()
    mcs_indices = []

    for rec in records:
        d = orch.route(rec)["downstream"]
        st = d.get("status", "unknown")
        status_counts[st] += 1
        mcs = d.get("recommended_mcs")
        if mcs:
            mcs_counts[mcs] += 1
            mcs_indices.append(MCS_TO_INDEX.get(mcs, 0))

    total = len(records)
    ok = status_counts.get("ok", 0)
    defer = status_counts.get("deferred", 0)
    skip = status_counts.get("skipped", 0)

    ok_rate = ok / total if total else 0.0
    defer_rate = defer / total if total else 0.0
    skip_rate = skip / total if total else 0.0
    mean_mcs_idx = float(np.mean(mcs_indices)) if mcs_indices else 0.0
    mean_mcs_norm = mean_mcs_idx / 9.0
    ent = entropy_from_counts(mcs_counts)
    ent_norm = ent / math.log(10.0) if ent > 0 else 0.0

    # Heuristic utility: prioritize useful high-rate decisions, penalize excessive defer.
    utility = (0.65 * mean_mcs_norm) + (0.20 * ok_rate) + (0.15 * ent_norm) - (0.35 * defer_rate)

    return {
        "min_confidence": min_confidence,
        "snr_offset_db": snr_offset_db,
        "utility": round(utility, 6),
        "ok_rate": round(ok_rate, 6),
        "defer_rate": round(defer_rate, 6),
        "skip_rate": round(skip_rate, 6),
        "mean_mcs_index": round(mean_mcs_idx, 6),
        "mcs_entropy_norm": round(ent_norm, 6),
        "status_counts": dict(status_counts),
        "mcs_counts": dict(mcs_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune link adaptation policy on real test data")
    parser.add_argument("--config", type=str, default="conf/config_wifi_noise.yaml")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/main/best_checkpoint.pth")
    parser.add_argument("--h5", type=str, default="sdr_wifi_test.hdf5")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output", type=str, default="artifacts/logs/run/link_adaptation_tuning.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device("cpu")
    model, _ = create_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_names = cfg["dataset"]["class_names"]
    records = collect_predictions(
        model=model,
        class_names=class_names,
        h5_path=Path(args.h5),
        device=device,
        batch_size=args.batch_size,
    )

    conf_grid = [0.55, 0.60, 0.65, 0.70, 0.75]
    offset_grid = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0]
    runs = []
    for conf in conf_grid:
        for offset in offset_grid:
            runs.append(evaluate_policy(records, conf, offset))

    runs_sorted = sorted(runs, key=lambda r: r["utility"], reverse=True)
    baseline = next(r for r in runs_sorted if r["min_confidence"] == 0.60 and r["snr_offset_db"] == 0.0)
    best = runs_sorted[0]

    result = {
        "dataset": args.h5,
        "samples": len(records),
        "search_space": {"min_confidence": conf_grid, "snr_offset_db": offset_grid},
        "baseline": baseline,
        "best": best,
        "top5": runs_sorted[:5],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["best"], indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
