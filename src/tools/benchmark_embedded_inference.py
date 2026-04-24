"""
Benchmark multitask model inference latency for edge-like deployments.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch

from src.models.multitask_net import create_model
from src.utils.logger import load_config


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[index]


def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize linear layers for CPU-oriented deployment."""
    supported = [engine for engine in torch.backends.quantized.supported_engines if engine and engine != "none"]
    if not supported:
        raise RuntimeError(
            "No quantization engine available in this PyTorch build. "
            "Run without --quantize or use a build with qnnpack/fbgemm support."
        )

    if torch.backends.quantized.engine not in supported:
        preferred_order = ["qnnpack", "fbgemm"]
        selected = next((engine for engine in preferred_order if engine in supported), supported[0])
        torch.backends.quantized.engine = selected

    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def run_benchmark(
    model: torch.nn.Module,
    signal_length: int,
    input_channels: int,
    batch_size: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sample = torch.randn(batch_size, input_channels, signal_length, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)

    timings_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

    mean_ms = statistics.mean(timings_ms)
    p50_ms = percentile(timings_ms, 50)
    p95_ms = percentile(timings_ms, 95)
    fps = (1000.0 / mean_ms) * batch_size if mean_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "throughput_samples_per_sec": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference for embedded/edge deployment")
    parser.add_argument("--config", type=str, default="conf/config_wifi_noise.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (edge-like: 1)")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic int8 quantization")
    parser.add_argument("--output", type=str, default="logs/embedded_inference_benchmark.json")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)

    model, _ = create_model(config, device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.quantize:
        if device.type != "cpu":
            raise ValueError("Dynamic quantization is only supported for CPU in this script")
        model = apply_dynamic_quantization(model)

    model = model.to(device)

    benchmark = run_benchmark(
        model=model,
        signal_length=config["model"]["signal_length"],
        input_channels=config["model"]["input_channels"],
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    result = {
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "device": args.device,
        "threads": args.threads,
        "quantized": bool(args.quantize),
        "total_parameters": total_params,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "benchmark": benchmark,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nBenchmark saved to: {output_path}")


if __name__ == "__main__":
    main()
