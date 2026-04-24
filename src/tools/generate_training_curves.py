"""
Generate training curves from logs/training.log.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
TRAIN_RE = re.compile(r"Train Loss:\s*([0-9]*\.?[0-9]+),\s*Train Acc:\s*([0-9]*\.?[0-9]+)")
VAL_RE = re.compile(r"Val Loss:\s*([0-9]*\.?[0-9]+),\s*Val Acc:\s*([0-9]*\.?[0-9]+)")


def parse_training_log(log_path: Path) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Parse epochs, train/val loss and train/val accuracy."""
    epochs: List[int] = []
    train_loss: List[float] = []
    val_loss: List[float] = []
    train_acc: List[float] = []
    val_acc: List[float] = []

    current_epoch = None
    train_pending = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        train_match = TRAIN_RE.search(line)
        if train_match and current_epoch is not None:
            train_pending = (
                float(train_match.group(1)),
                float(train_match.group(2)),
            )
            continue

        val_match = VAL_RE.search(line)
        if val_match and current_epoch is not None and train_pending is not None:
            epochs.append(current_epoch)
            train_loss.append(train_pending[0])
            train_acc.append(train_pending[1])
            val_loss.append(float(val_match.group(1)))
            val_acc.append(float(val_match.group(2)))
            train_pending = None

    if not epochs:
        raise ValueError(f"No epoch metrics found in {log_path}")

    return epochs, train_loss, val_loss, train_acc, val_acc


def plot_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
    train_acc: List[float],
    val_acc: List[float],
    output_path: Path,
) -> None:
    """Plot and save combined training curves figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="Validation Loss", linewidth=1.8)
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy", linewidth=1.8)
    axes[1].plot(epochs, val_acc, label="Validation Accuracy", linewidth=1.8)
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training curves from a training log")
    parser.add_argument("--log", type=str, default="logs/training.log", help="Path to training.log")
    parser.add_argument(
        "--output",
        type=str,
        default="Images/training_curves.png",
        help="Output image path",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    parsed = parse_training_log(log_path)
    plot_curves(*parsed, output_path=Path(args.output))
    print(f"Training curves saved to: {args.output}")


if __name__ == "__main__":
    main()
