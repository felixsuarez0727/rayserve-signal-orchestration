"""
Train deep-learning model for spectrum sensing (multi-label occupancy).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.h5_loader import SignalTransform
from src.data.spectrum_h5_loader import create_spectrum_loaders
from src.models.spectrum_sensing_net import SpectrumSensingLoss, SpectrumSensingNet
from src.utils.logger import load_config, setup_logging
from src.utils.metrics import save_metrics
from src.utils.seed import get_device, set_seed


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    exact_match = float((preds == targets).all(dim=1).float().mean().item())
    y_true = targets.detach().cpu().numpy().astype(int)
    y_pred = preds.detach().cpu().numpy().astype(int)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"exact_match": exact_match, "macro_f1": macro_f1}


def run_epoch(
    model: SpectrumSensingNet,
    loader,
    loss_fn: SpectrumSensingLoss,
    device: torch.device,
    optimizer=None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    all_logits = []
    all_targets = []
    all_snr_pred = []
    all_snr_true = []

    with torch.set_grad_enabled(training):
        for x, y, snr in tqdm(loader, disable=False):
            x = x.to(device)
            y = y.to(device)
            snr = snr.to(device)
            if training:
                optimizer.zero_grad()
            out = model(x)
            ldict = loss_fn(out, y, snr)
            loss = ldict["total_loss"]
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            all_logits.append(out["occupancy_logits"].detach().cpu())
            all_targets.append(y.detach().cpu())
            if "snr_estimate" in out:
                all_snr_pred.append(out["snr_estimate"].detach().cpu())
                all_snr_true.append(snr.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    m = compute_metrics(logits, targets)
    m["avg_loss"] = float(np.mean(losses)) if losses else 0.0
    if all_snr_pred:
        snr_pred = torch.cat(all_snr_pred, dim=0).numpy()
        snr_true = torch.cat(all_snr_true, dim=0).numpy()
        m["snr_mae"] = float(np.mean(np.abs(snr_true - snr_pred)))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spectrum sensing DL model")
    parser.add_argument("--config", type=str, default="conf/config_spectrum_sensing.yaml")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(
        log_level=cfg["logging"]["level"],
        log_dir=cfg["logging"]["log_dir"],
        log_file=cfg["logging"]["log_file"],
    )
    set_seed(cfg["seed"])
    device = get_device(args.device)

    train_transform = SignalTransform(
        normalize=True, add_noise=True, time_shift=True, freq_shift=True, amplitude_scale=True
    )
    eval_transform = SignalTransform(
        normalize=True, add_noise=False, time_shift=False, freq_shift=False, amplitude_scale=False
    )
    train_loader, val_loader, test_loader = create_spectrum_loaders(
        data_dir=cfg["dataset"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["hardware"]["num_workers"],
        pin_memory=cfg["hardware"]["pin_memory"],
        train_transform=train_transform,
        val_transform=eval_transform,
    )

    model = SpectrumSensingNet(
        input_channels=cfg["model"]["input_channels"],
        signal_length=cfg["model"]["signal_length"],
        conv_layers=tuple(cfg["model"]["feature_extractor"]["conv_layers"]),
        kernel_sizes=tuple(cfg["model"]["feature_extractor"]["kernel_sizes"]),
        pool_sizes=tuple(cfg["model"]["feature_extractor"]["pool_sizes"]),
        dropout_rate=cfg["model"]["feature_extractor"]["dropout_rate"],
        sensing_hidden_dims=tuple(cfg["model"]["sensing_head"]["hidden_dims"]),
        sensing_dropout=cfg["model"]["sensing_head"]["dropout_rate"],
        num_channels=cfg["model"]["num_channels"],
        enable_snr_head=cfg["model"]["snr_head"]["enabled"],
        snr_hidden_dims=tuple(cfg["model"]["snr_head"]["hidden_dims"]),
        snr_dropout=cfg["model"]["snr_head"]["dropout_rate"],
    ).to(device)
    loss_fn = SpectrumSensingLoss(
        occupancy_weight=cfg["model"]["loss_weights"]["occupancy_weight"],
        snr_weight=cfg["model"]["loss_weights"]["snr_weight"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["num_epochs"])
    writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])

    ckpt_dir = Path(cfg["training"]["checkpoint"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(cfg["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{cfg['training']['num_epochs']}")
        train_m = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
        val_m = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step()

        logger.info(
            f"Train loss={train_m['avg_loss']:.4f} exact={train_m['exact_match']:.4f} f1={train_m['macro_f1']:.4f}"
        )
        logger.info(
            f"Val   loss={val_m['avg_loss']:.4f} exact={val_m['exact_match']:.4f} f1={val_m['macro_f1']:.4f}"
        )

        writer.add_scalar("Loss/Train", train_m["avg_loss"], epoch)
        writer.add_scalar("Loss/Val", val_m["avg_loss"], epoch)
        writer.add_scalar("ExactMatch/Train", train_m["exact_match"], epoch)
        writer.add_scalar("ExactMatch/Val", val_m["exact_match"], epoch)
        writer.add_scalar("MacroF1/Train", train_m["macro_f1"], epoch)
        writer.add_scalar("MacroF1/Val", val_m["macro_f1"], epoch)

        if val_m["avg_loss"] < best_val:
            best_val = val_m["avg_loss"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val,
                },
                ckpt_dir / "best_spectrum_checkpoint.pth",
            )

    # Load best and evaluate on test.
    best_ckpt = torch.load(ckpt_dir / "best_spectrum_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_m = run_epoch(model, test_loader, loss_fn, device, optimizer=None)
    logger.info(
        f"Test loss={test_m['avg_loss']:.4f} exact={test_m['exact_match']:.4f} f1={test_m['macro_f1']:.4f}"
    )

    final = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_metrics": test_m,
    }
    save_metrics(final, Path(cfg["logging"]["log_dir"]) / "spectrum_final_metrics.json")
    writer.close()


if __name__ == "__main__":
    main()
