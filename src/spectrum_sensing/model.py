"""
Deep learning model for spectrum sensing (multi-label channel occupancy).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SpectrumSensingNet(nn.Module):
    """
    CNN model with shared backbone and optional SNR head.

    Outputs:
    - occupancy_logits: multi-label logits for channel occupancy (size = num_channels)
    - snr_estimate: optional scalar SNR estimate
    """

    def __init__(
        self,
        input_channels: int = 2,
        signal_length: int = 128,
        conv_layers: Tuple[int, ...] = (64, 128, 256, 512),
        kernel_sizes: Tuple[int, ...] = (9, 7, 5, 3),
        pool_sizes: Tuple[int, ...] = (2, 2, 2, 2),
        dropout_rate: float = 0.3,
        sensing_hidden_dims: Tuple[int, ...] = (512, 256),
        sensing_dropout: float = 0.4,
        num_channels: int = 4,
        enable_snr_head: bool = True,
        snr_hidden_dims: Tuple[int, ...] = (256, 128),
        snr_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.num_channels = num_channels
        self.enable_snr_head = enable_snr_head

        conv_blocks = []
        in_ch = input_channels
        for out_ch, k, p in zip(conv_layers, kernel_sizes, pool_sizes):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(p),
                    nn.Dropout(dropout_rate),
                )
            )
            in_ch = out_ch
        self.backbone = nn.ModuleList(conv_blocks)

        feature_size = self._infer_feature_size()

        sensing_layers = []
        prev = feature_size
        for h in sensing_hidden_dims:
            sensing_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(sensing_dropout)])
            prev = h
        sensing_layers.append(nn.Linear(prev, num_channels))
        self.sensing_head = nn.Sequential(*sensing_layers)

        if enable_snr_head:
            snr_layers = []
            prev = feature_size
            for h in snr_hidden_dims:
                snr_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(snr_dropout)])
                prev = h
            snr_layers.append(nn.Linear(prev, 1))
            self.snr_head = nn.Sequential(*snr_layers)
        else:
            self.snr_head = None

    def _infer_feature_size(self) -> int:
        x = torch.randn(1, self.input_channels, self.signal_length)
        with torch.no_grad():
            for block in self.backbone:
                x = block(x)
        return int(x.numel())

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")
        # Accept (B, L, 2) and (B, 2, L)
        if x.shape[2] == self.input_channels and x.shape[1] != self.input_channels:
            x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._prepare_input(x)
        for block in self.backbone:
            x = block(x)
        features = x.reshape(x.size(0), -1)
        occupancy_logits = self.sensing_head(features)
        output = {"occupancy_logits": occupancy_logits}
        if self.enable_snr_head and self.snr_head is not None:
            output["snr_estimate"] = self.snr_head(features).squeeze(-1)
        return output


class SpectrumSensingLoss(nn.Module):
    """Combined loss for occupancy (BCE) and optional SNR regression."""

    def __init__(self, occupancy_weight: float = 1.0, snr_weight: float = 0.0) -> None:
        super().__init__()
        self.occupancy_weight = occupancy_weight
        self.snr_weight = snr_weight
        self.occ_loss = nn.BCEWithLogitsLoss()
        self.snr_loss = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        occupancy_targets: torch.Tensor,
        snr_targets: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        occupancy_loss = self.occ_loss(outputs["occupancy_logits"], occupancy_targets.float())
        total = self.occupancy_weight * occupancy_loss
        losses: Dict[str, torch.Tensor] = {"occupancy_loss": occupancy_loss}
        if "snr_estimate" in outputs and snr_targets is not None and self.snr_weight > 0.0:
            s_loss = self.snr_loss(outputs["snr_estimate"], snr_targets.float())
            losses["snr_loss"] = s_loss
            total = total + self.snr_weight * s_loss
        losses["total_loss"] = total
        return losses
