"""
WiFi link adaptation downstream module.
Maps SNR estimates to an MCS recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.downstream.base import DownstreamTaskModule


@dataclass(frozen=True)
class MCSProfile:
    min_snr_db: float
    mcs: str
    modulation: str
    code_rate: str


class LinkAdaptationModule(DownstreamTaskModule):
    """
    Recommend a WiFi MCS from model outputs.

    This module is intentionally lightweight and deterministic so it can be
    used as a reliable downstream block in orchestration experiments.
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        snr_offset_db: float = 0.0,
        profiles: Optional[List[MCSProfile]] = None,
    ):
        self.min_confidence = min_confidence
        self.snr_offset_db = snr_offset_db
        self._profiles: List[MCSProfile] = profiles or [
            MCSProfile(30.0, "MCS9", "256-QAM", "5/6"),
            MCSProfile(27.0, "MCS8", "256-QAM", "3/4"),
            MCSProfile(23.0, "MCS7", "64-QAM", "5/6"),
            MCSProfile(20.0, "MCS6", "64-QAM", "3/4"),
            MCSProfile(17.0, "MCS5", "64-QAM", "2/3"),
            MCSProfile(14.0, "MCS4", "16-QAM", "3/4"),
            MCSProfile(11.0, "MCS3", "16-QAM", "1/2"),
            MCSProfile(8.0, "MCS2", "QPSK", "3/4"),
            MCSProfile(5.0, "MCS1", "QPSK", "1/2"),
            MCSProfile(float("-inf"), "MCS0", "BPSK", "1/2"),
        ]

    def _choose_profile(self, snr_db: float) -> MCSProfile:
        snr_db = snr_db + self.snr_offset_db
        for profile in self._profiles:
            if snr_db >= profile.min_snr_db:
                return profile
        return self._profiles[-1]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        predicted_class_name = str(context.get("predicted_class_name", "")).lower()
        confidence = float(context.get("confidence", 0.0))
        snr_db = float(context.get("snr_estimate", 0.0))

        if predicted_class_name != "wifi":
            return {
                "module": "link_adaptation",
                "status": "skipped",
                "reason": "non_wifi_prediction",
                "recommended_action": "no_link_adaptation_required",
                "recommended_mcs": None,
            }

        if confidence < self.min_confidence:
            return {
                "module": "link_adaptation",
                "status": "deferred",
                "reason": "low_confidence",
                "recommended_action": "defer_transmission_and_resense",
                "recommended_mcs": None,
                "min_confidence": self.min_confidence,
                "observed_confidence": confidence,
                "snr_offset_db": self.snr_offset_db,
            }

        profile = self._choose_profile(snr_db)
        return {
            "module": "link_adaptation",
            "status": "ok",
            "recommended_action": "transmit",
            "recommended_mcs": profile.mcs,
            "modulation": profile.modulation,
            "code_rate": profile.code_rate,
            "snr_db": snr_db,
            "confidence": confidence,
            "snr_offset_db": self.snr_offset_db,
        }
