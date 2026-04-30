"""
Inference orchestration pipeline.
Routes predictions into a pluggable downstream task.
"""

from __future__ import annotations

from typing import Any, Dict

from src.downstream.base import DownstreamTaskModule


class SignalOrchestrator:
    """Two-stage orchestrator: classify/estimate, then downstream decision."""

    def __init__(self, downstream_module: DownstreamTaskModule):
        self.downstream_module = downstream_module

    def route(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        predicted_class_name = str(prediction.get("predicted_class_name", "")).lower()

        if predicted_class_name == "noise":
            return {
                "pipeline_status": "completed",
                "route": "noise_exit",
                "downstream": {
                    "module": "link_adaptation",
                    "status": "skipped",
                    "reason": "noise_prediction",
                    "recommended_action": "channel_idle_or_unoccupied",
                    "recommended_mcs": None,
                },
            }

        return {
            "pipeline_status": "completed",
            "route": "signal_to_downstream",
            "downstream": self.downstream_module.run(prediction),
        }
