"""Pluggable downstream modules for inference orchestration."""

from src.downstream.base import DownstreamTaskModule
from src.downstream.link_adaptation import LinkAdaptationModule

__all__ = ["DownstreamTaskModule", "LinkAdaptationModule"]
