"""
Base contract for pluggable downstream tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class DownstreamTaskModule(ABC):
    """
    Abstract interface for downstream decision modules.

    Implementations receive the routed inference context and return a
    JSON-serializable decision payload.
    """

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute downstream logic for one inference context."""
        raise NotImplementedError
