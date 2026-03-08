"""Pydantic models: OrigamiAction, OrigamiObservation, OrigamiState.

Subclasses of OpenEnv base types (Action, Observation, State).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class OrigamiAction(Action):
    """One fold operation. Sent by the client each step."""

    fold_type: str = "valley"
    """'valley' | 'mountain' | 'pleat' | 'crimp' | 'stop'"""

    fold_line: Dict[str, List[float]] = Field(default_factory=lambda: {"start": [0, 0.5], "end": [1, 0.5]})
    """{'start': [x, y], 'end': [x, y]} — normalized 0-1 coordinates on the sheet."""

    fold_angle: float = 180.0
    """Degrees, 0-180. 180 = fully folded."""

    layer_select: str = "all"
    """'all' | 'top' | 'bottom' — which layers to fold."""


class OrigamiObservation(Observation):
    """Everything the viewer and the LLM need. Returned by reset() and step().

    No render_urls — paper_state contains all geometry data for Three.js
    to render directly. During training, reward functions read metrics.
    """

    # Task description
    task: Dict[str, Any] = Field(default_factory=dict)

    # Paper state (FOLD-compatible geometry + physics data)
    paper_state: Dict[str, Any] = Field(default_factory=dict)

    # All computed metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # History of folds applied
    fold_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Error message if fold failed
    error: Optional[str] = None


class OrigamiState(State):
    """Server-side episode tracking."""

    task_name: str = ""
    num_folds_applied: int = 0
    is_valid: bool = True
    total_reward: float = 0.0
