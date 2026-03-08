"""Pydantic models: OrigamiAction, OrigamiObservation, OrigamiState."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrigamiAction(BaseModel):
    """One fold operation. Sent by the client each step."""

    fold_type: str = "valley"
    """'valley' | 'mountain' | 'pleat' | 'crimp' | 'stop'"""

    fold_line: Dict[str, List[float]] = Field(default_factory=lambda: {"start": [0, 0.5], "end": [1, 0.5]})
    """{'start': [x, y], 'end': [x, y]} — normalized 0-1 coordinates on the sheet."""

    fold_angle: float = 180.0
    """Degrees, 0-180. 180 = fully folded."""

    layer_select: str = "all"
    """'all' | 'top' | 'bottom' — which layers to fold."""

    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrigamiObservation(BaseModel):
    """Everything the frontend and the LLM need. Returned by reset() and step()."""

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Task description
    task: Dict[str, Any] = Field(default_factory=dict)

    # Paper state (FOLD-compatible)
    paper_state: Dict[str, Any] = Field(default_factory=dict)

    # All computed metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # History of folds applied
    fold_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Error message if fold failed
    error: Optional[str] = None

    # URLs to rendered images
    render_urls: Dict[str, str] = Field(default_factory=dict)


class OrigamiState(BaseModel):
    """Server-side episode tracking."""

    episode_id: Optional[str] = None
    step_count: int = 0
    task_name: str = ""
    num_folds_applied: int = 0
    is_valid: bool = True
    total_reward: float = 0.0
    current_fold_percent: float = 1.0
