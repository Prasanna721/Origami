"""OrigamiEnvironment — the OpenEnv Environment class.

Wraps engine. Does NOT contain origami logic itself.
Calls engine.fold, engine.physics, engine.validation, engine.metrics.
No server-side rendering — paper_state in observation IS the render data.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from .engine.paper import PaperState, create_flat_sheet
from .engine.fold import apply_fold, FoldError
from .engine.physics import simulate
from .engine.validation import validate_state
from .engine.metrics import compute_all_metrics
from .engine.materials import MATERIALS, Material
from .models import OrigamiAction, OrigamiObservation, OrigamiState
from .tasks import sample_task, get_task_by_name


class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    """
    OpenEnv-compatible environment for origami folding.

    Each episode:
      1. reset() creates a flat sheet with a sampled task
      2. step(action) applies one fold, runs physics, validates, computes metrics
      3. "stop" action or max_folds ends the episode with final reward

    No server-side rendering. The observation contains paper_state (FOLD-compatible
    geometry data) which Three.js renders in the browser, and metrics which reward
    functions read during training.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paper: Optional[PaperState] = None
        self._task: Optional[Dict[str, Any]] = None
        self._fold_history: list = []
        self._metrics: Dict[str, Any] = {}
        self._validation: Dict[str, Any] = {}
        self._error: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._total_reward: float = 0.0

    # ── reset ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> OrigamiObservation:
        """Reset environment: sample task, create flat sheet, compute initial metrics."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._fold_history = []
        self._error = None
        self._total_reward = 0.0

        # Get task
        task_name = kwargs.get("task_name")
        if task_name:
            self._task = get_task_by_name(task_name)
        if not self._task:
            self._task = kwargs.get("task") or sample_task(
                seed=seed,
                difficulty=kwargs.get("difficulty"),
            )

        # Resolve material
        material = self._task.get("material", "paper")
        if isinstance(material, str):
            mat_obj = MATERIALS.get(material, MATERIALS["paper"])
        elif isinstance(material, dict):
            mat_obj = Material(**material)
        else:
            mat_obj = material

        # Create flat sheet
        self._paper = create_flat_sheet(
            width=self._task["width"],
            height=self._task["height"],
            material=mat_obj,
            subdivisions=kwargs.get("subdivisions", 2),
        )

        # Initial validation + metrics
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        return self._make_observation(done=False, reward=None)

    # ── step ───────────────────────────────────────────────────

    def step(
        self,
        action: OrigamiAction | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> OrigamiObservation:
        """Apply one fold, run physics, validate, compute metrics, return observation."""
        # Accept dict input for convenience
        if isinstance(action, dict):
            action = OrigamiAction(**action)

        self._step_count += 1
        self._error = None

        # Handle "stop" action
        if action.fold_type == "stop":
            return self._finalize_episode()

        # Build fold dict
        fold_dict = {
            "type": action.fold_type,
            "line": action.fold_line,
            "angle": action.fold_angle,
            "layer_select": action.layer_select,
        }

        # Apply fold
        try:
            self._paper = apply_fold(self._paper, fold_dict)
            self._fold_history.append({**fold_dict, "step": self._step_count})
        except FoldError as e:
            self._error = str(e)
            return self._make_observation(done=True, reward=-5.0)

        # Run physics
        try:
            self._paper = simulate(self._paper, fold_percent=1.0)
        except Exception as e:
            self._error = f"Physics failed: {e}"

        # Validate
        self._validation = validate_state(self._paper)

        # Compute metrics
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # Check if episode should end
        done = False

        # Auto-end on max folds
        max_folds = self._task.get("max_folds", 50) if self._task else 50
        if self._step_count >= max_folds:
            done = True

        if done:
            return self._finalize_episode()

        return self._make_observation(done=False, reward=None)

    # ── state property ─────────────────────────────────────────

    @property
    def state(self) -> OrigamiState:
        return OrigamiState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task.get("name", "") if self._task else "",
            num_folds_applied=len(self._fold_history),
            is_valid=self._metrics.get("is_valid", True),
            total_reward=self._total_reward,
        )

    # ── finalize ───────────────────────────────────────────────

    def _finalize_episode(self) -> OrigamiObservation:
        """End episode: compute final reward."""
        reward = self._compute_reward()
        self._total_reward = reward
        return self._make_observation(done=True, reward=reward)

    # ── helpers ────────────────────────────────────────────────

    def _make_observation(
        self,
        done: bool,
        reward: Optional[float],
    ) -> OrigamiObservation:
        return OrigamiObservation(
            done=done,
            reward=reward,
            task=self._task or {},
            paper_state=self._paper.to_observation_dict() if self._paper else {},
            metrics=self._metrics,
            fold_history=self._fold_history,
            error=self._error,
        )

    # ── load_pattern ────────────────────────────────────────────

    def load_pattern(self, format: str, content: str) -> OrigamiObservation:
        """Load a crease pattern from SVG or FOLD content."""
        if format == "svg":
            from .engine.svg_importer import parse_svg
            self._paper = parse_svg(content)
        elif format == "fold":
            import json
            data = json.loads(content)
            self._paper = PaperState.from_fold_json(data)

        self._fold_history = []
        self._step_count = 0
        self._error = None
        self._task = {
            "name": "custom",
            "width": self._paper.width,
            "height": self._paper.height,
        }
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)
        return self._make_observation(done=False, reward=None)

    # ── animate ───────────────────────────────────────────────

    def animate(self, fold_percent: float) -> OrigamiObservation:
        """Re-run physics with given fold_percent (0=flat, 1=fully folded)."""
        if self._paper is None:
            return self._make_observation(done=False, reward=None)
        # Reset to rest positions
        paper_copy = self._paper.copy()
        paper_copy.vertices_coords = paper_copy.rest_positions.copy()
        # Simulate with fold_percent
        paper_copy = simulate(paper_copy, fold_percent=fold_percent)
        # Update current paper
        self._paper = paper_copy
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)
        return self._make_observation(done=False, reward=None)

    # ── reward ────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Compute episode reward from metrics."""
        m = self._metrics
        reward = 0.0

        # Compactness reward (main signal)
        reward += m.get("compactness", 0) * 20.0

        # Fits target box bonus
        if m.get("fits_target_box", False):
            reward += 10.0

        # Deployability bonus
        if m.get("is_deployable", False):
            reward += 5.0

        # Validity penalties (capped to prevent overwhelming other signals)
        reward -= min(m.get("kawasaki_violations", 0), 5) * 2.0
        reward -= min(m.get("maekawa_violations", 0), 5) * 2.0
        reward -= min(m.get("self_intersections", 0), 5) * 3.0

        # Fold count penalty (encourage efficiency)
        reward -= m.get("fold_count", 0) * 0.5

        # Strain penalty
        max_strain = m.get("max_strain", 0)
        mat_limit = self._paper.material.max_strain if self._paper else 0.05
        if max_strain > mat_limit:
            reward -= 3.0 * (max_strain / mat_limit)

        return reward
