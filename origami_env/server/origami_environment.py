"""OrigamiEnvironment — the OpenEnv Environment class.

Wraps engine + renderer. Does NOT contain origami logic itself.
Calls engine.fold, engine.physics, engine.validation, engine.metrics, renderer.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from .engine.paper import PaperState, create_flat_sheet
from .engine.fold import apply_fold, FoldError
from .engine.physics import simulate
from .engine.validation import validate_state
from .engine.metrics import compute_all_metrics
from .engine.materials import MATERIALS, Material
from .renderer.screenshots import capture_step, capture_episode_summary
from .renderer.recorder import record_fold_animation
from .renderer.exporter import save_fold_json
from .models import OrigamiAction, OrigamiObservation, OrigamiState
from .tasks import sample_task, get_task_by_name


class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    """
    OpenEnv-compatible environment for origami folding.

    Each episode:
      1. reset() creates a flat sheet with a sampled task
      2. step(action) applies one fold, runs physics, validates, renders
      3. "stop" action or max_folds ends the episode with final reward
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, renders_dir: str = "renders", **kwargs):
        super().__init__(**kwargs)
        self._paper: Optional[PaperState] = None
        self._task: Optional[Dict[str, Any]] = None
        self._fold_history: list = []
        self._metrics: Dict[str, Any] = {}
        self._validation: Dict[str, Any] = {}
        self._error: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._episode_dir: Optional[str] = None
        self._renders_dir = renders_dir
        self._total_reward: float = 0.0

    # ── reset ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> OrigamiObservation:
        """Reset environment: sample task, create flat sheet, render initial state."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._fold_history = []
        self._error = None
        self._total_reward = 0.0

        # Create episode render directory
        self._episode_dir = os.path.join(
            self._renders_dir, f"ep_{self._episode_id[:8]}"
        )
        os.makedirs(self._episode_dir, exist_ok=True)

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
            subdivisions=kwargs.get("subdivisions", 4),
        )

        # Initial validation + metrics
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # Render initial state
        render_urls = capture_step(self._paper, 0, self._episode_dir)

        return self._make_observation(done=False, reward=None, render_urls=render_urls)

    # ── step ───────────────────────────────────────────────────

    def step(
        self,
        action: OrigamiAction | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> OrigamiObservation:
        """Apply one fold, run physics, validate, render, return observation."""
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
            return self._make_observation(done=True, reward=-5.0, render_urls={})

        # Run physics
        try:
            self._paper = simulate(self._paper, fold_percent=1.0)
        except Exception as e:
            self._error = f"Physics failed: {e}"

        # Validate
        self._validation = validate_state(self._paper)

        # Compute metrics
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # Render this step
        render_urls = capture_step(self._paper, self._step_count, self._episode_dir)

        # Check if episode should end
        done = False

        # Auto-end on max folds
        max_folds = self._task.get("max_folds", 50) if self._task else 50
        if self._step_count >= max_folds:
            done = True

        # Self-intersections are tracked as a penalty metric, not an auto-end.
        # The simplified triangle-triangle test is too aggressive for subdivided meshes.

        if done:
            return self._finalize_episode()

        return self._make_observation(done=False, reward=None, render_urls=render_urls)

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
        """End episode: compute final reward, generate summary + animation."""
        reward = self._compute_reward()
        self._total_reward = reward

        render_urls = capture_step(self._paper, self._step_count, self._episode_dir)

        # Episode summary image
        try:
            summary_path = capture_episode_summary(
                self._paper, self._fold_history,
                self._task or {}, self._metrics, self._episode_dir,
            )
            render_urls["episode_summary"] = summary_path
        except Exception:
            pass

        # Fold animation GIF (non-critical)
        try:
            mat = self._task.get("material", "paper") if self._task else "paper"
            if isinstance(mat, str):
                mat_obj = MATERIALS.get(mat, MATERIALS["paper"])
            else:
                mat_obj = MATERIALS["paper"]

            initial_paper = create_flat_sheet(
                self._task["width"], self._task["height"], mat_obj,
            )
            gif_path = record_fold_animation(
                initial_paper, self._fold_history,
                os.path.join(self._episode_dir, "animation.gif"),
            )
            render_urls["episode_gif"] = gif_path
        except Exception:
            pass

        # FOLD JSON export
        try:
            fold_path = os.path.join(self._episode_dir, "state.fold")
            save_fold_json(self._paper, fold_path, self._fold_history)
            render_urls["fold_json"] = fold_path
        except Exception:
            pass

        return self._make_observation(done=True, reward=reward, render_urls=render_urls)

    # ── helpers ────────────────────────────────────────────────

    def _make_observation(
        self,
        done: bool,
        reward: Optional[float],
        render_urls: Dict[str, str],
    ) -> OrigamiObservation:
        return OrigamiObservation(
            done=done,
            reward=reward,
            task=self._task or {},
            paper_state=self._paper.to_observation_dict() if self._paper else {},
            metrics=self._metrics,
            fold_history=self._fold_history,
            error=self._error,
            render_urls=render_urls,
        )

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
