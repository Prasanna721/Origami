"""Training Runner — parallel episode executor with broadcast hooks.

Runs G strategy functions against in-process OrigamiEnvironment instances.
After each step, broadcasts the observation to the TrainingBroadcastServer
so the grid viewer can show live progress.

Usage:
    from origami_env.training.runner import TrainingRunner
    from origami_env.server.training_broadcast import TrainingBroadcastServer

    broadcast = TrainingBroadcastServer()
    runner = TrainingRunner(broadcast=broadcast, task_name="half_fold")

    scores = runner.run_batch(strategies, batch_id=1)
"""
from __future__ import annotations

import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from origami_env.server.origami_environment import OrigamiEnvironment
from origami_env.server.models import OrigamiAction


class TrainingRunner:
    """Execute G strategy functions in parallel with live broadcast."""

    def __init__(
        self,
        broadcast=None,
        task_name: str = "half_fold",
        max_steps: int = 20,
        max_workers: int = 8,
        timeout_s: float = 10.0,
    ):
        self.broadcast = broadcast
        self.task_name = task_name
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.timeout_s = timeout_s

    def run_episode(
        self,
        strategy_fn: Callable,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single episode with a strategy function.

        Returns dict with keys: score, metrics, status, steps, episode_id
        """
        episode_id = episode_id or str(uuid.uuid4())[:8]
        task = task_name or self.task_name

        env = OrigamiEnvironment()
        obs = env.reset(task_name=task)

        # Broadcast reset
        obs_dict = _obs_to_dict(obs)
        if self.broadcast:
            self.broadcast.episode_reset(episode_id, task, obs_dict)

        steps = 0
        try:
            while not obs.done and steps < self.max_steps:
                paper_state = obs.paper_state
                fold = strategy_fn(paper_state)

                if fold is None:
                    action = OrigamiAction(
                        fold_type="stop",
                        fold_line={"start": [0, 0], "end": [0, 0]},
                        fold_angle=0,
                    )
                else:
                    action = OrigamiAction(
                        fold_type=fold.get("type", "valley"),
                        fold_line=fold.get("line", {"start": [0, 0.5], "end": [1, 0.5]}),
                        fold_angle=fold.get("angle", 180),
                        layer_select=fold.get("layer_select", "all"),
                    )

                obs = env.step(action)
                steps += 1

                # Broadcast step
                obs_dict = _obs_to_dict(obs)
                if self.broadcast:
                    self.broadcast.episode_step(episode_id, steps, obs_dict)

            # Episode complete
            score = obs.reward if obs.reward is not None else 0.0
            metrics = obs.metrics if hasattr(obs, "metrics") else {}
            status = "success"

        except TimeoutError:
            score = -1.0
            metrics = {}
            status = "timeout"
            obs_dict = {}
        except Exception as e:
            score = -3.0
            metrics = {}
            status = "error"
            obs_dict = {}
            traceback.print_exc()

        # Broadcast done
        if self.broadcast:
            self.broadcast.episode_done(episode_id, score, status, obs_dict)

        return {
            "episode_id": episode_id,
            "score": score,
            "metrics": metrics,
            "status": status,
            "steps": steps,
        }

    def run_batch(
        self,
        strategies: List[Callable],
        batch_id: int = 0,
        prompt_index: int = 0,
    ) -> List[float]:
        """Run G strategies in parallel, return scores.

        Each strategy gets its own OrigamiEnvironment instance (in-process).
        Broadcast updates are sent for the grid viewer.
        """
        g = len(strategies)

        # Broadcast batch start
        if self.broadcast:
            self.broadcast.start_batch(batch_id, g, prompt_index)

        # Generate episode IDs
        episode_ids = [f"ep_{batch_id}_{i}" for i in range(g)]

        # Run episodes in parallel
        results = [None] * g
        with ThreadPoolExecutor(max_workers=min(self.max_workers, g)) as pool:
            futures = {
                pool.submit(
                    self.run_episode,
                    strategy_fn=strategies[i],
                    episode_id=episode_ids[i],
                ): i
                for i in range(g)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=self.timeout_s)
                except Exception:
                    results[idx] = {
                        "episode_id": episode_ids[idx],
                        "score": -3.0,
                        "metrics": {},
                        "status": "error",
                        "steps": 0,
                    }

        scores = [r["score"] for r in results]

        # Find best
        best_idx = max(range(g), key=lambda i: scores[i])
        best_ep_id = episode_ids[best_idx]

        # Broadcast batch done
        if self.broadcast:
            self.broadcast.end_batch(scores, best_ep_id)

        return scores


def _obs_to_dict(obs) -> dict:
    """Convert OrigamiObservation to a JSON-serializable dict."""
    try:
        d = obs.model_dump()
    except Exception:
        d = {}

    # Ensure numpy arrays are converted to lists
    ps = d.get("paper_state", {})
    for key, val in ps.items():
        if hasattr(val, "tolist"):
            ps[key] = val.tolist()

    return d
