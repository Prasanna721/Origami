"""Training Broadcast Server — spectator WebSocket for live training grid.

Manages an episode registry and broadcasts updates to connected viewers.
Training process pushes observations via publish_sync(). Viewers connect via
/ws/training and receive all episode updates in real-time.

Viewers are read-only spectators — they never send actions.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class EpisodeInfo:
    """State of a single training episode."""
    episode_id: str
    task_name: str = ""
    status: str = "running"          # "running" | "success" | "timeout" | "error"
    step: int = 0
    score: Optional[float] = None
    observation: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


class TrainingBroadcastServer:
    """Broadcast training episode updates to spectator WebSocket clients.

    Usage from training code (sync, any thread):
        broadcast.publish_sync({"type": "episode_update", ...})

    Usage from FastAPI (async):
        @app.websocket("/ws/training")
        async def ws(websocket):
            await broadcast.connect_spectator(websocket)
    """

    def __init__(self):
        self._spectators: List[WebSocket] = []
        self._episodes: Dict[str, EpisodeInfo] = {}
        self._batch_id: int = 0
        self._prompt_index: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._training_active: bool = False

    # ── Spectator management ───────────────────────────────────

    async def connect_spectator(self, websocket: WebSocket):
        """Accept a viewer WebSocket and stream updates until disconnect."""
        await websocket.accept()
        self._spectators.append(websocket)

        # Capture the event loop from the async context (uvicorn's loop)
        self._loop = asyncio.get_running_loop()
        self._start_drain_task()

        # Send current registry snapshot
        try:
            await websocket.send_text(json.dumps(self._registry_snapshot(), default=str))
        except Exception:
            self._spectators.remove(websocket)
            return

        # Keep connection alive, listen for disconnect
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in self._spectators:
                self._spectators.remove(websocket)

    def _registry_snapshot(self) -> dict:
        """Build current state snapshot for newly connected viewers."""
        return {
            "type": "registry",
            "batch_id": self._batch_id,
            "training_active": self._training_active,
            "episodes": {
                ep_id: {
                    "episode_id": ep.episode_id,
                    "task_name": ep.task_name,
                    "status": ep.status,
                    "step": ep.step,
                    "score": ep.score,
                    "observation": ep.observation,
                    "metrics": ep.metrics,
                }
                for ep_id, ep in self._episodes.items()
            },
        }

    # ── Async broadcast + queue drain ──────────────────────────

    async def _broadcast(self, message: dict):
        """Send a JSON message to all connected spectators."""
        if not self._spectators:
            return

        text = json.dumps(message, default=str)
        disconnected = []
        for ws in self._spectators:
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in self._spectators:
                self._spectators.remove(ws)

    async def _drain_queue(self):
        """Background task: drain the sync→async queue and broadcast."""
        while True:
            try:
                message = await self._queue.get()
                if message is None:
                    break
                await self._broadcast(message)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _start_drain_task(self):
        """Start the drain task in the current event loop if not already running."""
        if self._loop is None:
            return
        if not hasattr(self, '_queue'):
            self._queue = asyncio.Queue(maxsize=1000)
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = self._loop.create_task(self._drain_queue())

    # ── Sync publish API (called from training threads) ────────

    def publish_sync(self, message: dict):
        """Fire-and-forget publish from sync training code (any thread).

        Uses call_soon_threadsafe to safely enqueue from non-async threads.
        """
        if self._loop is None or self._loop.is_closed():
            return  # No event loop yet (no spectator connected) — discard

        def _enqueue():
            if not hasattr(self, '_queue'):
                self._queue = asyncio.Queue(maxsize=1000)
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # Drop if full — viewer is non-critical

        try:
            self._loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            pass  # Loop closed — discard

    # ── Training lifecycle methods (called from runner) ────────

    def start_training(self):
        """Signal that training has started."""
        self._training_active = True
        self.publish_sync({"type": "training_start"})

    def end_training(self, total_batches: int = 0, best_score: float = 0.0):
        """Signal that training has ended."""
        self._training_active = False
        self.publish_sync({
            "type": "training_done",
            "total_batches": total_batches,
            "best_score": best_score,
        })

    def start_batch(self, batch_id: int, num_episodes: int, prompt_index: int = 0):
        """Clear registry and start a new batch of episodes."""
        self._batch_id = batch_id
        self._prompt_index = prompt_index
        self._episodes.clear()
        self.publish_sync({
            "type": "batch_start",
            "batch_id": batch_id,
            "num_episodes": num_episodes,
            "prompt_index": prompt_index,
        })

    def end_batch(self, scores: List[float], best_episode_id: str = ""):
        """Signal batch completion with scores."""
        self.publish_sync({
            "type": "batch_done",
            "batch_id": self._batch_id,
            "scores": scores,
            "best_episode_id": best_episode_id,
            "avg_score": sum(scores) / max(len(scores), 1),
        })

    def episode_reset(self, episode_id: str, task_name: str, observation: dict):
        """Register a new episode after env.reset()."""
        self._episodes[episode_id] = EpisodeInfo(
            episode_id=episode_id,
            task_name=task_name,
        )
        self._episodes[episode_id].observation = observation
        self._episodes[episode_id].metrics = observation.get("metrics", {})
        self.publish_sync({
            "type": "episode_reset",
            "episode_id": episode_id,
            "task_name": task_name,
            "observation": observation,
        })

    def episode_step(self, episode_id: str, step: int, observation: dict):
        """Update episode with new step observation."""
        ep = self._episodes.get(episode_id)
        if ep:
            ep.step = step
            ep.observation = observation
            ep.metrics = observation.get("metrics", {})
        self.publish_sync({
            "type": "episode_update",
            "episode_id": episode_id,
            "step": step,
            "status": "running",
            "observation": observation,
        })

    def episode_done(
        self,
        episode_id: str,
        score: float,
        status: str = "success",
        observation: Optional[dict] = None,
    ):
        """Mark episode as complete."""
        ep = self._episodes.get(episode_id)
        if ep:
            ep.status = status
            ep.score = score
            ep.finished_at = time.time()
            if observation:
                ep.observation = observation
                ep.metrics = observation.get("metrics", {})
        self.publish_sync({
            "type": "episode_done",
            "episode_id": episode_id,
            "status": status,
            "score": score,
            "final_metrics": ep.metrics if ep else {},
        })
