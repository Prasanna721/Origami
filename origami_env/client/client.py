"""OrigamiEnvClient — connects to the origami environment server."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class OrigamiEnvClient:
    """
    Client for connecting to the origami environment server.

    Usage:
        client = OrigamiEnvClient("http://localhost:8000")
        obs = client.reset(task_name="half_fold")
        while not obs["done"]:
            action = my_strategy(obs["paper_state"])
            obs = client.step(action)
        print(f"Reward: {obs['reward']}")
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = requests.get(f"{self.server_url}/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        seed: Optional[int] = None,
        task_name: Optional[str] = None,
        difficulty: Optional[int] = None,
        episode_id: Optional[str] = None,
        subdivisions: int = 4,
    ) -> Dict[str, Any]:
        """Reset environment. Returns observation dict."""
        payload = {
            "seed": seed,
            "task_name": task_name,
            "difficulty": difficulty,
            "episode_id": episode_id,
            "subdivisions": subdivisions,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = requests.post(f"{self.server_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply one fold. Returns observation dict.

        action = {
            "fold_type": "valley",
            "fold_line": {"start": [0, 0.5], "end": [1, 0.5]},
            "fold_angle": 180,
            "layer_select": "all",
        }
        """
        resp = requests.post(
            f"{self.server_url}/step",
            json={"action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def stop(self) -> Dict[str, Any]:
        """Send stop action to end the episode."""
        return self.step({
            "fold_type": "stop",
            "fold_line": {"start": [0, 0], "end": [0, 0]},
            "fold_angle": 0,
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        resp = requests.get(f"{self.server_url}/state")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> list:
        """Get available tasks."""
        resp = requests.get(f"{self.server_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def get_task(self, task_name: str) -> Dict[str, Any]:
        """Get a specific task definition."""
        resp = requests.get(f"{self.server_url}/tasks/{task_name}")
        resp.raise_for_status()
        return resp.json()
