"""OrigamiEnvClient — OpenEnv EnvClient for the origami environment."""
from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from origami_env.server.models import OrigamiAction, OrigamiObservation, OrigamiState


class OrigamiEnvClient(EnvClient[OrigamiAction, OrigamiObservation, OrigamiState]):
    """
    OpenEnv client for connecting to the origami environment server.

    Usage:
        with OrigamiEnvClient("http://localhost:8000") as client:
            result = client.reset(task_name="half_fold")
            while not result.done:
                action = OrigamiAction(
                    fold_type="valley",
                    fold_line={"start": [0, 0.5], "end": [1, 0.5]},
                    fold_angle=180.0,
                )
                result = client.step(action)
            print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: OrigamiAction) -> Dict[str, Any]:
        """Convert an OrigamiAction to JSON payload for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OrigamiObservation]:
        """Parse server JSON response into StepResult[OrigamiObservation]."""
        return StepResult(
            observation=OrigamiObservation(**payload.get("observation", {})),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OrigamiState:
        """Parse server state JSON into OrigamiState."""
        return OrigamiState(**payload)
