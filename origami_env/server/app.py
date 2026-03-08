"""FastAPI entry point — OpenEnv create_app() + static viewer."""

import os

from openenv.core import create_app

from .environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation

app = create_app(
    env=OrigamiEnvironment,
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="origami_env",
    max_concurrent_envs=4,
)

# Serve Three.js viewer as static files
from fastapi.staticfiles import StaticFiles

viewer_dir = os.path.join(os.path.dirname(__file__), "..", "viewer")
if os.path.isdir(viewer_dir):
    app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")


def main():
    """Entry point for openenv serve / uv run."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
