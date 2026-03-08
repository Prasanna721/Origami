"""FastAPI entry point — OpenEnv create_app() + static Three.js viewer + training broadcast."""
from __future__ import annotations

import os

from fastapi import Request, WebSocket
from openenv.core.env_server.http_server import create_app

from .origami_environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation
from .training_broadcast import TrainingBroadcastServer

# ── OpenEnv app ───────────────────────────────────────────────

app = create_app(
    env=lambda: OrigamiEnvironment(),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="origami_env",
    max_concurrent_envs=1,
)

# ── Training broadcast (shared instance) ─────────────────────

broadcast = TrainingBroadcastServer()


@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket):
    """Spectator WebSocket for live training grid viewer."""
    await broadcast.connect_spectator(websocket)


# ── Pattern loading + examples ─────────────────────────────────

@app.post("/load_pattern")
async def load_pattern(request: Request):
    """Load a crease pattern from SVG or FOLD content."""
    data = await request.json()
    format = data.get("format", "fold")
    content = data.get("content", "")
    env = OrigamiEnvironment()
    obs = env.load_pattern(format, content)
    return obs.model_dump()


@app.get("/examples")
async def list_examples():
    """Return list of bundled SVG/FOLD examples."""
    from .engine.svg_importer import list_bundled_examples
    return list_bundled_examples()


# ── Serve Three.js viewers as static files ────────────────────

from fastapi.staticfiles import StaticFiles  # noqa: E402

viewer_dir = os.path.join(os.path.dirname(__file__), "..", "viewer")
if os.path.isdir(viewer_dir):
    app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")


def main():
    """Entry point for `openenv serve` and `[project.scripts]`."""
    import uvicorn
    uvicorn.run("origami_env.server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
