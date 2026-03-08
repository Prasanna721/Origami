"""FastAPI entry point — OpenEnv create_app() + static renders."""
from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from .origami_environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation

# ── Renders directory ─────────────────────────────────────────

RENDERS_DIR = os.environ.get("RENDERS_DIR", "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)


# ── OpenEnv app ───────────────────────────────────────────────

app = create_app(
    env=lambda: OrigamiEnvironment(renders_dir=RENDERS_DIR),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="origami_env",
    max_concurrent_envs=1,
)

# ── Serve rendered images ─────────────────────────────────────

from fastapi.staticfiles import StaticFiles  # noqa: E402

app.mount("/renders", StaticFiles(directory=RENDERS_DIR), name="renders")
