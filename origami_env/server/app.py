"""FastAPI entry point — OpenEnv create_app() + static viewer."""

import os
from pathlib import Path

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

# Task list endpoints (registered BEFORE static mount so they take priority)
from .tasks import TASKS


@app.get("/tasks")
def get_tasks():
    """List all available tasks with their target fold data."""
    return {
        name: {
            "name": task["name"],
            "description": task["description"],
            "difficulty": task["difficulty"],
            "paper": task["paper"],
            "target_fold": task["target_fold"],
        }
        for name, task in TASKS.items()
    }


@app.get("/tasks/{task_name}")
def get_task_detail(task_name: str):
    """Get a specific task by name."""
    if task_name not in TASKS:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    task = TASKS[task_name]
    return {
        "name": task["name"],
        "description": task["description"],
        "difficulty": task["difficulty"],
        "paper": task["paper"],
        "target_fold": task["target_fold"],
    }


# Serve Three.js viewer — static mount at / must come LAST (catch-all)
from fastapi.staticfiles import StaticFiles

viewer_dir = Path(__file__).resolve().parent.parent / "viewer"
if viewer_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(viewer_dir), html=True), name="viewer")


def main():
    """Entry point for openenv serve / uv run."""
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
