"""FastAPI entry point — serves OpenEnv API + renders."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .origami_environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation, OrigamiState
from .tasks import get_task_pool, get_task_by_name

# ── App setup ──────────────────────────────────────────────────

RENDERS_DIR = os.environ.get("RENDERS_DIR", "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)

app = FastAPI(
    title="Origami RL Environment",
    description="OpenEnv-compatible origami folding environment",
    version="0.1.0",
)

# CORS for React frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (SUPPORTS_CONCURRENT_SESSIONS = False)
env = OrigamiEnvironment(renders_dir=RENDERS_DIR)


# ── Request/Response models ───────────────────────────────────

class ResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None
    task_name: str | None = None
    difficulty: int | None = None
    subdivisions: int = 4


class StepRequest(BaseModel):
    action: OrigamiAction


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_name": "origami_env"}


@app.post("/reset", response_model=OrigamiObservation)
def reset(req: ResetRequest):
    obs = env.reset(
        seed=req.seed,
        episode_id=req.episode_id,
        task_name=req.task_name,
        difficulty=req.difficulty,
        subdivisions=req.subdivisions,
    )
    return obs


@app.post("/step", response_model=OrigamiObservation)
def step(req: StepRequest):
    obs = env.step(req.action)
    return obs


@app.get("/state", response_model=OrigamiState)
def get_state():
    return env.state


@app.get("/tasks")
def list_tasks():
    return get_task_pool()


@app.get("/tasks/{task_name}")
def get_task(task_name: str):
    task = get_task_by_name(task_name)
    if task is None:
        return {"error": f"Task '{task_name}' not found"}
    return task


# ── Serve rendered images ─────────────────────────────────────

app.mount("/renders", StaticFiles(directory=RENDERS_DIR), name="renders")


# ── WebSocket (for real-time OpenEnv protocol) ────────────────

from fastapi import WebSocket, WebSocketDisconnect
import json as _json


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = _json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                obs = env.reset(
                    seed=msg.get("seed"),
                    episode_id=msg.get("episode_id"),
                    task_name=msg.get("task_name"),
                    difficulty=msg.get("difficulty"),
                )
                await ws.send_text(_json.dumps({
                    "type": "observation",
                    "observation": obs.model_dump(),
                }))

            elif msg_type == "step":
                action_data = msg.get("action", {})
                action = OrigamiAction(**action_data)
                obs = env.step(action)
                await ws.send_text(_json.dumps({
                    "type": "observation",
                    "observation": obs.model_dump(),
                }))

            elif msg_type == "state":
                state = env.state
                await ws.send_text(_json.dumps({
                    "type": "state",
                    "state": state.model_dump(),
                }))

            else:
                await ws.send_text(_json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                }))

    except WebSocketDisconnect:
        pass
