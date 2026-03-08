---
title: Origami Env Environment Server
emoji: 🦢
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Origami RL

OpenEnv RL environment for origami folding. Submit FOLD crease patterns, get physics simulation + shape similarity reward.

Rewards inspired by AlphaFold — chamfer distance shape matching with rotational alignment across 24 orientations.

---

## How It Works

| Step | Action | Description |
|------|--------|-------------|
| **1** | **Reset** | Pick a task → get target shape and flat paper starting state |
| **2** | **Step** | Submit a FOLD JSON crease pattern describing your fold |
| **3** | **Reward** | shape_similarity × 20.0 — score from 0 to 20 |

---

## FOLD Format

| Field | Type | Description |
|-------|------|-------------|
| `vertices_coords` | `[[x,y], ...]` | Vertex positions **(required)** |
| `edges_vertices` | `[[v1,v2], ...]` | Edge connectivity **(required)** |
| `edges_assignment` | `["B"\|"M"\|"V"\|"F"\|"U"]` | Edge types. Need ≥1 M or V, ≥1 B **(required)** |
| `edges_foldAngle` | `[degrees, ...]` | Fold angles. Defaults: V→180°, M→−180° *(optional)* |
| `faces_vertices` | `[[v0,v1,...], ...]` | Face polygons. Auto-computed if missing *(optional)* |

---

## Observation

| Field | Type | Description |
|-------|------|-------------|
| `shape_similarity` | `float 0.0–1.0` | Procrustes match to target shape |
| `final_positions` | `[[x,y,z], ...]` | Folded vertex positions |
| `target_positions` | `[[x,y,z], ...]` | Expected target positions |
| `max_strain` | `float` | Edge deformation metric |
| `is_stable` | `bool` | Convergence flag |
| `reward` | `float` | similarity × 20.0, or −2.0 on error |

---

## Rewards

The environment computes reward from the physics simulation result. Two reward functions are available for training:

| Function | Type | Description |
|----------|------|-------------|
| `valid_fold` | format reward | +1.0 valid FOLD JSON, −0.5 parseable but invalid structure, −2.0 not parseable |
| `shape_match` | main reward | similarity × 20.0 (0–20). −1.0 if simulation fails, −2.0 if invalid |

### How shape_similarity is computed

| Step | Description |
|------|-------------|
| **1. Simulate** | Run physics engine on submitted crease pattern → get final 3D vertex positions |
| **2. Center** | Center both predicted and target point clouds at origin |
| **3. Align** | Try 24 rotation alignments (90° rotations + mirrors) to handle equivalent orientations |
| **4. Chamfer** | Bidirectional nearest-neighbor distance, normalized by bounding box diagonal |
| **5. Score** | similarity = 1 − (chamfer / diagonal), clamped to [0, 1]. Reward = similarity × 20 |

`max_strain` measures edge length deviation after folding (0 = no deformation). `is_stable` indicates whether the simulation converged.

---

## Tasks

### Paper Folds

| Task | Difficulty | Description |
|------|-----------|-------------|
| **triangle** | 1 | Diagonal valley fold — fold the paper in half diagonally to make a triangle |
| **half_fold** | 1 | Horizontal valley fold — fold the paper in half at y=0.5 |
| **quarter_fold** | 2 | Two perpendicular valley folds — fold the paper into quarters |
| **letter_fold** | 2 | Two parallel folds (V + M) — tri-fold like a letter at y=1/3 and y=2/3 |

### Real-World Applications

| Task | Domain | Difficulty | Description |
|------|--------|-----------|-------------|
| **solar_panel** | space | 2 | Deployable solar panel array — three alternating M/V/M folds compact the panel for launch, then unfold to full area in orbit |
| **shelter** | architecture | 2 | Emergency A-frame tent — four folds create a tent shape with a ridge and two floor flaps |
| **starshade** | space | 3 | NASA starshade flasher fold — four petal-pairs wrap in a pinwheel from corner valley folds |

---

## API Reference

### WebSocket

```
WS /ws    Persistent connection
```

**Send: Reset**
```json
{"type": "reset", "data": {"task_name": "triangle"}}
```

**Send: Step**
```json
{"type": "step", "data": {"fold_data": {...}}}
```

**Receive: Observation**
```json
{"type": "observation", "data": {"reward": 20.0, "done": true, ...}}
```

### REST

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create session |
| `POST` | `/sessions/{id}/reset` | Reset with task_name |
| `POST` | `/sessions/{id}/step` | Submit fold action |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{name}` | Task detail + target fold |

---

## Quick Start

```python
from origami_env.client import OrigamiEnv
from origami_env.models import OrigamiAction

with OrigamiEnv(base_url="http://localhost:8000") as env:
    env.reset(task_name="triangle")
    result = env.step(OrigamiAction(fold_data={
        "vertices_coords": [[0,0],[1,0],[1,1],[0,1]],
        "edges_vertices": [[0,1],[1,2],[2,3],[3,0],[0,2]],
        "edges_assignment": ["B","B","B","B","V"],
        "edges_foldAngle": [0,0,0,0,180]
    }))
    print(result.observation.shape_similarity)  # 1.0
```

### Iterate all tasks

```python
import requests

# Fetch available tasks
tasks = requests.get("http://localhost:8000/tasks").json()

for name, info in tasks.items():
    print(f"{name}: difficulty {info['difficulty']}")

    # Get target crease pattern for this task
    detail = requests.get(f"http://localhost:8000/tasks/{name}").json()
    target = detail["target_fold"]

    # Create session, reset, step
    s = requests.post("http://localhost:8000/sessions").json()
    sid = s["session_id"]
    requests.post(f"http://localhost:8000/sessions/{sid}/reset", json={"task_name": name})
    obs = requests.post(f"http://localhost:8000/sessions/{sid}/step", json={"fold_data": target}).json()
    print(f"  reward: {obs['reward']}, similarity: {obs['shape_similarity']}")
```

---

## Architecture

- **Paradigm**: LLM generates FOLD JSON (not step-by-step actions). Single-shot episodes.
- **Physics**: CPU bar-and-hinge solver (numpy) matching OrigamiSimulator conventions
- **Reward**: Chamfer distance shape match (0–1 similarity × 20 for GRPO)
- **Viewer**: Three.js single HTML file with realistic paper rendering, shadow mapping, and interactive crease slider

### Key Files

```
origami_server/
├── engine/
│   ├── simulate.py        # Bar-and-hinge physics solver
│   ├── fold_parser.py     # FOLD JSON validation/parsing
│   └── shape_match.py     # Chamfer distance
├── environment.py          # OpenEnv Environment subclass
├── models.py               # OrigamiAction/Observation/State
├── tasks.py                # 7 tasks: triangle → starshade
└── app.py                  # FastAPI + create_app()
training/
├── reward.py               # valid_fold + shape_match GRPO rewards
└── train_grpo.py           # GRPO training script
viewer/
└── index.html              # Three.js viewer (grid + detail)
```
