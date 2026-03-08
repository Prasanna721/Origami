# OpenEnv Environment Architecture — Origami RL

> Updated plan reflecting actual OpenEnv patterns (from 2048 reference),
> proper rendering strategy (Three.js viewer, not matplotlib),
> and clear separation between training vs demo contexts.

---

## 1. Overview

Two distinct contexts use the SAME server code:

```
CONTEXT 1: RL TRAINING (Colab/GPU machine)
┌──────────────────────────────────────────────────────────┐
│  Colab Notebook                                          │
│                                                          │
│  ┌──────────────────┐    ┌────────────────────────────┐  │
│  │  GRPOTrainer     │    │  OpenEnv Server (subprocess)│  │
│  │                  │    │  port 9000                  │  │
│  │  LLM generates   │    │                            │  │
│  │  fold_strategy()  │───▶│  reset() → step() → state │  │
│  │                  │◀───│  returns: paper_state +    │  │
│  │  Reward functions │    │    metrics + reward        │  │
│  │  score the result │    │                            │  │
│  └──────────────────┘    │  NO RENDERING              │  │
│                          │  Just geometry + numbers     │  │
│                          └────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

CONTEXT 2: DEMO / HACKATHON (HF Space + Browser)
┌──────────────────────────────────────────────────────────┐
│  HF Space (Docker)                                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  FastAPI via create_app()                           │  │
│  │                                                    │  │
│  │  /ws                  → OpenEnv WebSocket protocol  │  │
│  │  /reset, /step, /state → OpenEnv HTTP (stateless)   │  │
│  │  /health, /schema      → OpenEnv metadata           │  │
│  │  /web                  → OpenEnv built-in generic UI │  │
│  │  /viewer               → Three.js origami viewer    │  │
│  └──────────────────┬─────────────────────────────────┘  │
│                     │                                     │
│  ┌──────────────────▼─────────────────────────────────┐  │
│  │  OrigamiEnvironment                                 │  │
│  │  reset() / step() / state                           │  │
│  │                                                     │  │
│  │  ┌──────────┐  ┌──────────────────┐                │  │
│  │  │  Engine   │  │  Task System     │                │  │
│  │  │          │  │                  │                │  │
│  │  │ paper    │  │ task pool       │                │  │
│  │  │ fold     │  │ materials       │                │  │
│  │  │ physics  │  │ curriculum      │                │  │
│  │  │ validate │  │                  │                │  │
│  │  │ metrics  │  │                  │                │  │
│  │  └──────────┘  └──────────────────┘                │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Three.js Viewer (static HTML/JS, served by same   │  │
│  │  FastAPI at /viewer)                                │  │
│  │                                                    │  │
│  │  Connects to /ws → receives paper_state            │  │
│  │  Renders: 2D crease (Canvas) + 3D mesh (WebGL)    │  │
│  │  + strain heatmap (vertex colors) + animation      │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

Browser opens /viewer → sees origami fold live in 3D
```

### Key Design Decisions

1. **No server-side image rendering.** No matplotlib, no Pillow, no imageio.
   The server returns FOLD JSON data (vertices, edges, faces, strain) in every
   observation. Rendering happens in the browser via Three.js (demo) or not at
   all (training).

2. **`render_urls` removed from Observation.** The old design saved PNGs to disk
   and returned file paths — nobody fetched them during training (wasted CPU/disk).
   Instead, `paper_state` IS the render data. The Three.js viewer reads it directly.

3. **Same server code for both contexts.** The only difference is who consumes it:
   - Training: Python reward functions read `paper_state.metrics` → compute reward
   - Demo: Three.js reads `paper_state.vertices_coords` → renders 3D mesh

4. **`openenv push`** deploys to HF Spaces. Sets `ENABLE_WEB_INTERFACE=true`
   which gives us the built-in generic UI at `/web`. Our custom Three.js viewer
   is served as static files at `/viewer`.

---

## 2. Repository Structure

```
origami_env/                             # THE deliverable — one package
│
├── server/                              # Python backend (OpenEnv server)
│   │
│   ├── engine/                          # Origami simulation core
│   │   ├── __init__.py
│   │   ├── paper.py                     # PaperState dataclass, FOLD I/O, create_flat_sheet()
│   │   ├── fold.py                      # apply_fold() — quaternion rotation, face splitting
│   │   ├── physics.py                   # Bar-and-hinge Verlet solver, strain computation
│   │   ├── validation.py               # Kawasaki, Maekawa, self-intersection detection
│   │   ├── metrics.py                   # ALL metrics — compactness, strain, shape, deployability
│   │   └── materials.py                 # Material presets + stiffness parameter derivation
│   │
│   ├── renderer/                        # Minimal — FOLD JSON export only (no images)
│   │   ├── __init__.py
│   │   └── exporter.py                 # FOLD JSON export, OBJ export
│   │
│   ├── models.py                        # OpenEnv types: OrigamiAction, OrigamiObservation, OrigamiState
│   ├── origami_environment.py           # Environment class (subclasses openenv Environment)
│   ├── tasks.py                         # Task pool, curriculum levels, difficulty sampling
│   ├── app.py                           # create_app() + mount static viewer
│   ├── __init__.py
│   ├── requirements.txt                 # openenv-core, numpy, scipy, pydantic (NO matplotlib)
│   └── Dockerfile
│
├── viewer/                              # Three.js origami viewer (static files)
│   ├── index.html                       # Single page: 2D + 3D + metrics + controls
│   ├── origami-viewer.js                # Three.js rendering from FOLD data
│   └── style.css                        # Layout styles
│
├── client/                              # OpenEnv client (for RL training)
│   ├── __init__.py
│   ├── client.py                        # OrigamiEnvClient (EnvClient subclass, WebSocket)
│   └── reward_functions.py              # code_valid, no_cheating, fold_quality (GRPO rewards)
│
├── training/                            # GRPO training script (runs on Colab)
│   └── train_grpo.py                    # Launches server, runs GRPOTrainer (2048 pattern)
│
├── openenv.yaml                         # Manifest for openenv push
├── pyproject.toml
└── README.md
```

### What Changed from Previous Plan

| Before | After | Why |
|--------|-------|-----|
| `renderer/render_2d.py` (matplotlib) | REMOVED | Browser renders via Three.js |
| `renderer/render_3d.py` (matplotlib) | REMOVED | Browser renders via Three.js |
| `renderer/screenshots.py` (PNG capture) | REMOVED | No server-side images |
| `renderer/recorder.py` (GIF via imageio) | REMOVED | Browser can record (MediaRecorder) |
| `web/` (React + R3F) | `viewer/` (plain HTML + Three.js) | Simpler, no build step, same quality |
| `render_urls` in Observation | REMOVED | paper_state IS the render data |
| matplotlib, Pillow, imageio deps | REMOVED | Lighter Docker image |

---

## 3. Pydantic Models (`server/models.py`)

### OrigamiAction

```python
class OrigamiAction(Action):
    """One fold operation. Sent by the client each step."""
    # metadata: Dict[str, Any]  (inherited from Action)

    fold_type: str = "valley"             # "valley" | "mountain" | "pleat" | "crimp" | "stop"
    fold_line: Dict[str, List[float]]     # {"start": [x,y], "end": [x,y]} (normalized 0-1)
    fold_angle: float = 180.0             # degrees, 0-180
    layer_select: str = "all"             # "all" | "top" | "bottom"
```

### OrigamiObservation

```python
class OrigamiObservation(Observation):
    """Everything the LLM and Three.js viewer need."""
    # done: bool           (inherited from Observation)
    # reward: float|None   (inherited from Observation)
    # metadata: Dict       (inherited from Observation)

    task: Dict[str, Any]                  # Task definition
    paper_state: Dict[str, Any]           # FOLD-compatible geometry + physics
    # {
    #   "vertices_coords": [[x,y,z], ...],
    #   "edges_vertices": [[v1,v2], ...],
    #   "faces_vertices": [[v0,v1,v2,...], ...],
    #   "edges_assignment": ["M","V","B","F",...],
    #   "edges_foldAngle": [180, -180, 0, ...],
    #   "num_vertices": 36, "num_edges": 85, "num_faces": 50,
    #   "bounding_box": [0.5, 1.0, 0.02],
    #   "num_layers": 2,
    #   "width": 1.0, "height": 1.0,
    #   "material": {"name": "paper", ...},
    #   "strain_per_vertex": [0.001, 0.005, ...],
    #   "energy": {"total": 0.12, "bar": 0.05, "facet": 0.03, "fold": 0.04},
    #   "fold_count": 2,
    # }
    metrics: Dict[str, Any]               # All computed metrics
    fold_history: List[Dict[str, Any]]    # History of folds applied
    error: Optional[str] = None           # Error message if fold failed
```

**Note:** No `render_urls`. The `paper_state` dict contains all geometry data
needed for Three.js to render. During training, reward functions read `metrics`.
During demo, the viewer reads `paper_state.vertices_coords` etc.

### OrigamiState

```python
class OrigamiState(State):
    # episode_id: Optional[str]  (inherited from State)
    # step_count: int            (inherited from State)

    task_name: str = ""
    num_folds_applied: int = 0
    is_valid: bool = True
    total_reward: float = 0.0
```

### Wire Format (what goes over WebSocket)

OpenEnv's `serialize_observation()` extracts `done`, `reward`, `metadata` to top level:

```json
{
  "observation": {
    "task": {"name": "half_fold", "width": 1.0, ...},
    "paper_state": {
      "vertices_coords": [[0,0,0], [1,0,0], ...],
      "edges_vertices": [[0,1], [1,2], ...],
      "edges_assignment": ["B", "B", "V", ...],
      "strain_per_vertex": [0.001, 0.005, ...],
      ...
    },
    "metrics": {"compactness": 0.45, "is_valid": true, ...},
    "fold_history": [{"type": "valley", "step": 1, ...}],
    "error": null
  },
  "reward": null,
  "done": false
}
```

The Three.js viewer reads `observation.paper_state` → builds mesh.
The reward functions read `observation.metrics` → compute score.
Same data, different consumers.

---

## 4. Engine (`server/engine/`)

*Unchanged from previous plan. All engine code is already implemented and working.*

### 4.1 Paper State (`paper.py`)

FOLD-format compatible dataclass. Key fields: `vertices_coords` (N,3),
`edges_vertices` (E,2), `faces_vertices` (ragged), `edges_assignment`,
`edges_foldAngle`, `rest_lengths`, `rest_positions`, `strain_per_vertex`,
`energy`, `material`, `face_orders`, `num_layers`.

Key methods: `create_flat_sheet()`, `to_fold_json()`, `from_fold_json()`,
`to_observation_dict()`, `bounding_box`, `triangulated_faces`.

### 4.2 Fold Operations (`fold.py`)

10-step pipeline: validate → split faces → classify vertices → quaternion
rotation → update assignments → update angles → update topology → compute
rest lengths → update layers → increment fold_count.

Pleat = valley + mountain. Crimp = mountain + valley.

### 4.3 Physics Solver (`physics.py`)

Bar-and-hinge Verlet integration. Three energy components:
- E_bar (axial springs, prevents stretching)
- E_facet (panel bending, keeps faces flat)
- E_fold (crease rotation, drives folding)

Numerical stability: force clamping, NaN detection, stiffness caps,
reduced dt=0.005, damping=0.15.

### 4.4 Validation (`validation.py`)

- Kawasaki-Justin: alternating angle sum at interior vertices
- Maekawa-Justin: |M - V| = 2 at interior vertices
- Self-intersection: Z-separation + normal alignment check (not simple overlap)
- Strain limits: per-vertex strain vs material.max_strain

### 4.5 Metrics (`metrics.py`)

20+ metrics: compactness, deployment_ratio, volume_compaction, packing_efficiency,
fits_target_box, max/mean strain, energy breakdown, fold_count, folding_efficiency,
crease_complexity, is_deployable, deployment_force_estimate, chamfer/hausdorff distance.

### 4.6 Materials (`materials.py`)

Four presets: paper, mylar, aluminum, nitinol. Each has thickness, Young's modulus,
max strain, Poisson ratio, density. Derived stiffness properties for physics.

---

## 5. Renderer (`server/renderer/`)

### What's LEFT (kept)

```python
# exporter.py — FOLD JSON + OBJ export (lightweight, no image deps)

def save_fold_json(paper: PaperState, path: str, fold_history: list):
    """Export FOLD-format JSON with metadata."""

def export_obj(paper: PaperState) -> str:
    """Wavefront OBJ for external renderers / 3D printing."""
```

### What's REMOVED

| File | Was | Why Removed |
|------|-----|-------------|
| `render_2d.py` | matplotlib crease pattern PNG | Three.js viewer renders 2D in browser |
| `render_3d.py` | matplotlib 3D wireframe PNG | Three.js viewer renders 3D in browser |
| `screenshots.py` | Per-step PNG capture to disk | No server-side images needed |
| `recorder.py` | GIF assembly via imageio | Browser can use MediaRecorder API |

### Dependencies REMOVED from requirements.txt

```
matplotlib>=3.7    ← REMOVED
imageio>=2.31      ← REMOVED
Pillow>=10.0       ← REMOVED
```

---

## 6. Three.js Viewer (`viewer/`)

Static HTML + JS served by FastAPI at `/viewer`. No build step. No React.
Just one HTML file with embedded Three.js from CDN.

### Architecture

```
Browser opens /viewer
    │
    ├── Loads index.html (contains Three.js via CDN)
    │
    ├── Connects to /ws (OpenEnv WebSocket)
    │
    ├── Sends: {"type": "reset", "task_name": "solar_panel"}
    │
    ├── Receives: observation with paper_state
    │
    └── Three.js renders:
        ├── LEFT PANEL: 2D Crease Pattern (Canvas2D)
        │   - edges colored by assignment (M=red, V=blue, B=black, F=gray)
        │   - vertices as dots
        │   - uses rest_positions[:, :2]
        │
        ├── RIGHT PANEL: 3D Folded State (WebGL)
        │   - BufferGeometry from vertices_coords + triangulated faces
        │   - Vertex colors from strain_per_vertex (blue→red gradient)
        │   - OrbitControls (rotate, zoom, pan)
        │   - DoubleSide material
        │   - Edge overlay (M=red, V=blue, B=black lines)
        │
        ├── BOTTOM: Metrics Dashboard
        │   - Compactness, strain, fold count, validity, energy
        │
        └── CONTROLS
            - Task selector dropdown
            - Fold input (type, line start/end, angle)
            - Reset / Step / Stop buttons
            - Animation slider (fold_percent 0→1)
```

### Data Flow (Same FOLD Data, Browser Renders)

```
Server: env.step(action)
    → paper_state = {
        vertices_coords: [[x,y,z], ...],     ← Three.js positions
        faces_vertices: [[0,1,2], ...],       ← Three.js index buffer
        edges_assignment: ["M","V","B",...],   ← Edge colors
        strain_per_vertex: [0.01, ...],       ← Vertex colors
        edges_foldAngle: [180, -180, ...],    ← Fold visualization
      }
    → sent via WebSocket as JSON

Browser: viewer receives paper_state
    → positions = new Float32Array(vertices_coords.flat())
    → indices = new Uint16Array(triangulated_faces.flat())
    → colors = strainToColor(strain_per_vertex)  // blue→red
    → geometry.setAttribute('position', ...)
    → geometry.setIndex(...)
    → geometry.setAttribute('color', ...)
    → renderer.render(scene, camera)
```

### Key Three.js Pattern (from origami simulator reference)

```javascript
// Build mesh from FOLD data
function updateMesh(paperState) {
    const vertices = paperState.vertices_coords;
    const faces = paperState.faces_vertices;
    const strain = paperState.strain_per_vertex;

    // Triangulate faces (server already provides triangulated)
    const positions = new Float32Array(vertices.flat());
    const indices = [];
    for (const face of faces) {
        // Fan triangulation for polygons > 3 vertices
        for (let i = 1; i < face.length - 1; i++) {
            indices.push(face[0], face[i], face[i + 1]);
        }
    }

    // Strain → vertex colors (blue=0, red=max)
    const colors = new Float32Array(vertices.length * 3);
    const maxStrain = Math.max(...strain, 0.001);
    for (let i = 0; i < vertices.length; i++) {
        const t = Math.min(strain[i] / maxStrain, 1.0);
        colors[i * 3]     = t;           // R
        colors[i * 3 + 1] = 0.2;         // G
        colors[i * 3 + 2] = 1.0 - t;     // B
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
}

// Draw crease edges
function drawCreaseEdges(paperState) {
    const edgeColors = { M: 0xe74c3c, V: 0x3498db, B: 0x2c3e50 };
    for (let i = 0; i < paperState.edges_vertices.length; i++) {
        const [v1, v2] = paperState.edges_vertices[i];
        const assignment = paperState.edges_assignment[i];
        if (assignment in edgeColors) {
            // Draw line from vertices[v1] to vertices[v2]
        }
    }
}
```

---

## 7. Environment (`server/origami_environment.py`)

```python
class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, renders_dir: str = "renders", **kwargs):
        super().__init__(**kwargs)
        self._paper = None
        self._task = None
        self._fold_history = []
        self._metrics = {}
        self._error = None
        self._episode_id = None
        self._step_count = 0
        self._total_reward = 0.0

    def reset(self, seed=None, episode_id=None, **kwargs) -> OrigamiObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._fold_history = []
        self._error = None
        self._total_reward = 0.0

        # Sample task
        task_name = kwargs.get("task_name")
        self._task = get_task_by_name(task_name) or sample_task(seed=seed)

        # Create flat sheet
        self._paper = create_flat_sheet(
            self._task["width"], self._task["height"],
            MATERIALS[self._task["material"]]
        )

        # Initial validation + metrics
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        return self._make_observation(done=False, reward=None)

    def step(self, action: OrigamiAction, timeout_s=None, **kwargs) -> OrigamiObservation:
        self._step_count += 1
        self._error = None

        if action.fold_type == "stop":
            return self._finalize_episode()

        # Apply fold → physics → validate → metrics
        try:
            self._paper = apply_fold(self._paper, fold_dict)
            self._fold_history.append({**fold_dict, "step": self._step_count})
        except FoldError as e:
            self._error = str(e)
            return self._make_observation(done=True, reward=-5.0)

        self._paper = simulate(self._paper, fold_percent=1.0)
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        done = self._step_count >= self._task.get("max_folds", 50)
        if done:
            return self._finalize_episode()

        return self._make_observation(done=False, reward=None)

    @property
    def state(self) -> OrigamiState:
        return OrigamiState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task.get("name", "") if self._task else "",
            num_folds_applied=len(self._fold_history),
            is_valid=self._metrics.get("is_valid", True),
            total_reward=self._total_reward,
        )

    def _make_observation(self, done, reward) -> OrigamiObservation:
        return OrigamiObservation(
            done=done,
            reward=reward,
            task=self._task or {},
            paper_state=self._paper.to_observation_dict() if self._paper else {},
            metrics=self._metrics,
            fold_history=self._fold_history,
            error=self._error,
        )
```

**Key change:** No `render_urls`, no `capture_step()`, no `capture_episode_summary()`.
The observation contains `paper_state` (all geometry) and `metrics` (all numbers).
That's all anyone needs.

---

## 8. App + Docker (`server/app.py` + `server/Dockerfile`)

### `app.py`

```python
"""FastAPI entry point — OpenEnv create_app() + static viewer."""
import os
from openenv.core.env_server.http_server import create_app
from .origami_environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation

app = create_app(
    env=lambda: OrigamiEnvironment(),
    action_cls=OrigamiAction,
    observation_cls=OrigamiObservation,
    env_name="origami_env",
    max_concurrent_envs=1,
)

# Serve Three.js viewer as static files
from fastapi.staticfiles import StaticFiles
viewer_dir = os.path.join(os.path.dirname(__file__), "..", "viewer")
if os.path.isdir(viewer_dir):
    app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Python dependencies (lightweight — no matplotlib/Pillow/imageio)
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy server code
COPY server/ ./server/

# Copy Three.js viewer (static HTML/JS)
COPY viewer/ ./viewer/

WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

When `openenv push` deploys this, it:
1. Moves Dockerfile to repo root
2. Injects `ENV ENABLE_WEB_INTERFACE=true` → enables `/web` (generic OpenEnv UI)
3. Our `/viewer` is the custom Three.js origami viewer

### `requirements.txt`

```
openenv-core>=0.2.1
numpy>=1.24
scipy>=1.10
pydantic>=2.0
fastapi>=0.100
uvicorn>=0.22
websockets>=11.0
```

No matplotlib. No Pillow. No imageio. Docker image drops from ~500MB to ~200MB.

### `openenv.yaml`

```yaml
spec_version: 1
name: origami_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## 9. RL Training (`training/train_grpo.py`)

Follows the exact 2048 Unsloth pattern. Runs on Colab with GPU.

### Flow

```
1. Launch OpenEnv server as local subprocess (port 9000)
2. Load LLM model (e.g., gpt-oss-20b with LoRA)
3. Define prompt: "Write a fold_strategy(paper_state) function..."
4. Define 3 reward functions:
   - code_valid: Does the code parse? (+1 / -2)
   - no_cheating: Only stdlib imports? (+1 / -20)
   - fold_quality: Does strategy produce good folds? (scored from metrics)
5. GRPOTrainer trains with these rewards
6. Each reward eval: reset env → run strategy → check metrics
```

### The Prompt

```python
prompt = """
Write a Python function that folds origami to maximize compactness.
You are given a paper_state dict with vertices, edges, and faces.
Return a fold dict or None to stop:

```python
def fold_strategy(paper_state):
    # paper_state has: vertices_coords, edges_vertices, edges_assignment,
    #   bounding_box, num_layers, material, strain_per_vertex, fold_count
    return {
        "type": "valley",  # or "mountain"
        "line": {"start": [x1, y1], "end": [x2, y2]},
        "angle": 180,
    }
    # Return None when done folding
```
Only output the short function `fold_strategy`.
""".strip()
```

### Reward Functions

```python
def fold_quality(completions, **kwargs):
    """
    Execute strategy against live environment, score from metrics.
    +20.0  if compactness > 0.8 AND valid (optimal folding!)
    +5.0   if compactness > 0.5
    +2.0   if function runs but poor result
    -1.0   timeout
    -3.0   exception
     0     broken function
    """
    scores = []
    for completion in completions:
        function = extract_function(completion)
        if function is None:
            scores.append(0)
            continue
        try:
            strategy = create_locked_down_function(function)
            # Reset OpenEnv
            port, process = launch_openenv(port, process)
            result = process.reset()
            obs = result.observation

            # Run strategy loop (same as 2048)
            while not obs.done:
                fold = execute_with_time_limit(5)(strategy)(obs.paper_state)
                if fold is None:
                    action = OrigamiAction(fold_type="stop")
                else:
                    action = OrigamiAction(**fold)
                result = process.step(action)
                obs = result.observation

            # Score from final metrics
            m = obs.metrics
            if m.get("compactness", 0) > 0.8 and m.get("is_valid", False):
                scores.append(20.0)
            elif m.get("compactness", 0) > 0.5:
                scores.append(5.0)
            else:
                scores.append(2.0)
        except TimeoutError:
            scores.append(-1.0)
        except Exception:
            scores.append(-3.0)
    return scores
```

### Key Difference from 2048

2048: LLM generates `strategy(board) → action_id` (one action per call, game loops externally)
Origami: LLM generates `fold_strategy(paper_state) → fold_dict | None` (one fold per call, loops externally)

Same pattern. The reward function resets the env, loops strategy, scores the outcome.

---

## 10. Client (`client/`)

### `client.py`

```python
class OrigamiEnvClient(EnvClient[OrigamiAction, OrigamiObservation, OrigamiState]):

    def _step_payload(self, action: OrigamiAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[OrigamiObservation]:
        return StepResult(
            observation=OrigamiObservation(**payload.get("observation", {})),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> OrigamiState:
        return OrigamiState(**payload)
```

### `reward_functions.py`

Three reward functions for GRPO: `code_valid`, `no_cheating`, `fold_quality`.
Run on Colab client side, NOT on the server.

---

## 11. Task System (`server/tasks.py`)

### Curriculum (4 difficulty levels)

| Level | Task | Material | Target Ratio | Max Folds | Key Challenge |
|-------|------|----------|-------------|-----------|---------------|
| 1 | half_fold | paper | 0.50 | 3 | Learn the format |
| 1 | quarter_fold | paper | 0.25 | 5 | Two perpendicular folds |
| 2 | letter_fold | paper | 0.33 | 5 | Tri-fold, parallel lines |
| 2 | map_fold | paper | 0.125 | 8 | Grid fold, must deploy |
| 3 | solar_panel | mylar | 0.05 | 20 | Miura-ori discovery, deployability |
| 3 | shelter_wall | aluminum | 0.10 | 15 | Rigid material, strain limits |
| 4 | stent | nitinol | 0.09 | 25 | Cylindrical target shape, superelastic |

---

## 12. API Reference

### Endpoints (auto-generated by create_app())

| Endpoint | Method | Source | Purpose |
|----------|--------|--------|---------|
| `/health` | GET | OpenEnv | `{"status": "healthy"}` |
| `/ws` | WebSocket | OpenEnv | Persistent session (reset/step/state) |
| `/reset` | POST | OpenEnv | Stateless reset (creates new env per call) |
| `/step` | POST | OpenEnv | Stateless step (creates new env per call) |
| `/state` | GET | OpenEnv | Get current state |
| `/schema` | GET | OpenEnv | Action + Observation JSON schemas |
| `/metadata` | GET | OpenEnv | Environment name, description, version |
| `/web` | GET | OpenEnv | Built-in generic web UI (when ENABLE_WEB_INTERFACE=true) |
| `/viewer` | GET | Custom | Three.js origami viewer (static files) |

**Important:** HTTP `/reset` and `/step` are stateless — each creates a fresh env.
For multi-step episodes, use WebSocket `/ws`. This is OpenEnv's design.

### WebSocket Message Format

```json
// Client → Server (reset)
{"type": "reset", "task_name": "solar_panel"}

// Client → Server (step)
{
  "type": "step",
  "action": {
    "fold_type": "valley",
    "fold_line": {"start": [0, 0.5], "end": [1, 0.5]},
    "fold_angle": 180,
    "layer_select": "all"
  }
}

// Server → Client (observation)
{
  "type": "observation",
  "data": {
    "observation": {
      "task": {"name": "solar_panel", ...},
      "paper_state": {
        "vertices_coords": [[0,0,0], [1,0,0], ...],
        "edges_vertices": [[0,1], ...],
        "edges_assignment": ["B", "V", ...],
        "strain_per_vertex": [0.001, ...],
        ...
      },
      "metrics": {"compactness": 0.45, ...},
      "fold_history": [...],
      "error": null
    },
    "reward": null,
    "done": false
  }
}
```

---

## 13. Deployment

### Push to HF Spaces

```bash
cd origami_env/
huggingface-cli login
openenv push --repo-id <username>/origami-env
```

`openenv push` does:
1. Validates `openenv.yaml`
2. Moves `server/Dockerfile` → root `Dockerfile`
3. Injects `ENV ENABLE_WEB_INTERFACE=true`
4. Adds HF Space frontmatter to README
5. Uploads to HF Spaces (Docker SDK)

### Or manually via Docker

```bash
docker build -t origami-env -f server/Dockerfile .
docker run -p 8000:8000 origami-env
curl http://localhost:8000/health
# Open http://localhost:8000/viewer in browser → Three.js origami viewer
```

### HF Space README header

```yaml
---
title: Origami RL Environment
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
---
```

---

## 14. What's Already Implemented vs TODO

### DONE (engine + server + client)

- [x] `engine/paper.py` — PaperState, create_flat_sheet, FOLD I/O
- [x] `engine/fold.py` — apply_fold with full 10-step pipeline
- [x] `engine/physics.py` — Bar-and-hinge Verlet solver (stabilized)
- [x] `engine/validation.py` — Kawasaki, Maekawa, self-intersection
- [x] `engine/metrics.py` — 20+ metrics computation
- [x] `engine/materials.py` — 4 material presets
- [x] `models.py` — OpenEnv Action/Observation/State subclasses
- [x] `origami_environment.py` — Environment subclass with reset/step/state
- [x] `tasks.py` — 7 tasks across 4 difficulty levels
- [x] `app.py` — create_app() integration
- [x] `client/client.py` — EnvClient subclass
- [x] `client/reward_functions.py` — GRPO reward functions
- [x] `renderer/exporter.py` — FOLD JSON + OBJ export
- [x] `openenv.yaml`, `pyproject.toml`, `Dockerfile`

### TODO

- [x] Remove matplotlib rendering (render_2d, render_3d, screenshots, recorder)
- [x] Remove render_urls from OrigamiObservation
- [x] Remove matplotlib/Pillow/imageio from requirements.txt
- [x] Update origami_environment.py to not call capture_step()
- [x] Build Three.js viewer (`viewer/index.html`)
- [x] Update Dockerfile to copy viewer/
- [x] Write `training/train_grpo.py` (2048 pattern)
- [x] Test openenv validate (passes: "Ready for multi-mode deployment")
