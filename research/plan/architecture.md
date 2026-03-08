# Origami RL Environment — Final Architecture

## The Idea

An OpenEnv environment where an LLM learns to design optimal fold patterns for real-world folding problems — solar panel packing, medical stent deployment, shelter construction. The LLM generates a `fold_strategy()` function (code-as-policy, same as the 2048 pattern), which is executed against our origami simulation engine.

---

## 1. Action Space

Based on the Huzita-Justin axioms and fold type research, we use a **named-fold-level** action space (not raw axioms — too low-level for LLM code generation).

### Fold Operations Available to the Agent

```python
class FoldAction:
    fold_type: str        # "valley", "mountain", "reverse_inside", "reverse_outside",
                          # "squash", "petal", "pleat", "crimp"
    fold_line: FoldLine   # Defined by two points OR angle + offset from edge
    fold_angle: float     # 0-180 degrees (how far to fold)
    layer_select: str     # "top", "all", "specific" (which layers to fold)
```

### Simplified Action Space (Start Here)

For the **code-as-policy** approach, the LLM writes a function that returns a list of fold instructions:

```python
def fold_strategy(paper_state: dict) -> list[dict]:
    """
    paper_state = {
        "width": 1.0, "height": 1.0,
        "material": {"name": "mylar", "thickness_mm": 0.05, "youngs_modulus_gpa": 4.0},
        "vertices": [[x,y,z], ...],
        "edges": [[v1,v2], ...],
        "assignments": ["B","B",...],  # current M/V/B assignments
        "fold_angles": [0, 0, ...],    # current fold angles in degrees
        "num_layers_at_center": 1,
        "bounding_box": {"x": 1.0, "y": 1.0, "z": 0.0},
    }

    Returns list of fold operations:
    [
        {"type": "valley", "line": {"start": [0,0.5], "end": [1,0.5]}, "angle": 180},
        {"type": "mountain", "line": {"start": [0.5,0], "end": [0.5,0.5]}, "angle": 180},
        ...
    ]
    """
```

### Why This Works

- LLM can reason about geometry in natural language / code
- Fold operations map directly to the simulation engine
- The strategy function is self-contained (sandbox-safe)
- Same pattern as 2048 — proven with GRPO training

---

## 2. State Representation

Based on FOLD format (the standard) + bar-and-hinge physics model.

### Internal State (Simulation Engine)

```python
@dataclass
class PaperState:
    # Geometry (FOLD format compatible)
    vertices: np.ndarray          # (N, 3) vertex positions
    edges: np.ndarray             # (E, 2) edge vertex indices
    faces: list[list[int]]        # face vertex indices
    assignments: list[str]        # ["M","V","B","F"] per edge
    fold_angles: np.ndarray       # (E,) current fold angle per edge, degrees

    # Physics
    rest_lengths: np.ndarray      # (E,) original edge lengths
    strain: np.ndarray            # (N,) per-vertex Cauchy strain
    energy: float                 # total elastic energy

    # Layer tracking
    face_orders: list[tuple]      # [(f1, f2, +1/-1), ...] layer ordering
    num_layers: int               # max layer count

    # Material
    material: Material            # thickness, Young's modulus, max_strain

    # Metrics (computed after each fold)
    bounding_box: np.ndarray      # (3,) folded bounding box dimensions
    deployment_ratio: float       # folded_area / unfolded_area
    is_valid: bool                # no self-intersections, theorems satisfied
    kawasaki_violation: float     # sum of angle violations
    maekawa_violation: float      # sum of M-V violations
    self_intersections: int       # count of face-face penetrations
```

### Observation (What the LLM Sees via Prompt)

```python
class OrigamiObservation(Observation):
    paper_state: dict             # Serialized PaperState (simplified)
    task: dict                    # What to optimize
    metrics: dict                 # Current scores
    fold_history: list[dict]      # Previous folds applied
    error: str | None             # If last fold was invalid, why
```

### Text Prompt Format

```
TASK: Fold a 1m x 1m Mylar sheet to minimize packed volume while maintaining deployability.

MATERIAL:
  - Name: Mylar (space-grade)
  - Thickness: 0.05mm
  - Young's modulus: 4 GPa
  - Max strain before failure: 3%

CONSTRAINTS:
  - Must pack into bounding box <= 15cm x 15cm x 5cm
  - Must deploy to >= 0.8m^2 area when unfolded
  - Maximum 20 fold operations
  - No self-intersections allowed

CURRENT STATE:
  - Sheet: 1.0m x 1.0m, flat (0 folds applied)
  - Bounding box: 1.0m x 1.0m x 0.0m
  - Deployment ratio: 1.0 (fully deployed)
  - Strain: 0.0 (no stress)

Write a fold_strategy(paper_state) function that returns a list of fold operations.
Each fold: {"type": "valley"|"mountain", "line": {"start": [x,y], "end": [x,y]}, "angle": 0-180}
```

---

## 3. Physics Engine

Based on the bar-and-hinge model (Ghassaei's approach, ported to NumPy).

### Module Layout

```
engine/
  paper.py          -> Paper data structure, FOLD I/O
  fold_engine.py    -> Apply fold operations to paper geometry
  physics.py        -> Bar-and-hinge energy computation, strain calculation
  validation.py     -> Kawasaki, Maekawa, self-intersection checks
  metrics.py        -> Deployment ratio, compactness, shape similarity
  materials.py      -> Material definitions (paper, mylar, aluminum, etc.)
```

### Fold Execution Pipeline

```
Input: FoldAction (type, line, angle)
  |
  +-- 1. Validate fold line (does it intersect the paper?)
  +-- 2. Determine affected vertices (which side of fold line)
  +-- 3. Apply rotation to affected vertices
  |       - Rotate around fold line by fold_angle
  |       - Using quaternion rotation for numerical stability
  +-- 4. Update edge assignments (M/V based on fold type)
  +-- 5. Update fold angles array
  +-- 6. Compute new face topology (split faces at fold line if needed)
  +-- 7. Run physics step
  |       - Compute bar energies (stretching)
  |       - Compute facet hinge energies (panel bending)
  |       - Compute fold hinge energies (crease folding)
  |       - Iterative solver: minimize total energy
  +-- 8. Compute strain per vertex
  +-- 9. Check validity
  |       - Kawasaki theorem at all vertices
  |       - Maekawa theorem at all vertices
  |       - Self-intersection detection (triangle-triangle)
  +-- 10. Update metrics
  |       - Bounding box, deployment ratio, layer count
  |
  Output: Updated PaperState + validity report
```

### Energy Formulation (from research)

```python
# Bar-and-hinge model: three energy components
E_total = E_bar + E_facet + E_fold

E_bar   = sum_bars   (1/2) * k_axial * (L - L0)^2     # stretching
E_facet = sum_facets (1/2) * k_facet * l * (theta - pi)^2  # panel bending
E_fold  = sum_folds  (1/2) * k_fold  * l * (rho - rho_target)^2  # crease folding

# Stiffness parameters (from material properties)
k_axial = E * t * w / L0
k_facet = E * t^3 / (12 * (1 - nu^2))
k_fold  = kappa  # crease torsional stiffness
```

### Strain Computation (Ghassaei's formula)

```python
def compute_strain(vertices, edges, rest_lengths):
    """Per-vertex Cauchy strain = avg percent deviation of edge lengths."""
    strain = np.zeros(len(vertices))
    for v_idx in range(len(vertices)):
        neighbor_edges = get_edges_at_vertex(v_idx, edges)
        deviations = []
        for e_idx in neighbor_edges:
            v1, v2 = edges[e_idx]
            L = np.linalg.norm(vertices[v1] - vertices[v2])
            L0 = rest_lengths[e_idx]
            deviations.append(abs(L - L0) / L0)
        strain[v_idx] = np.mean(deviations) if deviations else 0.0
    return strain
```

---

## 4. Reward Functions

Three reward functions (same pattern as 2048):

### Reward 1: `code_valid(completions)`

Does the LLM output compile and produce valid fold instructions?

| Condition | Score |
|-----------|-------|
| Valid function returning fold list | +1.0 |
| Correct structure but exec fails | -0.5 |
| No function / syntax error | -2.0 |
| Non-stdlib imports | -20.0 |

### Reward 2: `physically_valid(completions)`

Are the folds physically possible?

| Condition | Score |
|-----------|-------|
| All folds valid, no violations | +1.0 |
| Per Kawasaki violation | -2.0 each |
| Per Maekawa violation | -2.0 each |
| Any self-intersection | -5.0 |
| Strain exceeds material limit | -1.0 |
| Function broken / can't run | 0.0 |

### Reward 3: `fold_quality(completions)`

How good is the folding solution?

| Condition | Score |
|-----------|-------|
| Compactness (1 - deployment_ratio) | +20.0 * compactness |
| Meets volume constraint (fits in target box) | +10.0 bonus |
| Deployable (can unfold cleanly) | +5.0 bonus |
| Per fold (efficiency penalty) | -0.5 each |
| High strain (material stress) | -3.0 * (max_strain / limit) |
| Timeout (>5 sec) | -1.0 |
| Exception during execution | -3.0 |

---

## 5. OpenEnv Integration

### Server: OrigamiEnvironment

```python
class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):

    def reset(self, seed=None, episode_id=None, **kwargs):
        task = self._sample_task()
        self._paper = create_flat_sheet(task["width"], task["height"], task["material"])
        self._task = task
        self._fold_history = []
        return self._make_observation()

    def step(self, action: OrigamiAction, **kwargs):
        # Extract and sandbox the fold strategy code
        strategy_fn = sandbox_execute(action.fold_code)
        folds = strategy_fn(self._paper.to_dict())

        # Apply each fold
        for fold in folds:
            result = self._engine.apply_fold(self._paper, fold)
            if not result.valid:
                return self._make_observation(error=result.error, done=True, reward=-5.0)
            self._paper = result.new_state
            self._fold_history.append(fold)

        # Compute final metrics and reward
        metrics = self._compute_metrics()
        reward = self._compute_reward(metrics)
        return self._make_observation(done=True, reward=reward)
```

### Task Pool (Curriculum)

```python
TASK_POOL = [
    # Level 1: Simple folds
    {"name": "half_fold", "width": 1.0, "height": 1.0,
     "material": "paper", "target_ratio": 0.5, "max_folds": 3},

    # Level 2: Multi-fold packing
    {"name": "letter_fold", "width": 1.0, "height": 1.0,
     "material": "paper", "target_ratio": 0.33, "max_folds": 5},

    # Level 3: Miura-ori discovery
    {"name": "solar_panel", "width": 1.0, "height": 1.0,
     "material": "mylar", "target_ratio": 0.05, "max_folds": 20,
     "must_deploy": True, "target_box": [0.15, 0.15, 0.05]},

    # Level 4: Constrained engineering
    {"name": "stent_fold", "width": 0.1, "height": 0.03,
     "material": "nitinol", "target_shape": "cylinder",
     "deployed_diameter": 0.01, "compressed_diameter": 0.003},
]
```

---

## 6. Rendering Pipeline

### Training Time (Server-Side, Headless)

```python
# matplotlib — fast, no GPU needed
def render_crease_pattern_2d(state) -> Image:
    """M=red dashed, V=blue dash-dot, B=black solid"""

def render_folded_3d(state) -> Image:
    """3D wireframe with strain colors (blue->red)"""
```

### Demo Time (Client-Side, React + Three.js)

```
React App (Docker Space on HF)
+-- CreasePatternPanel (SVG, left)
|   +-- Edges colored by M/V/B assignment
+-- FoldedView3D (@react-three/fiber, right)
|   +-- BufferGeometry with vertex colors (strain heatmap)
|   +-- OrbitControls for rotation
|   +-- Animation: step through fold sequence
+-- MetricsDashboard (bottom)
|   +-- Compactness, fold count, strain, validity
+-- MaterialSelector (sidebar)
    +-- Paper, Mylar, Aluminum, Nitinol
```

### Tech Stack

| Component | Library |
|-----------|---------|
| 3D scene | @react-three/fiber |
| Controls | @react-three/drei |
| Strain heatmap | Three.js Lut + vertex colors |
| 2D crease pattern | Inline SVG |
| Screenshots | canvas.toDataURL() |
| Recording | CCapture.js / MediaRecorder |
| HF Spaces | Docker Space (FastAPI + static React build) |

---

## 7. Project Structure

```
origami/
  research/                      # All research docs
    research.md                  # Index
    openenv/                     # OpenEnv framework research
    origami/                     # Domain research
    plan/                        # Architecture docs

  engine/                        # Core simulation (numpy/scipy only)
    __init__.py
    paper.py                     # Paper data structure, FOLD I/O
    fold_engine.py               # Apply folds (quaternion rotation)
    physics.py                   # Bar-and-hinge energy, strain
    validation.py                # Kawasaki, Maekawa, self-intersection
    metrics.py                   # Deployment ratio, compactness
    materials.py                 # Material definitions

  environment/                   # OpenEnv server
    __init__.py
    models.py                    # Action, Observation, State
    origami_environment.py       # Environment (reset/step/state)
    tasks.py                     # Task pool / curriculum
    app.py                       # create_app()
    Dockerfile
    requirements.txt

  client/                        # OpenEnv client + training bridge
    __init__.py
    client.py                    # EnvClient subclass
    reward_functions.py          # code_valid, physically_valid, fold_quality

  renderer/                      # Visualization
    server_render.py             # matplotlib headless
    web/                         # React app
      package.json
      src/
        App.tsx
        CreasePattern.tsx        # 2D SVG view
        FoldedView3D.tsx         # R3F 3D view
        StrainHeatmap.tsx        # Vertex color mapping
        FoldAnimation.tsx        # Step-through animation
        MetricsDashboard.tsx     # Scores display

  training/                      # Colab notebook
    train_origami.ipynb          # GRPO training (Unsloth + TRL)
    prompts.py                   # LLM prompt templates

  openenv.yaml                   # Manifest
  pyproject.toml
  README.md
```

---

## 8. Implementation Order

### Phase 1: Engine (first)
1. `paper.py` — Paper class, flat sheet creation, FOLD JSON serialize
2. `fold_engine.py` — Valley/mountain folds via quaternion rotation
3. `validation.py` — Kawasaki check, Maekawa check, basic self-intersection
4. `metrics.py` — Bounding box, deployment ratio, fold count
5. Test: fold a sheet in half, verify metrics

### Phase 2: OpenEnv Server
1. `models.py` — Pydantic models
2. `origami_environment.py` — reset/step
3. `app.py` + `Dockerfile`
4. `tasks.py` — 3-4 tasks
5. Test: curl the server, verify reset/step

### Phase 3: Reward + Training
1. `reward_functions.py` — three reward functions
2. `prompts.py` — prompt template
3. `train_origami.ipynb` — Colab GRPO notebook
4. Test: few training steps, verify rewards

### Phase 4: Rendering + Demo
1. `server_render.py` — matplotlib 2D + 3D
2. React app scaffold — R3F + SVG
3. Docker Space deployment
4. Record 1-minute demo video

---

## 9. Decisions (Locked)

| Decision | Choice | Why |
|----------|--------|-----|
| LLM interaction | Code-as-policy | Proven with 2048, hackathon expects it |
| Action space | Named fold ops + line + angle | Right level for LLM code gen |
| State format | FOLD-compatible JSON | Industry standard |
| Physics | Bar-and-hinge (NumPy) | Fast for RL, captures strain |
| Validation | Kawasaki + Maekawa + tri-tri intersection | Sound, polynomial time |
| Primary task | Solar panel packing | Unique, real-world, great demo |
| Training render | matplotlib headless | No GPU needed |
| Demo render | React + @react-three/fiber | Interactive, strain heatmap |
| Training | GRPO via TRL + Unsloth | Required by hackathon |
| Deployment | Docker Space on HF | Full control |

---

## 10. Why This Wins

1. **Unique** — nobody else doing origami RL
2. **Real-world** — NASA solar panels, medical stents, deployable shelters
3. **Demoable** — visual folding animation, strain heatmap, 1-min video
4. **Technically deep** — bar-and-hinge physics, Kawasaki/Maekawa math, material science
5. **Scalable** — FOLD format standard, material system, task curriculum
6. **Multi-statement** — Statement 2 (long-horizon planning) + Statement 3.1 (world modeling) + Statement 4 (self-improvement via curriculum)
