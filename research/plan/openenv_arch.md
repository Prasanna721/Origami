# OpenEnv Environment Architecture — Origami RL

> Complete blueprint. Everything lives inside the environment.
> Engine, physics, rendering, recording — all one deployable unit.

---

## 1. Overview

One Docker container on HF Spaces. Serves the OpenEnv API (WebSocket/REST) AND the React demo UI. Contains the full origami simulation engine, physics solver, validator, renderer, and metric system.

```
┌─────────────────────────────────────────────────────────┐
│                  HF Space (Docker)                       │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              FastAPI (app.py)                       │  │
│  │                                                    │  │
│  │  /ws, /reset, /step, /state   →  OpenEnv API       │  │
│  │  /                            →  React build       │  │
│  │  /renders/*                   →  Screenshots/GIFs  │  │
│  │  /export/*                    →  FOLD JSON export  │  │
│  └─────────────────┬──────────────────────────────────┘  │
│                    │                                      │
│  ┌─────────────────▼──────────────────────────────────┐  │
│  │         OrigamiEnvironment                          │  │
│  │         reset() / step() / state                    │  │
│  │                                                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │  Engine   │  │ Renderer │  │  Task System     │  │  │
│  │  │          │  │          │  │                  │  │  │
│  │  │ paper    │  │ render2d │  │ task pool       │  │  │
│  │  │ fold     │  │ render3d │  │ materials       │  │  │
│  │  │ physics  │  │ capture  │  │ curriculum      │  │  │
│  │  │ validate │  │ record   │  │                  │  │  │
│  │  │ metrics  │  │ export   │  │                  │  │  │
│  │  │ material │  │          │  │                  │  │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         React Frontend (static build)               │  │
│  │  CreasePattern(SVG) + FoldedView3D(R3F) + Metrics  │  │
│  │  FoldAnimation + StrainHeatmap + MaterialSelector  │  │
│  │  ScreenshotButton + RecordButton                   │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
origami_env/                             # THE deliverable — one package
│
├── server/                              # Python backend (everything)
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
│   ├── renderer/                        # All visualization
│   │   ├── __init__.py
│   │   ├── render_2d.py                 # matplotlib: crease pattern SVG/PNG (M=red, V=blue, B=black)
│   │   ├── render_3d.py                 # matplotlib: 3D wireframe + strain heatmap
│   │   ├── screenshots.py              # Per-step PNG capture, episode summary grid
│   │   ├── recorder.py                 # GIF assembly from frames (imageio), fold animation
│   │   └── exporter.py                 # FOLD JSON export, OBJ/STL export for 3D printing
│   │
│   ├── models.py                        # Pydantic: OrigamiAction, OrigamiObservation, OrigamiState
│   ├── origami_environment.py           # Environment class — reset/step/state (calls engine + renderer)
│   ├── tasks.py                         # Task pool, curriculum levels, difficulty sampling
│   ├── app.py                           # FastAPI: OpenEnv API + static React + render serving
│   ├── requirements.txt                 # numpy, scipy, matplotlib, imageio, pydantic, openenv-core
│   └── Dockerfile                       # Build React + run FastAPI
│
├── web/                                 # React frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx                      # Main layout: 2D + 3D + metrics + controls
│       ├── components/
│       │   ├── CreasePattern.tsx         # SVG: edges colored by M/V/B/F/U assignment
│       │   ├── FoldedView3D.tsx         # R3F: BufferGeometry + OrbitControls + DoubleSide
│       │   ├── StrainHeatmap.tsx        # Three.js Lut: vertex colors blue→red
│       │   ├── FoldAnimation.tsx        # Timeline slider, play/pause, fold interpolation
│       │   ├── MetricsDashboard.tsx     # Cards: all metrics with live updates
│       │   ├── MaterialSelector.tsx     # Dropdown: paper/mylar/aluminum/nitinol
│       │   ├── TaskSelector.tsx         # Pick task from curriculum
│       │   └── CaptureControls.tsx      # Screenshot (canvas.toDataURL) + Record (MediaRecorder)
│       ├── hooks/
│       │   └── useEnvironment.ts        # WebSocket connection to OpenEnv server
│       └── types.ts                     # TypeScript interfaces matching server models
│
├── client/                              # OpenEnv client (for remote connection + training)
│   ├── __init__.py
│   ├── client.py                        # OrigamiEnvClient (EnvClient subclass)
│   └── reward_functions.py              # code_valid, no_cheating, fold_quality
│
├── openenv.yaml                         # Manifest
├── pyproject.toml
└── README.md
```

---

## 3. Pydantic Models (`server/models.py`)

### OrigamiAction

One fold per step. Multi-step episodes (like 2048 where each step = one move).

```python
class OrigamiAction(Action):
    """One fold operation."""
    # metadata: Dict[str, Any]  (inherited)

    fold_type: str                    # "valley" | "mountain" | "pleat" | "crimp" | "stop"
    fold_line: Dict[str, List[float]] # {"start": [x,y], "end": [x,y]}  (normalized 0-1)
    fold_angle: float = 180.0         # degrees, 0-180 (180 = fully folded)
    layer_select: str = "all"         # "all" | "top" | "bottom"
```

`"stop"` action ends the episode and triggers final metrics + reward.

### OrigamiObservation

Everything the frontend AND the LLM need. Returned by both `reset()` and `step()`.

```python
class OrigamiObservation(Observation):
    # done: bool           (inherited)
    # reward: float|None   (inherited)
    # metadata: Dict       (inherited)

    # ── Task ──────────────────────────────────────────
    task: Dict[str, Any] = Field(default_factory=dict)
    # {
    #   "name": "solar_panel",
    #   "description": "Pack a 1m x 1m Mylar solar panel...",
    #   "width": 1.0, "height": 1.0,
    #   "material": {"name": "mylar", "thickness_mm": 0.05, "youngs_modulus_gpa": 4.0, "max_strain": 0.03},
    #   "target_ratio": 0.05,
    #   "max_folds": 20,
    #   "target_box": [0.15, 0.15, 0.05],
    #   "must_deploy": True,
    #   "difficulty": 3,
    # }

    # ── Paper State (FOLD-compatible) ─────────────────
    paper_state: Dict[str, Any] = Field(default_factory=dict)
    # {
    #   "vertices_coords": [[x,y,z], ...],          # (N,3) current 3D positions
    #   "edges_vertices": [[v1,v2], ...],            # (E,2) edge connectivity
    #   "faces_vertices": [[v0,v1,v2,...], ...],     # face polygons (CCW)
    #   "edges_assignment": ["M","V","B","F",...],   # per-edge type
    #   "edges_foldAngle": [-180, 180, 0, ...],     # per-edge target angle (degrees)
    #   "num_vertices": 25,
    #   "num_edges": 48,
    #   "num_faces": 24,
    #   "bounding_box": [0.15, 0.15, 0.05],         # folded dimensions
    #   "num_layers": 8,
    #   "material": {"name": "mylar", ...},
    #
    #   # Physics state
    #   "strain_per_vertex": [0.01, 0.005, ...],    # per-vertex Cauchy strain
    #   "energy": {
    #     "total": 0.34,
    #     "bar": 0.12,                               # stretching energy
    #     "facet": 0.08,                              # panel bending energy
    #     "fold": 0.14,                               # crease folding energy
    #   },
    # }

    # ── Metrics ───────────────────────────────────────
    metrics: Dict[str, Any] = Field(default_factory=dict)
    # {
    #   ## Validity
    #   "is_valid": True,
    #   "kawasaki_violations": 0,                     # count of vertices violating Kawasaki
    #   "kawasaki_total_error": 0.0,                  # sum of |alt_angle_sum - 180| degrees
    #   "maekawa_violations": 0,                      # count of vertices violating |M-V|=2
    #   "self_intersections": 0,                      # count of face-face penetrations
    #   "strain_exceeded": False,                     # any vertex > material.max_strain?
    #
    #   ## Compactness
    #   "deployment_ratio": 0.05,                     # folded_area / original_area
    #   "compactness": 0.95,                          # 1 - deployment_ratio
    #   "volume_compaction": 0.001,                   # bbox_folded / bbox_original
    #   "packing_efficiency": 0.72,                   # material_volume / bbox_volume
    #   "fits_target_box": True,                      # fits inside task.target_box?
    #
    #   ## Structural
    #   "max_strain": 0.02,                           # max per-vertex strain
    #   "mean_strain": 0.008,                         # mean per-vertex strain
    #   "total_energy": 0.34,                         # total elastic energy
    #   "energy_bar": 0.12,
    #   "energy_facet": 0.08,
    #   "energy_fold": 0.14,
    #
    #   ## Efficiency
    #   "fold_count": 8,                              # number of folds applied
    #   "folding_efficiency": 0.119,                  # compactness / fold_count
    #   "crease_complexity": 0.85,                    # entropy of M/V assignment distribution
    #
    #   ## Deployability
    #   "is_deployable": True,                        # reverse fold simulation passed?
    #   "deployment_force_estimate": 0.45,            # Newtons (from energy gradient)
    #
    #   ## Shape similarity (if task has target_shape)
    #   "chamfer_distance": null,                     # avg nearest-point distance
    #   "hausdorff_distance": null,                   # max distance
    # }

    # ── Fold History ──────────────────────────────────
    fold_history: List[Dict[str, Any]] = Field(default_factory=list)
    # [ {"type":"valley", "line":{"start":[0,0.5],"end":[1,0.5]}, "angle":180, "step":1}, ... ]

    # ── Error ─────────────────────────────────────────
    error: Optional[str] = None
    # "Fold line does not intersect paper boundary"
    # "Self-intersection detected after fold 3"
    # etc.

    # ── Render URLs ───────────────────────────────────
    render_urls: Dict[str, str] = Field(default_factory=dict)
    # {
    #   "crease_2d": "/renders/ep_abc123/crease_step_4.png",
    #   "folded_3d": "/renders/ep_abc123/folded_step_4.png",
    #   "strain_heatmap": "/renders/ep_abc123/strain_step_4.png",
    #   "episode_gif": "/renders/ep_abc123/animation.gif",    # only when done=True
    #   "fold_json": "/export/ep_abc123/state.fold",          # FOLD format export
    # }
```

### OrigamiState

Server-side episode tracking.

```python
class OrigamiState(State):
    # episode_id: Optional[str]  (inherited)
    # step_count: int            (inherited)

    task_name: str = ""
    num_folds_applied: int = 0
    is_valid: bool = True
    total_reward: float = 0.0
    current_fold_percent: float = 1.0     # for animation: how far current fold has progressed
```

---

## 4. Engine (`server/engine/`)

### 4.1 Paper State (`paper.py`)

The core data structure. FOLD-format compatible.

```python
@dataclass
class PaperState:
    # ── Geometry (FOLD format) ────────────────────────
    vertices_coords: np.ndarray       # (N, 3) vertex positions (3D)
    edges_vertices: np.ndarray        # (E, 2) edge connectivity, dtype int
    faces_vertices: list[list[int]]   # ragged: face polygons as vertex index lists (CCW)
    edges_assignment: list[str]       # (E,) "M" | "V" | "B" | "F" | "U" per edge
    edges_foldAngle: np.ndarray       # (E,) target fold angle in degrees per edge

    # ── Physics ───────────────────────────────────────
    rest_lengths: np.ndarray          # (E,) original edge lengths (set at creation)
    rest_positions: np.ndarray        # (N, 3) original flat positions (for strain reference)
    strain_per_vertex: np.ndarray     # (N,) per-vertex Cauchy strain
    energy: dict                      # {"total": float, "bar": float, "facet": float, "fold": float}

    # ── Layers ────────────────────────────────────────
    face_orders: list[tuple]          # [(face_i, face_j, +1/-1), ...] layer ordering
    num_layers: int                   # max stacking depth

    # ── Material ──────────────────────────────────────
    material: Material                # thickness_mm, youngs_modulus_gpa, max_strain, poisson_ratio

    # ── Metadata ──────────────────────────────────────
    width: float                      # original sheet width (meters)
    height: float                     # original sheet height (meters)
    fold_count: int                   # number of folds applied so far
```

**Key methods:**

```python
create_flat_sheet(width, height, material) -> PaperState
    # Creates a rectangular sheet: 4 vertices, 4 boundary edges, 1 face
    # (or subdivided grid for higher resolution physics)

PaperState.to_fold_json() -> dict
    # Exports FOLD-format JSON (vertices_coords, edges_vertices, edges_assignment, etc.)

PaperState.from_fold_json(data: dict) -> PaperState
    # Imports from FOLD JSON

PaperState.to_observation_dict() -> dict
    # Simplified dict for the Observation (includes strain, energy, bounding_box)

PaperState.bounding_box -> np.ndarray  # (3,) min bounding box dimensions

PaperState.triangulated_faces -> list[list[int]]
    # Ear-clipping triangulation of polygon faces (needed for physics + rendering)
```

### 4.2 Fold Operations (`fold.py`)

Applies one fold to the paper. Returns new PaperState.

```python
def apply_fold(paper: PaperState, fold: dict) -> PaperState:
    """
    fold = {
        "type": "valley" | "mountain" | "pleat" | "crimp",
        "line": {"start": [x, y], "end": [x, y]},
        "angle": 0-180,
        "layer_select": "all" | "top" | "bottom",
    }
    """
```

**The 10-step fold pipeline:**

```
Step 1:  VALIDATE fold line
         - Does line intersect the paper boundary at 2+ points?
         - Is angle in valid range?
         - Raise FoldError if invalid

Step 2:  SPLIT FACES at fold line
         - Find all faces intersected by the fold line
         - Split each intersected face into sub-faces
         - Add new vertices at intersection points
         - Add new edges along the fold line
         - Update faces_vertices, edges_vertices, edges_assignment

Step 3:  CLASSIFY VERTICES
         - For each vertex, determine which side of fold line it's on
         - Use signed distance: d = (point - line_start) x line_direction
         - Positive side = moving side, negative side = fixed side
         - Vertices ON the line (|d| < epsilon) are hinge vertices

Step 4:  APPLY ROTATION (quaternion)
         - Rotation axis = fold line direction vector (normalized)
         - Rotation angle = fold_angle (valley=positive, mountain=negative)
         - For each moving vertex:
           translate to fold line origin → apply quaternion → translate back
         - Quaternion rotation for numerical stability (no gimbal lock)

Step 5:  UPDATE EDGE ASSIGNMENTS
         - New edges along fold line: "M" or "V" based on fold_type
         - Valley fold → "V" (positive fold angle, paper toward you)
         - Mountain fold → "M" (negative fold angle, paper away)

Step 6:  UPDATE FOLD ANGLES
         - New crease edges: set target angle (±180 for full fold, ±fold_angle otherwise)
         - Existing assignments unchanged

Step 7:  UPDATE FACE TOPOLOGY
         - Recompute faces_vertices after split
         - Recompute face_orders (layer ordering) based on rotation

Step 8:  COMPUTE REST LENGTHS for new edges
         - rest_length = euclidean distance in the FLAT (unfolded) configuration
         - This is the reference for strain computation

Step 9:  INCREMENT fold_count

Step 10: RETURN new PaperState
         - Physics and validation run SEPARATELY (called by environment)
         - This keeps fold.py focused on geometry only
```

**Pleat and crimp are compound folds:**

```python
def apply_pleat(paper, line1, line2, angle):
    """Two parallel folds: valley at line1, mountain at line2."""
    paper = apply_fold(paper, {"type": "valley", "line": line1, "angle": angle})
    paper = apply_fold(paper, {"type": "mountain", "line": line2, "angle": angle})
    return paper

def apply_crimp(paper, line1, line2, angle):
    """Two parallel folds: mountain at line1, valley at line2 (reverse of pleat)."""
    paper = apply_fold(paper, {"type": "mountain", "line": line1, "angle": angle})
    paper = apply_fold(paper, {"type": "valley", "line": line2, "angle": angle})
    return paper
```

### 4.3 Physics Solver (`physics.py`)

Bar-and-hinge model. NumPy port of Ghassaei's GPU solver.

**Three constraint types (energy components):**

```python
# E_total = E_bar + E_facet + E_fold

# 1. BAR (axial spring) — every edge resists stretching/compression
#    E_bar = Σ (1/2) * k_axial * (L - L0)²
#    k_axial = E * t * w / L0
#    Where: E = Young's modulus, t = thickness, w = tributary width, L0 = rest length

# 2. FACET HINGE — triangulation diagonals keep faces flat
#    E_facet = Σ (1/2) * k_facet * l * (θ - π)²
#    k_facet = E * t³ / (12 * (1 - ν²))
#    Target angle = π (flat), high stiffness
#    l = hinge edge length

# 3. FOLD HINGE — crease edges drive toward target fold angle
#    E_fold = Σ (1/2) * k_fold * l * (ρ - ρ_target)²
#    k_fold = κ (crease torsional stiffness, user-adjustable)
#    ρ_target from edges_foldAngle * fold_percent
```

**Stiffness hierarchy:**

```python
# Prevents stretching while allowing controlled folding:
k_axial  = 70.0    # very stiff — bars don't stretch
k_facet  = 0.2     # moderate — faces stay flat but can flex slightly
k_fold   = 0.7     # soft — drives folding motion
```

**Verlet integration solver:**

```python
def simulate(paper: PaperState, fold_percent: float = 1.0, n_steps: int = 500,
             dt: float = 0.02, damping: float = 0.1) -> PaperState:
    """
    Run bar-and-hinge physics simulation.
    Updates vertex positions to satisfy fold constraints while minimizing energy.
    Computes strain per vertex and total energy.
    """
    pos = paper.vertices_coords.copy()          # (N, 3) current positions
    last_pos = pos.copy()                        # (N, 3) previous positions

    # Precompute topology
    beams = build_beam_list(paper)               # [(node_a, node_b, rest_len, k), ...]
    creases = build_crease_list(paper)           # [(n1, n2, n3, n4, target_angle, k, type), ...]

    for step in range(n_steps):
        forces = np.zeros_like(pos)              # (N, 3)

        # ── Beam forces (vectorized) ─────────────────
        for (a, b, L0, k) in beams:
            delta = pos[b] - pos[a]
            L = np.linalg.norm(delta)
            if L < 1e-12: continue
            strain = (L - L0) / L0
            F_mag = k * strain * L0
            F_dir = delta / L
            forces[a] += F_mag * F_dir
            forces[b] -= F_mag * F_dir

        # ── Crease forces (dihedral angle springs) ───
        for (n1, n2, n3, n4, target, k, ctype) in creases:
            actual_target = target * fold_percent if ctype == "fold" else target
            theta = compute_dihedral_angle(pos[n1], pos[n2], pos[n3], pos[n4])
            delta_theta = theta - actual_target
            torque = k * edge_length(pos[n1], pos[n2]) * delta_theta

            # Convert torque to forces on the 4 nodes
            f3, f4, f1, f2 = torque_to_forces(
                pos[n1], pos[n2], pos[n3], pos[n4], torque
            )
            forces[n1] += f1
            forces[n2] += f2
            forces[n3] += f3
            forces[n4] += f4

        # ── Verlet integration ───────────────────────
        new_pos = pos + (1.0 - damping) * (pos - last_pos) + forces * dt * dt
        last_pos = pos
        pos = new_pos

        # ── Convergence check ────────────────────────
        kinetic_energy = np.sum((pos - last_pos) ** 2)
        if kinetic_energy < 1e-10:
            break

    # ── Compute strain ───────────────────────────────
    paper.vertices_coords = pos
    paper.strain_per_vertex = compute_strain(pos, paper.edges_vertices, paper.rest_lengths)

    # ── Compute energy breakdown ─────────────────────
    paper.energy = {
        "total": compute_total_energy(pos, beams, creases, fold_percent),
        "bar": compute_bar_energy(pos, beams),
        "facet": compute_facet_energy(pos, creases, fold_percent),
        "fold": compute_fold_energy(pos, creases, fold_percent),
    }

    return paper
```

**Strain computation (Ghassaei's formula):**

```python
def compute_strain(vertices, edges, rest_lengths) -> np.ndarray:
    """Per-vertex Cauchy strain = average percent deviation of incident edge lengths."""
    strain = np.zeros(len(vertices))
    counts = np.zeros(len(vertices))

    for e_idx, (v1, v2) in enumerate(edges):
        L = np.linalg.norm(vertices[v1] - vertices[v2])
        L0 = rest_lengths[e_idx]
        edge_strain = abs(L - L0) / L0

        strain[v1] += edge_strain
        strain[v2] += edge_strain
        counts[v1] += 1
        counts[v2] += 1

    counts[counts == 0] = 1  # avoid division by zero
    return strain / counts
```

**Dihedral angle computation:**

```python
def compute_dihedral_angle(p1, p2, p3, p4) -> float:
    """
    Dihedral angle between planes (p1,p2,p3) and (p1,p2,p4).
    p1-p2 is the hinge edge. p3 and p4 are wing tips.

         p3
        / | \
       /  |  \
    p1----+----p2   (hinge edge)
       \  |  /
        \ | /
         p4

    Returns angle in radians. 0 = flat, π = folded 180°.
    """
    e = p2 - p1                        # hinge edge vector
    n1 = np.cross(p3 - p1, e)         # normal of face 1
    n2 = np.cross(e, p4 - p1)         # normal of face 2
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    cos_theta = np.clip(np.dot(n1, n2), -1, 1)
    sin_theta = np.dot(np.cross(n1, n2), e / np.linalg.norm(e))
    return np.arctan2(sin_theta, cos_theta)
```

### 4.4 Validation (`validation.py`)

```python
def validate_state(paper: PaperState) -> dict:
    """Run all validation checks. Returns violation report."""
    return {
        # ── Kawasaki-Justin theorem ───────────────────
        # At each interior vertex: alternating angle sum = 180°
        "kawasaki_violations": count_kawasaki_violations(paper),
        "kawasaki_total_error": sum_kawasaki_error(paper),  # degrees

        # ── Maekawa-Justin theorem ────────────────────
        # At each interior vertex: |M - V| = 2
        "maekawa_violations": count_maekawa_violations(paper),

        # ── Self-intersection ─────────────────────────
        # Triangle-triangle intersection test on all non-adjacent face pairs
        "self_intersections": count_self_intersections(paper),

        # ── Material limits ───────────────────────────
        # Is max strain within material tolerance?
        "strain_exceeded": bool(np.max(paper.strain_per_vertex) > paper.material.max_strain),
        "max_strain_ratio": float(np.max(paper.strain_per_vertex) / paper.material.max_strain),

        # ── Summary ───────────────────────────────────
        "is_valid": (
            count_kawasaki_violations(paper) == 0 and
            count_maekawa_violations(paper) == 0 and
            count_self_intersections(paper) == 0 and
            np.max(paper.strain_per_vertex) <= paper.material.max_strain
        ),
    }
```

**Kawasaki check:**

```python
def check_kawasaki_at_vertex(paper, vertex_idx) -> float:
    """Returns angular error in degrees. 0 = valid."""
    # Get all crease edges incident on this vertex, sorted by angle
    angles = get_sorted_crease_angles(paper, vertex_idx)
    if len(angles) < 2:
        return 0.0
    # Alternating sum: a1 - a2 + a3 - a4 + ... should = 0
    alternating = sum(a * (-1)**i for i, a in enumerate(angles))
    return abs(alternating)
```

**Self-intersection detection:**

```python
def count_self_intersections(paper) -> int:
    """Triangle-triangle intersection test. O(F²) but F is small for our models."""
    triangles = paper.triangulated_faces
    count = 0
    for i in range(len(triangles)):
        for j in range(i + 2, len(triangles)):  # skip adjacent faces
            if not faces_share_edge(triangles[i], triangles[j]):
                if triangles_intersect(
                    paper.vertices_coords[triangles[i]],
                    paper.vertices_coords[triangles[j]]
                ):
                    count += 1
    return count
```

### 4.5 Metrics (`metrics.py`)

All metrics computed from PaperState + task. Returns flat dict for the Observation.

```python
def compute_all_metrics(paper: PaperState, task: dict, validation: dict) -> dict:
    """Compute every metric. Called after physics + validation."""

    original_area = paper.width * paper.height
    bb = paper.bounding_box  # (3,) array
    original_bbox_vol = paper.width * paper.height * paper.material.thickness_mm / 1000
    folded_bbox_vol = bb[0] * bb[1] * bb[2]

    return {
        # ── Validity (from validation) ────────────────
        "is_valid": validation["is_valid"],
        "kawasaki_violations": validation["kawasaki_violations"],
        "kawasaki_total_error": validation["kawasaki_total_error"],
        "maekawa_violations": validation["maekawa_violations"],
        "self_intersections": validation["self_intersections"],
        "strain_exceeded": validation["strain_exceeded"],

        # ── Compactness ──────────────────────────────
        "deployment_ratio": folded_area(paper) / original_area,
        "compactness": 1.0 - (folded_area(paper) / original_area),
        "volume_compaction": folded_bbox_vol / original_bbox_vol if original_bbox_vol > 0 else 0,
        "packing_efficiency": material_volume(paper) / folded_bbox_vol if folded_bbox_vol > 0 else 0,
        "fits_target_box": fits_in_box(bb, task.get("target_box")),
        "bounding_box": bb.tolist(),

        # ── Structural ───────────────────────────────
        "max_strain": float(np.max(paper.strain_per_vertex)),
        "mean_strain": float(np.mean(paper.strain_per_vertex)),
        "total_energy": paper.energy["total"],
        "energy_bar": paper.energy["bar"],
        "energy_facet": paper.energy["facet"],
        "energy_fold": paper.energy["fold"],

        # ── Efficiency ───────────────────────────────
        "fold_count": paper.fold_count,
        "folding_efficiency": (1.0 - folded_area(paper) / original_area) / max(paper.fold_count, 1),
        "crease_complexity": assignment_entropy(paper.edges_assignment),

        # ── Deployability ────────────────────────────
        "is_deployable": check_deployability(paper) if task.get("must_deploy") else None,
        "deployment_force_estimate": estimate_deployment_force(paper),

        # ── Shape similarity (if target given) ───────
        "chamfer_distance": (
            compute_chamfer_distance(paper, task["target_shape"])
            if "target_shape" in task else None
        ),
        "hausdorff_distance": (
            compute_hausdorff_distance(paper, task["target_shape"])
            if "target_shape" in task else None
        ),
    }
```

### 4.6 Materials (`materials.py`)

```python
@dataclass
class Material:
    name: str
    thickness_mm: float          # mm
    youngs_modulus_gpa: float    # GPa
    max_strain: float            # fraction (0.05 = 5%)
    poisson_ratio: float = 0.3
    density_kg_m3: float = 1000.0

    # Derived stiffness parameters (for physics solver)
    @property
    def k_axial(self) -> float:
        """Axial stiffness coefficient."""
        return self.youngs_modulus_gpa * 1e9 * (self.thickness_mm / 1000)

    @property
    def k_facet(self) -> float:
        """Facet bending stiffness."""
        t = self.thickness_mm / 1000
        E = self.youngs_modulus_gpa * 1e9
        nu = self.poisson_ratio
        return E * t**3 / (12 * (1 - nu**2))

MATERIALS = {
    "paper": Material(
        name="paper",
        thickness_mm=0.1,
        youngs_modulus_gpa=3.0,
        max_strain=0.05,          # 5% — paper is forgiving
        poisson_ratio=0.3,
        density_kg_m3=700,
    ),
    "mylar": Material(
        name="mylar",
        thickness_mm=0.05,
        youngs_modulus_gpa=4.0,
        max_strain=0.03,          # 3% — space-grade film
        poisson_ratio=0.38,
        density_kg_m3=1390,
    ),
    "aluminum": Material(
        name="aluminum",
        thickness_mm=0.2,
        youngs_modulus_gpa=70.0,
        max_strain=0.01,          # 1% — rigid, cracks easily at creases
        poisson_ratio=0.33,
        density_kg_m3=2700,
    ),
    "nitinol": Material(
        name="nitinol",
        thickness_mm=0.15,
        youngs_modulus_gpa=75.0,
        max_strain=0.08,          # 8% — superelastic shape memory alloy
        poisson_ratio=0.33,
        density_kg_m3=6450,
    ),
}
```

---

## 5. Renderer (`server/renderer/`)

### 5.1 2D Crease Pattern (`render_2d.py`)

```python
def render_crease_pattern(paper: PaperState, output_path: str = None) -> Image:
    """
    matplotlib: crease pattern with standard origami colors.

    Edge colors:
      M (mountain) = red, dashed (dash-dot-dot)
      V (valley)   = blue, dash-dot
      B (boundary)  = black, solid
      F (flat)     = lightgray, solid thin
      U (unassigned) = gray, dotted

    Also renders:
      - Vertex dots (gray)
      - Fold angle labels (optional)
      - Sheet dimensions annotation
    """

def render_crease_pattern_svg(paper: PaperState) -> str:
    """Returns inline SVG string (for React frontend fallback)."""
```

### 5.2 3D Folded View (`render_3d.py`)

```python
def render_folded_state(paper: PaperState, output_path: str = None,
                        view_angle: tuple = (30, 45)) -> Image:
    """
    matplotlib mplot3d: wireframe + face shading with strain colors.

    - Faces colored by strain: blue (0) → yellow → red (max)
    - Edges drawn: M=red, V=blue, B=black
    - Colorbar showing strain scale
    - Bounding box wireframe overlay
    """

def render_strain_heatmap(paper: PaperState, output_path: str = None) -> Image:
    """
    Top-down view of strain distribution.
    Uses matplotlib tricontourf for smooth interpolation.
    Colorbar: blue (0 strain) → red (max strain).
    Material limit marked as dashed line on colorbar.
    """

def render_side_by_side(paper: PaperState, output_path: str = None) -> Image:
    """
    Combined figure:
    Left: 2D crease pattern
    Right: 3D folded state with strain colors
    Bottom: metrics text summary
    """
```

### 5.3 Screenshots (`screenshots.py`)

```python
def capture_step(paper: PaperState, step_num: int,
                 episode_dir: str) -> dict:
    """
    Save renders for one step. Returns dict of file paths.
    Creates:
      - {episode_dir}/crease_step_{N}.png       (2D crease pattern)
      - {episode_dir}/folded_step_{N}.png       (3D folded state)
      - {episode_dir}/strain_step_{N}.png       (strain heatmap)
      - {episode_dir}/state_step_{N}.fold       (FOLD JSON snapshot)
    """

def capture_episode_summary(paper: PaperState, fold_history: list,
                            task: dict, metrics: dict,
                            episode_dir: str) -> str:
    """
    Grid summary of entire episode. Returns path.
    Creates: {episode_dir}/summary.png

    Layout:
    ┌──────┬──────┬──────┬──────┐
    │Step 0│Step 1│Step 2│Step 3│   (crease patterns)
    ├──────┼──────┼──────┼──────┤
    │Step 0│Step 1│Step 2│Step 3│   (3D folded states)
    ├──────┴──────┴──────┴──────┤
    │  Final metrics + task     │
    └───────────────────────────┘
    """
```

### 5.4 Recording (`recorder.py`)

```python
def record_fold_animation(paper_initial: PaperState, fold_history: list,
                          output_path: str, fps: int = 15,
                          frames_per_fold: int = 10) -> str:
    """
    Generate animated GIF of the folding sequence.

    For each fold in history:
      - Interpolate fold_percent from 0.0 to 1.0 over frames_per_fold frames
      - Run physics at each fold_percent
      - Render 3D frame via matplotlib
    Assemble frames into GIF via imageio.

    Returns path to GIF.
    """

def record_strain_evolution(paper_initial: PaperState, fold_history: list,
                            output_path: str) -> str:
    """
    GIF showing how strain develops through the fold sequence.
    Useful for understanding which folds cause the most stress.
    """
```

### 5.5 Exporter (`exporter.py`)

```python
def export_fold_json(paper: PaperState, fold_history: list) -> dict:
    """
    Full FOLD JSON with multi-frame animation data.
    Can be loaded by OrigamiSimulator or our React frontend.

    Includes:
      - file_spec, file_creator, file_classes
      - frame_classes: ["creasePattern"]
      - All vertex/edge/face data
      - fold_history as custom metadata
    """

def export_obj(paper: PaperState) -> str:
    """Wavefront OBJ format for 3D printing / external renderers."""

def export_stl(paper: PaperState) -> bytes:
    """STL binary format for 3D printing."""
```

---

## 6. Environment (`server/origami_environment.py`)

The OpenEnv wrapper. Calls engine + renderer. Does NOT contain origami logic.

```python
class OrigamiEnvironment(Environment[OrigamiAction, OrigamiObservation, OrigamiState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._paper = None
        self._task = None
        self._fold_history = []
        self._metrics = {}
        self._validation = {}
        self._error = None
        self._episode_id = None
        self._step_count = 0
        self._episode_dir = None       # for renders

    # ── reset ─────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> OrigamiObservation:
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._fold_history = []
        self._error = None

        # Create episode render directory
        self._episode_dir = f"renders/ep_{self._episode_id[:8]}"
        os.makedirs(self._episode_dir, exist_ok=True)

        # Sample task
        self._task = kwargs.get("task") or sample_task(seed=seed)

        # Create flat sheet
        self._paper = create_flat_sheet(
            self._task["width"], self._task["height"],
            MATERIALS[self._task["material"]] if isinstance(self._task["material"], str)
            else self._task["material"]
        )

        # Initial metrics + validation
        self._validation = validate_state(self._paper)
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # Render initial state
        render_urls = capture_step(self._paper, 0, self._episode_dir)

        return self._make_observation(done=False, reward=None, render_urls=render_urls)

    # ── step ──────────────────────────────────────────

    def step(self, action: OrigamiAction, timeout_s=None, **kwargs) -> OrigamiObservation:
        self._step_count += 1
        self._error = None

        # ── Handle "stop" action ──────────────────────
        if action.fold_type == "stop":
            return self._finalize_episode()

        # ── Apply fold ────────────────────────────────
        fold_dict = {
            "type": action.fold_type,
            "line": action.fold_line,
            "angle": action.fold_angle,
            "layer_select": action.layer_select,
        }

        try:
            self._paper = apply_fold(self._paper, fold_dict)
            self._fold_history.append({**fold_dict, "step": self._step_count})
        except FoldError as e:
            self._error = str(e)
            return self._make_observation(done=True, reward=-5.0, render_urls={})

        # ── Run physics ───────────────────────────────
        try:
            self._paper = simulate(self._paper, fold_percent=1.0)
        except Exception as e:
            self._error = f"Physics failed: {e}"

        # ── Validate ──────────────────────────────────
        self._validation = validate_state(self._paper)

        # ── Compute metrics ───────────────────────────
        self._metrics = compute_all_metrics(self._paper, self._task, self._validation)

        # ── Render this step ──────────────────────────
        render_urls = capture_step(self._paper, self._step_count, self._episode_dir)

        # ── Check if episode should end ───────────────
        done = False
        reward = None

        # Auto-end on max folds
        if self._step_count >= self._task.get("max_folds", 50):
            done = True

        # Auto-end on critical failure
        if self._validation["self_intersections"] > 0:
            done = True
            self._error = "Self-intersection detected"

        if done:
            return self._finalize_episode()

        return self._make_observation(done=False, reward=None, render_urls=render_urls)

    # ── finalize ──────────────────────────────────────

    def _finalize_episode(self) -> OrigamiObservation:
        """End episode: compute final reward, generate animation GIF, export FOLD."""
        reward = self._compute_reward()

        render_urls = capture_step(self._paper, self._step_count, self._episode_dir)

        # Episode summary image
        summary_path = capture_episode_summary(
            self._paper, self._fold_history, self._task, self._metrics, self._episode_dir
        )
        render_urls["episode_summary"] = summary_path

        # Fold animation GIF
        try:
            gif_path = record_fold_animation(
                create_flat_sheet(self._task["width"], self._task["height"], self._task["material"]),
                self._fold_history,
                f"{self._episode_dir}/animation.gif"
            )
            render_urls["episode_gif"] = gif_path
        except Exception:
            pass  # non-critical

        # FOLD JSON export
        fold_json = export_fold_json(self._paper, self._fold_history)
        fold_path = f"{self._episode_dir}/state.fold"
        with open(fold_path, "w") as f:
            json.dump(fold_json, f)
        render_urls["fold_json"] = fold_path

        return self._make_observation(done=True, reward=reward, render_urls=render_urls)

    # ── state ─────────────────────────────────────────

    @property
    def state(self) -> OrigamiState:
        return OrigamiState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task["name"] if self._task else "",
            num_folds_applied=len(self._fold_history),
            is_valid=self._metrics.get("is_valid", True),
            total_reward=0.0,
        )

    # ── helpers ───────────────────────────────────────

    def _make_observation(self, done, reward, render_urls) -> OrigamiObservation:
        return OrigamiObservation(
            done=done,
            reward=reward,
            task=self._task or {},
            paper_state=self._paper.to_observation_dict() if self._paper else {},
            metrics=self._metrics,
            fold_history=self._fold_history,
            error=self._error,
            render_urls=render_urls,
        )

    def _compute_reward(self) -> float:
        m = self._metrics
        reward = 0.0
        reward += m.get("compactness", 0) * 20.0
        if m.get("fits_target_box", False): reward += 10.0
        if m.get("is_deployable", False): reward += 5.0
        reward -= m.get("kawasaki_violations", 0) * 2.0
        reward -= m.get("maekawa_violations", 0) * 2.0
        reward -= m.get("self_intersections", 0) * 5.0
        reward -= m.get("fold_count", 0) * 0.5
        max_strain = m.get("max_strain", 0)
        limit = self._paper.material.max_strain if self._paper else 0.05
        if max_strain > limit: reward -= 3.0 * (max_strain / limit)
        return reward
```

---

## 7. App + Docker (`server/app.py` + `server/Dockerfile`)

### `app.py`

```python
"""FastAPI entry point — serves OpenEnv API + React frontend + renders."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from models import OrigamiAction, OrigamiObservation
from origami_environment import OrigamiEnvironment

# OpenEnv app (handles /ws, /reset, /step, /state, /health)
app = create_app(
    OrigamiEnvironment,
    OrigamiAction,
    OrigamiObservation,
    env_name="origami_env",
)

# Serve rendered images/GIFs/FOLD exports
app.mount("/renders", StaticFiles(directory="renders"), name="renders")
app.mount("/export", StaticFiles(directory="renders"), name="export")

# Serve React frontend (built at Docker build time)
app.mount("/", StaticFiles(directory="../web/dist", html=True), name="frontend")
```

### `Dockerfile`

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest

# ── Install Node.js for React build ──────────────────
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ──────────────────────────────
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

# ── Build React frontend ─────────────────────────────
COPY web/ ./web/
RUN cd web && npm install && npm run build

# ── Copy server code ─────────────────────────────────
COPY server/ ./server/

# ── Create renders directory ──────────────────────────
RUN mkdir -p /app/server/renders

WORKDIR /app/server

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`

```
openenv-core[core]>=0.2.1
numpy>=1.24
scipy>=1.10
pydantic>=2.0
matplotlib>=3.7
imageio>=2.31
Pillow>=10.0
```

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

## 8. React Frontend (`web/`)

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  ORIGAMI RL ENVIRONMENT        [Task: ▼] [Material: ▼]     │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                   │
│   CREASE PATTERN (2D)    │      FOLDED STATE (3D)           │
│                          │                                   │
│   SVG                    │    R3F Canvas                     │
│   M = red dashed         │    OrbitControls (rotate/zoom)    │
│   V = blue dash-dot      │    Vertex colors = strain         │
│   B = black solid        │    DoubleSide material            │
│                          │    Ambient + directional light    │
│                          │                                   │
├──────────────────────────┴──────────────────────────────────┤
│  [|<] [<] [>] [>|]  ████████░░░░ Step 4 / 8                │
│                      [Play] [Pause]   Speed: [1x ▼]        │
├─────────────────────────────────────────────────────────────┤
│  METRICS                                                     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │
│  │Compactness│ │Max Strain │ │Fold Count │ │  Validity   │ │
│  │   85.0%   │ │  0.021    │ │     8     │ │   VALID     │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────┘ │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │
│  │Deploy Rat.│ │  Energy   │ │Pack Eff.  │ │Fits Target  │ │
│  │   15.2x   │ │   0.34    │ │   72%     │ │     YES     │ │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  [Screenshot PNG]  [Record GIF]  [Export FOLD]  [Export OBJ]│
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
React App
  │
  ├── useEnvironment() hook
  │     │
  │     ├── WebSocket connect to /ws
  │     ├── reset() → receives OrigamiObservation
  │     ├── step(action) → receives OrigamiObservation
  │     └── Parses: paper_state, metrics, fold_history, render_urls
  │
  ├── CreasePattern.tsx
  │     └── Reads: paper_state.vertices_coords[:, :2], edges_vertices, edges_assignment
  │         Renders: SVG <line> elements with color/dash per assignment
  │
  ├── FoldedView3D.tsx
  │     └── Reads: paper_state.vertices_coords[:, :3], faces_vertices, strain_per_vertex
  │         Renders: R3F <mesh> with BufferGeometry
  │           positions → Float32Array from vertices_coords
  │           index → Uint16Array from triangulated faces_vertices
  │           colors → Float32Array from Lut(strain_per_vertex) blue→red
  │
  ├── FoldAnimation.tsx
  │     └── Reads: fold_history, paper_state per step
  │         Controls: step slider, play/pause, speed
  │         Interpolates fold_percent 0→1 via useFrame
  │
  ├── MetricsDashboard.tsx
  │     └── Reads: metrics dict → renders cards
  │
  └── CaptureControls.tsx
        ├── Screenshot: gl.domElement.toDataURL('image/png') (preserveDrawingBuffer=true)
        └── Record: canvas.captureStream(30) + MediaRecorder → WebM blob → download
```

### Key R3F Pattern

```tsx
// FoldedView3D.tsx — core rendering
<Canvas gl={{ preserveDrawingBuffer: true, antialias: true }}>
  <PerspectiveCamera makeDefault position={[0, 2, 3]} />
  <OrbitControls />
  <ambientLight intensity={0.4} />
  <directionalLight position={[5, 5, 5]} intensity={0.8} />

  <mesh>
    <bufferGeometry>
      <bufferAttribute attach="attributes-position"
        array={positionsFloat32} count={vertexCount} itemSize={3} />
      <bufferAttribute attach="attributes-color"
        array={strainColorsFloat32} count={vertexCount} itemSize={3} />
      <bufferAttribute attach="index"
        array={indicesUint16} count={indexCount} itemSize={1} />
    </bufferGeometry>
    <meshStandardMaterial vertexColors side={DoubleSide} />
  </mesh>

  {/* Crease lines in 3D */}
  {edges.map((edge, i) => (
    <Line key={i}
      points={[vertices[edge[0]], vertices[edge[1]]]}
      color={assignmentColor(assignments[i])}
      lineWidth={assignments[i] === 'B' ? 2 : 1}
      dashed={assignments[i] === 'M' || assignments[i] === 'V'}
    />
  ))}
</Canvas>
```

---

## 9. Client (`client/`)

### `client.py`

```python
class OrigamiEnvClient(EnvClient[OrigamiAction, OrigamiObservation, OrigamiState]):

    def _step_payload(self, action: OrigamiAction) -> Dict:
        return {
            "fold_type": action.fold_type,
            "fold_line": action.fold_line,
            "fold_angle": action.fold_angle,
            "layer_select": action.layer_select,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OrigamiObservation]:
        obs = OrigamiObservation(**payload.get("observation", payload))
        return StepResult(observation=obs, reward=obs.reward)

    def _parse_state(self, payload: Dict) -> OrigamiState:
        return OrigamiState(**payload)
```

### `reward_functions.py`

Three reward functions for GRPO training. These run on the Colab client side, NOT on the server.

The strategy function extracted from LLM output calls `step()` in a loop (same pattern as 2048):

```python
def _execute_strategy(strategy_fn, openenv_process):
    """
    Execute a fold_strategy against the environment.
    strategy_fn takes paper_state dict, returns one fold dict or None (stop).
    Loops until done or strategy returns None.
    """
    result = openenv_process.reset()
    obs = result.observation

    while not obs.done:
        paper_state = obs.paper_state
        fold = strategy_fn(paper_state)

        if fold is None:
            # Strategy says stop
            action = OrigamiAction(fold_type="stop", fold_line={"start":[0,0],"end":[0,0]})
        else:
            action = OrigamiAction(
                fold_type=fold.get("type", "valley"),
                fold_line=fold.get("line", {"start":[0,0.5],"end":[1,0.5]}),
                fold_angle=fold.get("angle", 180),
            )

        result = openenv_process.step(action)
        obs = result.observation

    return obs
```

Reward functions: `code_valid`, `no_cheating`, `fold_quality` — same structure as 2048 (extract function, sandbox, execute, score from metrics).

---

## 10. Task System (`server/tasks.py`)

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

### How Tasks Drive Reward

- **target_ratio** → compactness reward signal
- **target_box** → fits_target_box bonus (+10.0)
- **must_deploy** → deployability bonus (+5.0)
- **material.max_strain** → strain penalty threshold
- **max_folds** → episode length limit
- **target_shape** → shape similarity metrics (chamfer/hausdorff)

---

## 11. API Reference

### Endpoints (provided by OpenEnv + our extensions)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | `{"status": "ok", "env_name": "origami_env"}` |
| `/ws` | WebSocket | Main OpenEnv communication channel |
| `/reset` | POST | Reset environment, get initial observation |
| `/step` | POST | Send action, get observation |
| `/state` | GET | Get current OrigamiState |
| `/renders/{episode_id}/*` | GET | Serve screenshots, GIFs, FOLD exports |
| `/` | GET | React frontend (static) |

### WebSocket Message Format

```json
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
  "observation": {
    "done": false,
    "reward": null,
    "task": {...},
    "paper_state": {...},
    "metrics": {...},
    "fold_history": [...],
    "error": null,
    "render_urls": {
      "crease_2d": "/renders/ep_abc123/crease_step_1.png",
      "folded_3d": "/renders/ep_abc123/folded_step_1.png",
      "strain_heatmap": "/renders/ep_abc123/strain_step_1.png"
    }
  }
}
```

---

## 12. Deployment

### Push to HF Spaces

```bash
cd origami_env/
openenv push --repo-id <username>/origami-env
```

### Or manually via Docker

```bash
# Build
docker build -t origami-env -f server/Dockerfile .

# Run locally
docker run -p 8000:8000 origami-env

# Test
curl http://localhost:8000/health
```

### HF Space README header

```yaml
---
title: Origami RL Environment
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: true
---
```

---

## 13. Testing Checklist

```bash
# 1. Engine standalone
python -c "
from server.engine.paper import create_flat_sheet
from server.engine.materials import MATERIALS
p = create_flat_sheet(1.0, 1.0, MATERIALS['paper'])
print(f'Vertices: {p.vertices_coords.shape}, Edges: {p.edges_vertices.shape}')
"

# 2. Fold works
python -c "
from server.engine.paper import create_flat_sheet
from server.engine.fold import apply_fold
from server.engine.materials import MATERIALS
p = create_flat_sheet(1.0, 1.0, MATERIALS['paper'])
p = apply_fold(p, {'type':'valley','line':{'start':[0,0.5],'end':[1,0.5]},'angle':180})
print(f'After fold: {p.vertices_coords.shape[0]} vertices, fold_count={p.fold_count}')
"

# 3. Physics runs
python -c "
from server.engine.paper import create_flat_sheet
from server.engine.fold import apply_fold
from server.engine.physics import simulate
from server.engine.materials import MATERIALS
p = create_flat_sheet(1.0, 1.0, MATERIALS['paper'])
p = apply_fold(p, {'type':'valley','line':{'start':[0,0.5],'end':[1,0.5]},'angle':180})
p = simulate(p)
print(f'Max strain: {p.strain_per_vertex.max():.6f}, Energy: {p.energy}')
"

# 4. Validation works
python -c "
from server.engine.validation import validate_state
# ... create + fold paper ...
report = validate_state(p)
print(f'Valid: {report[\"is_valid\"]}, Kawasaki: {report[\"kawasaki_violations\"]}')
"

# 5. Metrics computed
python -c "
from server.engine.metrics import compute_all_metrics
# ... create + fold + validate paper ...
m = compute_all_metrics(p, task, validation)
print(f'Compactness: {m[\"compactness\"]:.3f}, Fits box: {m[\"fits_target_box\"]}')
"

# 6. Renderer works
python -c "
from server.renderer.render_2d import render_crease_pattern
from server.renderer.render_3d import render_folded_state
# ... create + fold paper ...
render_crease_pattern(p, 'test_crease.png')
render_folded_state(p, 'test_folded.png')
print('Renders saved')
"

# 7. Server starts
cd server && uvicorn app:app --port 8000 &
curl http://localhost:8000/health

# 8. Reset + step via API
python -c "
from client.client import OrigamiEnvClient
c = OrigamiEnvClient('ws://localhost:8000')
obs = c.reset()
print(f'Task: {obs.task[\"name\"]}, Sheet: {obs.paper_state[\"num_vertices\"]} vertices')
"

# 9. React frontend builds
cd web && npm install && npm run build
# Check web/dist/index.html exists

# 10. Docker builds and runs
docker build -t origami-env -f server/Dockerfile .
docker run -p 8000:8000 origami-env
curl http://localhost:8000/health
# Open http://localhost:8000 in browser — see React UI
```
