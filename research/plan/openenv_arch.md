# Origami RL — Architecture Plan

> **Paradigm:** LLM generates FOLD crease patterns → physics simulates folding →
> reward = shape similarity to target. Like AlphaFold but for origami.
>
> Viewer harvested from OrigamiSimulator. Minimalistic grid UI.

---

## 1. Core Concept

### The AlphaFold Analogy

| AlphaFold | Our System |
|---|---|
| Input: amino acid sequence | Input: "fold into a triangle" (task prompt) |
| Predict: 3D protein structure | Predict: crease pattern (FOLD format) |
| Simulate: molecular dynamics | Simulate: bar-and-hinge physics (creasePercent 0→1) |
| Validate: compare to known structure | Validate: overlay folded shape vs target shape |
| Reward: structural similarity (RMSD) | Reward: shape similarity (chamfer distance) |

### What the LLM Actually Generates

**NOT** step-by-step fold actions. **A FOLD crease pattern.**

```json
{
  "vertices_coords": [[0,0], [1,0], [1,1], [0,1], [0.5,0.5]],
  "edges_vertices": [[0,1], [1,2], [2,3], [3,0], [0,2]],
  "edges_assignment": ["B", "B", "B", "B", "V"],
  "edges_foldAngle": [0, 0, 0, 0, 180]
}
```

The LLM outputs a complete crease pattern — vertices, edges, assignments
(mountain/valley/boundary), and target fold angles. The physics engine takes
this, animates creasePercent from 0→1, and produces a final 3D shape.
We compare that shape against the target. That's the reward.

**Why this is correct:**
- OrigamiSimulator works exactly this way — fixed topology, animate creasePercent
- No topology modification headaches (no face splitting, no vertex insertion)
- FOLD is a well-defined format — LLMs can generate structured JSON
- Physics is solved (bar-and-hinge is well-understood)
- Reward is dead simple: does the shape match?

### Two Contexts, Same Engine

```
CONTEXT 1: RL TRAINING (Colab/GPU)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  LLM generates FOLD JSON (crease pattern)               │
│       ↓                                                 │
│  Physics engine: simulate(fold_data, creasePercent=1.0) │
│       ↓                                                 │
│  Final 3D shape (vertex positions)                      │
│       ↓                                                 │
│  Compare to target shape → reward (shape similarity)    │
│                                                         │
│  NO RENDERING. Just geometry → numbers → reward.        │
│                                                         │
└─────────────────────────────────────────────────────────┘

CONTEXT 2: DEMO (Web App + Browser)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Grid homepage: cards showing fold experiments          │
│       ↓ click                                           │
│  Detail page: OrigamiSimulator-style viewer             │
│                                                         │
│  ┌─────────────┐  ┌──────────────────────────────────┐  │
│  │ 2D Crease   │  │ 3D Folded Mesh (Three.js)        │  │
│  │ Pattern     │  │ + target shape overlay (ghost)    │  │
│  │ (SVG)       │  │ + orbit controls                 │  │
│  │ M=red V=blue│  │ + strain heatmap                 │  │
│  └─────────────┘  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐   │
│  │  creasePercent slider: flat ◄━━━━━━━━━━► folded  │   │
│  └──────────────────────────────────────────────────┘   │
│  Metrics: shape similarity, fold count, strain          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. How OrigamiSimulator Works (Reference)

We're harvesting from this. Understanding it is critical.

### Folding Paradigm: Fixed Topology + creasePercent Animation

1. Load crease pattern (SVG or FOLD) — all vertices, edges, faces defined upfront
2. All vertices start flat (z=0)
3. `creasePercent` uniform scales ALL target fold angles: `targetTheta = originalTheta × creasePercent`
4. GPU bar-and-hinge solver finds equilibrium 3D positions
5. **Topology NEVER changes** — same vertices, edges, faces throughout

### Physics (Bar-and-Hinge)

Four force types:
- **Beam (axial):** `F = -K(length - L₀)` — prevents stretching
- **Crease (torsional):** `τ = K(targetTheta × creasePercent - θ_current)` — drives folding
- **Face stiffness:** triangle angle preservation — prevents mesh collapse
- **Panel stiffness:** dihedral springs on facet edges — keeps faces flat

Solver: Verlet integration, adaptive timestep `dt = 0.9 / (2π × maxFreq)`,
per-beam critical damping `D = 0.45 × 2√(K × m_min)`.

### FOLD Format (What the LLM Generates)

```json
{
  "file_spec": 1.1,
  "vertices_coords": [[x, y], ...],
  "edges_vertices": [[v1, v2], ...],
  "edges_assignment": ["M", "V", "B", "F", ...],
  "edges_foldAngle": [-180, 180, 0, 0, ...],
  "faces_vertices": [[v0, v1, v2], ...]
}
```

| Assignment | Color | Meaning |
|---|---|---|
| B | black #000 | Boundary edge |
| M | red #f00 | Mountain fold (negative angle) |
| V | blue #00f | Valley fold (positive angle) |
| F | yellow #ff0 | Facet/triangulation edge |
| U | magenta #f0f | Unassigned/hinge |

### SVG Encoding

OrigamiSimulator's SVGs encode crease patterns via stroke color + opacity:
- `stroke="#ff0000" opacity="0.5"` → Mountain at -90° (0.5 × 180)
- `stroke="#0000ff" opacity="1.0"` → Valley at +180°
- `stroke="#000000"` → Boundary

### Coordinate System

OrigamiSimulator: XZ-plane flat (Y=0, Y is up).
We need to decide: match theirs or use XY-plane (Z up). Either way, swap on import/export.

---

## 3. What We Harvest vs Build

### Harvest from OrigamiSimulator

| Component | Source File | What We Take |
|---|---|---|
| 3D mesh rendering | `threeView.js` | Three.js scene setup, lighting, BufferGeometry from FOLD |
| Crease edge visualization | `model.js` | Color-coded line geometry (M=red, V=blue, B=black) |
| creasePercent animation | `dynamicSolver.js` | The concept + slider → uniform → physics loop |
| FOLD file parsing | `pattern.js` | `processFold()` — FOLD JSON → internal representation |
| Triangulation | `earcut.js` | Polygon → triangle fan for Three.js index buffer |
| Camera controls | TrackballControls | Orbit, zoom (already Three.js standard) |

### Strip Out (Don't Need)

- jQuery / jQuery UI → modern framework
- VR support (`VRInterface.js`, `datguivr.js`)
- Curved folding (`curvedFolding.js` — 2882 lines of complexity)
- GPU solver (`dynamicSolver.js` + `GPUMath.js`) — we run physics server-side for training; for the viewer we can either port a simplified CPU solver to JS or send pre-computed frames
- Static/rigid solvers (in development, not used)
- File import modals and drag-drop UI
- CCapture video recording
- SVG import pipeline (we work with FOLD directly)

### Build New

| Component | Tech | Purpose |
|---|---|---|
| Grid homepage | Next.js or plain HTML | Minimalistic card grid of experiments |
| Detail page | Next.js or plain HTML | Full simulator view (2D + 3D + slider) |
| Target shape overlay | Three.js | Ghost wireframe of goal shape in 3D view |
| Shape similarity display | UI | Visual score: how close to target |
| Server physics engine | Python (numpy/scipy) | Bar-and-hinge solver for RL reward computation |
| FOLD generation prompt | Python | LLM prompt engineering for crease pattern output |
| Reward function | Python | Shape similarity (chamfer distance to target) |

---

## 4. Frontend: Minimalistic Grid → Simulator

### Homepage (Grid)

```
┌──────────────────────────────────────────────────────────┐
│  ORIGAMI RL                                    [about]   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  ┌────┐  │  │  ┌────┐  │  │  ┌────┐  │  │  ┌────┐  │ │
│  │  │ 3D │  │  │  │ 3D │  │  │  │ 3D │  │  │  │ 3D │  │ │
│  │  └────┘  │  │  └────┘  │  │  └────┘  │  │  └────┘  │ │
│  │ Triangle │  │ Half Fold│  │ Map Fold │  │ Miura-ori│ │
│  │ 92% ██▓░ │  │ 87% ██▓░ │  │ 45% ██░░ │  │ 12% █░░░ │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  ┌────┐  │  │  ┌────┐  │  │  ┌────┐  │               │
│  │  │ 3D │  │  │  │ 3D │  │  │  │ 3D │  │               │
│  │  └────┘  │  │  └────┘  │  │  └────┘  │               │
│  │ Crane    │  │ Stent    │  │ Solar    │               │
│  │ 0% ░░░░  │  │ 0% ░░░░  │  │ 0% ░░░░  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└──────────────────────────────────────────────────────────┘
```

Each card:
- Mini Three.js preview (static or slowly rotating folded state)
- Task name
- Shape similarity score (progress bar)
- Click → navigates to detail page

### Detail Page (Full Simulator)

```
┌──────────────────────────────────────────────────────────┐
│  ← Back                              Triangle Fold       │
│                                                          │
│  ┌────────────────────┐  ┌────────────────────────────┐  │
│  │                    │  │                            │  │
│  │   2D Crease        │  │   3D Folded Mesh           │  │
│  │   Pattern          │  │                            │  │
│  │                    │  │   [solid mesh]              │  │
│  │   M ── red         │  │   [ghost target overlay]   │  │
│  │   V ── blue        │  │                            │  │
│  │   B ── black       │  │   orbit / zoom / pan       │  │
│  │                    │  │                            │  │
│  └────────────────────┘  └────────────────────────────┘  │
│                                                          │
│  flat ◄━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━► folded    │
│                    creasePercent                          │
│                                                          │
│  Shape Match: 92%    Folds: 1    Strain: 0.02            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Key elements:
- 2D crease pattern (left) — SVG or Canvas, color-coded edges
- 3D mesh (right) — Three.js with target shape ghost overlay
- creasePercent slider — scrub fold/unfold animation
- Target overlay — wireframe or transparent mesh showing the goal shape
- Metrics bar — shape similarity, fold count, max strain

### Design Principles

- **Minimalistic.** White/light background. No clutter. No unnecessary chrome.
- **No framework bloat.** Next.js for routing (grid → detail), or even plain HTML with dynamic pages.
- **Harvested renderer.** Three.js rendering logic from OrigamiSimulator, stripped to essentials.
- **Responsive grid.** CSS Grid `auto-fit, minmax(280px, 1fr)`.

---

## 5. RL Training: How It Works

### Flow

```
1. Define target shape (e.g., triangle — a known FOLD file with final positions)

2. LLM prompt:
   "Generate a FOLD crease pattern that, when folded, produces a triangle.
    Output JSON with vertices_coords, edges_vertices, edges_assignment,
    edges_foldAngle."

3. LLM generates FOLD JSON

4. Physics engine simulates:
   - Load FOLD data
   - Run bar-and-hinge solver with creasePercent=1.0
   - Get final 3D vertex positions

5. Reward = shape_similarity(final_positions, target_positions)

6. GRPO trains on this reward signal
```

### The Prompt

```
You are an origami designer. Given a target shape description,
generate a FOLD-format crease pattern that folds into that shape.

Target: {task_description}
Paper size: {width} x {height}

Output a JSON object with:
- vertices_coords: [[x, y], ...] — 2D vertex positions on flat paper
- edges_vertices: [[v1, v2], ...] — edge connectivity
- edges_assignment: ["B"|"M"|"V", ...] — B=boundary, M=mountain, V=valley
- edges_foldAngle: [angle, ...] — target fold angles in degrees
  (M: negative, V: positive, B: 0)

Rules:
- Boundary edges must form the paper outline
- All fold lines must connect to vertices
- Mountain angles are negative (-180 to 0)
- Valley angles are positive (0 to 180)

Output ONLY the JSON object.
```

### Reward Function (Simple)

```python
def shape_similarity(completions, target_positions, **kwargs):
    """
    AlphaFold-style reward: how close is the folded shape to the target?

    1. Parse LLM output as FOLD JSON
    2. Run physics simulation (creasePercent=1.0)
    3. Compare final vertex positions to target
    4. Score = similarity metric
    """
    scores = []
    for completion in completions:
        fold_data = parse_fold_json(completion)
        if fold_data is None:
            scores.append(-1.0)  # invalid JSON
            continue

        try:
            final_positions = simulate(fold_data, crease_percent=1.0)
            score = compute_shape_match(final_positions, target_positions)
            # score in [0, 1]: 0 = no match, 1 = perfect match
            scores.append(score * 20.0)  # scale for GRPO
        except SimulationError:
            scores.append(-2.0)  # unstable fold pattern

    return scores


def compute_shape_match(predicted, target):
    """
    Chamfer distance between predicted and target vertex clouds.
    Normalized to [0, 1] where 1 = perfect match.
    """
    from scipy.spatial.distance import cdist
    d = cdist(predicted, target)
    chamfer = (d.min(axis=1).mean() + d.min(axis=0).mean()) / 2
    # Normalize: 0 distance = 1.0 score, large distance = 0.0
    return max(0, 1.0 - chamfer / diagonal_length)
```

That's it. One reward function. Shape match. No "code_valid", no "no_cheating",
no "fold_quality" with 6 different score branches. Just: does the shape match?

### Starting Simple: Triangle Fold

**Target:** Paper folded in half diagonally → triangle.

**Known solution FOLD data:**
```json
{
  "vertices_coords": [[0,0], [1,0], [1,1], [0,1]],
  "edges_vertices": [[0,1], [1,2], [2,3], [3,0], [0,2]],
  "edges_assignment": ["B", "B", "B", "B", "V"],
  "edges_foldAngle": [0, 0, 0, 0, 180]
}
```

One valley fold along the diagonal. creasePercent=1.0 folds it into a triangle.
The LLM needs to discover this crease pattern. That's the training signal.

**Progression:**
1. Triangle (1 fold) — learn the format, learn what a valley fold does
2. Half fold (1 fold) — horizontal/vertical line
3. Quarter fold (2 folds) — two perpendicular folds
4. Letter fold (2 parallel folds) — tri-fold
5. More complex patterns as the model improves

---

## 6. Physics Engine (Python, Server-Side)

For RL training, we need a Python physics engine that takes FOLD data and
produces final 3D positions. This runs headless — no rendering.

### What It Does

```python
def simulate(fold_data: dict, crease_percent: float = 1.0) -> np.ndarray:
    """
    Take FOLD crease pattern, run bar-and-hinge physics,
    return final 3D vertex positions.

    Args:
        fold_data: FOLD format dict (vertices, edges, assignments, angles)
        crease_percent: 0.0 = flat, 1.0 = fully folded

    Returns:
        positions: (N, 3) array of final vertex positions
    """
    # 1. Build nodes, beams, creases from FOLD data
    # 2. Set target angles: targetTheta = foldAngle × crease_percent
    # 3. Run Verlet integration until convergence
    # 4. Return final positions
```

### Forces (matching OrigamiSimulator)

| Force | Formula | Purpose |
|---|---|---|
| Beam (axial) | `F = -(K/L)(len - L₀) × direction` | Prevent stretching |
| Crease (torsional) | `τ = K × L × (θ_target × cp - θ_current)` | Drive folding |
| Face stiffness | `F = K_face × (angle - angle₀)` per triangle angle | Prevent collapse |
| Damping | `D = 0.45 × 2√(K × m_min)` per beam | Remove oscillation |

### Parameters (matching OrigamiSimulator defaults)

```python
DEFAULTS = {
    "axial_stiffness": 20,
    "crease_stiffness": 0.7,
    "panel_stiffness": 0.7,
    "face_stiffness": 0.2,
    "damping_ratio": 0.45,
    "density": 1.0,
}
```

### Convergence

Run solver until energy delta < threshold or max iterations reached.
For simple folds (triangle, half fold), convergence is fast (~200 steps).

---

## 7. Target Shapes: How We Define Them

Each task has a **target**: the known-good folded shape.

### Option A: Reference FOLD File (Preferred)

We have the solved FOLD file. We simulate it ourselves to get target positions.
The LLM's job is to discover the same (or equivalent) crease pattern.

```python
TASKS = {
    "triangle": {
        "description": "Fold the paper into a triangle",
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [[0,0], [1,0], [1,1], [0,1]],
            "edges_vertices": [[0,1], [1,2], [2,3], [3,0], [0,2]],
            "edges_assignment": ["B", "B", "B", "B", "V"],
            "edges_foldAngle": [0, 0, 0, 0, 180],
        },
        "max_vertices": 10,
        "max_edges": 15,
    },
}
```

### Option B: 3D Point Cloud / Mesh

For complex shapes where multiple crease patterns could work,
define the target as a 3D point cloud or mesh silhouette.
Reward = how close the folded shape's silhouette/volume matches.

### Viewer: Target Shape Overlay

In the detail page, the target shape renders as a **ghost wireframe** (transparent,
dashed edges) overlaid on the 3D view. The LLM's folded result renders solid
on top. You can visually see how close they match — like AlphaFold's predicted
vs actual structure overlay.

---

## 8. Iterative FOLD Generation (Advanced)

The LLM can also build the crease pattern step-by-step:

```
Step 1: Start with boundary (4 vertices, 4 edges, all "B")
Step 2: Add vertex at (0.5, 0.5), add edge [0, 4] as "V" at 180°
Step 3: Add edge [4, 2] as "V" at 180° → completes diagonal crease
Step 4: "done"
```

Each step adds to the FOLD data. The physics runs after the final step.
This gives more training signal (partial rewards for partial progress)
and mirrors how a human designer thinks — adding creases one at a time.

**But we start with single-shot generation.** Iterative is Phase 2.

---

## 9. Repository Structure

```
origami/
├── app/                                # Frontend (Next.js or plain HTML)
│   ├── page.tsx                        # Grid homepage
│   ├── fold/[id]/page.tsx              # Detail page (simulator view)
│   ├── components/
│   │   ├── FoldCard.tsx                # Grid card (mini 3D preview)
│   │   ├── CreasePattern2D.tsx         # 2D crease pattern (SVG/Canvas)
│   │   ├── FoldViewer3D.tsx            # 3D mesh viewer (Three.js)
│   │   ├── TargetOverlay.tsx           # Ghost wireframe target shape
│   │   ├── CreaseSlider.tsx            # creasePercent slider
│   │   └── MetricsBar.tsx              # Shape match, folds, strain
│   └── lib/
│       ├── fold-parser.ts              # FOLD JSON → Three.js geometry
│       ├── physics-client.ts           # Optional: browser-side physics
│       └── three-setup.ts             # Scene, camera, lights, controls
│
├── server/                             # Python backend
│   ├── engine/
│   │   ├── simulate.py                 # Bar-and-hinge solver (CPU, numpy)
│   │   ├── fold_parser.py              # FOLD JSON validation + loading
│   │   ├── shape_match.py              # Chamfer distance, shape similarity
│   │   └── materials.py                # Stiffness parameters
│   ├── environment.py                  # OpenEnv Environment subclass
│   ├── models.py                       # Action (FOLD JSON), Observation, State
│   ├── tasks.py                        # Task definitions + target shapes
│   ├── app.py                          # FastAPI server
│   └── Dockerfile
│
├── training/                           # GRPO training (Colab)
│   ├── train_grpo.py                   # GRPOTrainer setup
│   └── reward.py                       # shape_similarity reward function
│
├── assets/                             # Reference patterns (from OrigamiSimulator)
│   ├── targets/                        # Target FOLD files for each task
│   │   ├── triangle.fold
│   │   ├── half_fold.fold
│   │   └── ...
│   └── examples/                       # OrigamiSimulator demo patterns
│
└── research/
    └── plan/
        └── openenv_arch.md             # This file
```

### What's NOT Here Anymore

| Removed | Why |
|---|---|
| `renderer/` (render_2d, render_3d, screenshots, recorder) | Browser renders via Three.js |
| `client/reward_functions.py` (code_valid, no_cheating, fold_quality) | Single reward: shape_similarity |
| `engine/fold.py` (topology-modifying apply_fold) | No topology changes — fixed crease pattern |
| `engine/validation.py` (Kawasaki, Maekawa) | Not needed when LLM generates full pattern |
| `engine/metrics.py` (20+ metrics) | One metric: shape similarity |
| `training/runner.py` (parallel episode executor) | Simpler: just simulate + score |
| `viewer/training.html` (training grid) | Grid is the homepage now |
| `training_broadcast.py` (websocket broadcast) | Not needed for single-shot eval |
| matplotlib, Pillow, imageio deps | Never needed |

---

## 10. OpenEnv Integration

### Models

```python
class OrigamiAction(Action):
    """LLM submits a FOLD crease pattern."""
    fold_data: Dict[str, Any]    # FOLD format JSON
    # {
    #   "vertices_coords": [[x,y], ...],
    #   "edges_vertices": [[v1,v2], ...],
    #   "edges_assignment": ["B","M","V",...],
    #   "edges_foldAngle": [0, -180, 180, ...],
    # }

class OrigamiObservation(Observation):
    """Result of simulating the crease pattern."""
    task: Dict[str, Any]                    # Task description + target info
    fold_data: Dict[str, Any]               # The submitted crease pattern
    final_positions: List[List[float]]      # (N,3) after physics simulation
    target_positions: List[List[float]]     # (N,3) target shape
    shape_similarity: float                 # 0.0 to 1.0
    strain: float                           # Max strain in simulation
    is_stable: bool                         # Did simulation converge?
    error: Optional[str] = None

class OrigamiState(State):
    task_name: str = ""
    shape_similarity: float = 0.0
    is_stable: bool = True
```

### Environment

```python
class OrigamiEnvironment(Environment):

    def reset(self, task_name=None, **kwargs):
        self._task = get_task(task_name)
        # Simulate target to get target_positions
        self._target_positions = simulate(
            self._task["target_fold"], crease_percent=1.0
        )
        return self._make_observation(fold_data=None, done=False)

    def step(self, action: OrigamiAction, **kwargs):
        fold_data = action.fold_data

        # Simulate the LLM's crease pattern
        try:
            final_positions = simulate(fold_data, crease_percent=1.0)
            similarity = compute_shape_match(final_positions, self._target_positions)
            reward = similarity * 20.0
            is_stable = True
        except SimulationError as e:
            final_positions = []
            similarity = 0.0
            reward = -2.0
            is_stable = False

        return OrigamiObservation(
            done=True,  # single-shot: one action = one episode
            reward=reward,
            task=self._task,
            fold_data=fold_data,
            final_positions=final_positions,
            target_positions=self._target_positions,
            shape_similarity=similarity,
            strain=max_strain,
            is_stable=is_stable,
        )
```

**Key difference:** Each episode is ONE step. LLM submits crease pattern → simulate → score → done. No multi-step loop.

---

## 11. Open Decisions

### Physics: Server vs Browser

| | Server (Python) | Browser (JS) |
|---|---|---|
| **For training** | Required. Runs headless on GPU machine. | N/A |
| **For viewer** | Option: pre-compute frames, send positions per creasePercent | Option: port solver to JS, animate locally |
| **Latency** | HTTP round-trip per slider position | Instant (local compute) |
| **Complexity** | Simpler viewer (just renders positions) | More complex viewer (runs physics) |

**Recommendation:** Server for training. Browser for viewer (harvest OrigamiSimulator's solver approach, simplified CPU version in JS). The viewer physics doesn't need to be GPU-accelerated — our meshes are tiny (4-50 vertices).

### Frontend Framework

| Option | Pros | Cons |
|---|---|---|
| Next.js | Routing, SSR, ecosystem | Heavier setup |
| Plain HTML + Vite | Minimal, fast, no framework tax | Manual routing |
| Astro | Static-first, islands architecture | Less common |

For minimalistic design with grid→detail routing, even plain HTML with hash routing works. Next.js if we want it to feel like a proper app.

### LLM Output: JSON vs SVG

LLM could generate FOLD JSON directly, or SVG (which we parse into FOLD).
JSON is more structured and easier to validate. SVG is more visual and
the LLM might have better training data for SVG generation.

**Start with JSON.** It's unambiguous.

---

## 12. Phase Plan

### Phase 1: Triangle Fold (MVP)

- [ ] Python physics engine (bar-and-hinge, numpy)
- [ ] Shape similarity reward function (chamfer distance)
- [ ] Triangle target definition (reference FOLD file)
- [ ] LLM prompt for FOLD generation
- [ ] Basic GRPO training loop on Colab
- [ ] Verify: LLM can discover the diagonal valley fold

### Phase 2: Viewer

- [ ] Three.js viewer harvested from OrigamiSimulator
- [ ] 2D crease pattern rendering (SVG, color-coded)
- [ ] 3D mesh rendering (BufferGeometry from FOLD)
- [ ] creasePercent slider with browser-side physics
- [ ] Target shape ghost overlay
- [ ] Grid homepage with cards
- [ ] Detail page with full simulator view

### Phase 3: More Tasks

- [ ] Half fold, quarter fold, letter fold targets
- [ ] Task progression / curriculum
- [ ] Iterative FOLD generation (step-by-step crease pattern building)

### Phase 4: Scale

- [ ] Complex targets (crane base, miura-ori)
- [ ] Multiple materials (different stiffness → different physics)
- [ ] Dataset of known origami patterns as targets
- [ ] Deploy to HF Spaces

---

## 13. Reference: OrigamiSimulator Files We'll Use

```
temp/OrigamiSimulator/
├── js/
│   ├── threeView.js          → Three.js scene setup, lighting, camera
│   ├── model.js              → FOLD → BufferGeometry conversion
│   ├── pattern.js            → FOLD parsing (processFold function)
│   ├── node.js               → Vertex representation
│   ├── beam.js               → Edge constraint
│   ├── crease.js             → Fold crease constraint
│   └── controls.js           → Slider binding (creasePercent)
├── dependencies/
│   ├── three.min.js          → Three.js library
│   ├── fold.js               → FOLD format API
│   ├── earcut.js             → Polygon triangulation
│   └── TrackballControls.js  → Camera controls
└── assets/
    ├── Origami/              → Traditional patterns (targets)
    ├── Bases/                → Base folds (targets)
    ├── SimpleFolds/          → Simple patterns (targets)
    └── Tessellations/        → Tessellation patterns (targets)
```

---

## 14. Key Insight: Why This Works for RL

The old plan had the LLM generating fold actions (type, line, angle) step by step,
modifying topology each time. This is hard because:
- Topology changes are fragile (face splitting, vertex insertion)
- Multi-step episodes have sparse reward (only at the end)
- The action space is continuous and poorly defined
- Physics after topology change is unstable

The new plan has the LLM generating a complete crease pattern. This is better because:
- **Fixed topology** — physics is stable and well-understood
- **Single-shot** — dense reward (immediate score after one submission)
- **Structured output** — FOLD JSON is well-defined, validatable
- **Proven physics** — OrigamiSimulator's bar-and-hinge works on these exact patterns
- **Clear target** — shape match is simple, visual, and differentiable
- **Like AlphaFold** — predict structure, simulate, compare to known truth
