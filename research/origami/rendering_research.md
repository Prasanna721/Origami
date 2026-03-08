# Origami Rendering & Visualization Research

## 1. Existing Open-Source Origami Renderers

### 1.1 Amanda Ghassaei's Origami Simulator (Best-in-Class)

- **GitHub**: https://github.com/amandaghassaei/OrigamiSimulator
- **Live demo**: https://origamisimulator.org/
- **License**: MIT
- **Tech stack**: Three.js (rendering), GPU fragment shaders (simulation), FOLD format (data)
- **Paper**: "Fast, Interactive Origami Simulation using GPU Computation" (7OSME)

**How the simulation works:**
- Folds all creases simultaneously (not sequential folding)
- Iteratively solves for small displacements due to forces from creases
- Three constraint types: distance (no stretch/compress), face (no shear), angular (fold/flatten)
- Each constraint weighted by stiffness parameter
- All computation in GPU fragment shaders for real-time performance
- Target fold angle per crease: [-180, 180] degrees (positive=valley, negative=mountain)

**Strain visualization:**
- Maps strain to vertex colors: blue (no strain) to red (max strain)
- Strain = average percent deviation of distance constraints (Cauchy/engineering strain)

**I/O formats**: SVG, FOLD input; FOLD, STL, OBJ export

**Embedding in React:**
- Not published as an npm package; no React component wrapper exists
- Would need to fork and extract the core simulation + Three.js rendering
- The Three.js scene setup is tightly coupled to the app's DOM/UI
- **Best approach**: extract the GPU solver + mesh rendering and wrap in R3F

**External libraries used:**
- svgpath + path-data-polyfill (SVG parsing)
- FOLD library (internal data structure)
- Earcut (triangulation)
- Three.js (rendering)

### 1.2 OrigamiOdyssey (React + R3F -- Closest to What We Need)

- **GitHub**: https://github.com/robbwdoering/origamiodyssey
- **Tech stack**: React SPA, Three.js via react-three-fiber, hierarchical instructions
- **What it does**: Teaches origami via 3D simulations, step-through instructions, free rotation
- **Key relevance**: Proves that origami can be rendered well with R3F
- Already uses react-three-fiber to render a mesh representing paper
- Supports step-through fold sequences with 3D rotation

### 1.3 georgiee/origami (Three.js Origami Builder)

- **GitHub**: https://github.com/georgiee/origami
- **Demo**: https://georgiee.github.io/origami/
- **What it does**: Runtime origami folding rendered in Three.js
- Has an editor mode and view mode
- Not React-based, but demonstrates the mesh-folding-along-crease approach

### 1.4 flat-folder (Jason S. Ku -- Computation Tool)

- **GitHub**: https://github.com/origamimagiro/flat-folder
- **What it does**: Computes and analyzes valid flat-foldable states
- Renders three simultaneous views: crease pattern, x-ray overlap, folded state
- Supports FOLD, SVG, OPX, CP input formats
- Highlights Maekawa/Kawasaki violations with red circles
- **Not a library** -- standalone web app, not embeddable
- Useful as a reference for rendering conventions

### 1.5 Crease Pattern Editor (mwalczyk/crease)

- **GitHub**: https://github.com/mwalczyk/crease
- Vector-based SVG line drawing tool for crease patterns
- Geometric operations for diagramming
- Returns lists of changed nodes/edges for SVG element updates

### 1.6 Other Notable Projects

| Project | URL | Notes |
|---------|-----|-------|
| OriDomi | http://oridomi.com/ | CSS/DOM paper folding effects, not 3D simulation |
| paperfold.js | https://github.com/mrflix/paperfold | CSS3 transitions for paper fold UI effects |
| Origami3D | https://github.com/mariusGundersen/Origami3D | Simple 3D engine using 2D canvas, minimal |
| CurvedCreases | https://github.com/amandaghassaei/CurvedCreases | Amanda Ghassaei's curved crease simulator |

---

## 2. Three.js in React

### 2.1 @react-three/fiber (R3F)

- **GitHub**: https://github.com/pmndrs/react-three-fiber
- **Docs**: https://r3f.docs.pmnd.rs/
- The standard way to use Three.js in React
- Declarative scene graph via JSX
- `useFrame` hook for per-frame updates (animation, simulation stepping)
- `useThree` hook for accessing renderer, scene, camera
- Full Three.js API available through refs

**Key pattern for origami:**
```jsx
<Canvas gl={{ preserveDrawingBuffer: true }}>
  <PerspectiveCamera makeDefault position={[0, 5, 10]} />
  <OrbitControls />
  <mesh ref={paperMeshRef}>
    <bufferGeometry>
      <bufferAttribute attach="attributes-position" ... />
      <bufferAttribute attach="attributes-color" ... />
    </bufferGeometry>
    <meshStandardMaterial vertexColors side={DoubleSide} />
  </mesh>
</Canvas>
```

### 2.2 @react-three/drei

- **GitHub**: https://github.com/pmndrs/drei
- **Docs**: https://drei.docs.pmnd.rs/
- Provides: `OrbitControls`, `PerspectiveCamera`, `Html`, `Line`, `Text`, `Grid`, `Environment`
- `OrbitControls` -- rotate, zoom, pan the 3D view (essential for our rotatable viewer)
- `Line` / `meshline` -- for rendering crease lines in 3D

### 2.3 Rendering a Foldable Mesh

**Approach: Vertex manipulation on BufferGeometry**

The paper mesh is a `BufferGeometry` with a position attribute. To simulate folding:

1. Define crease lines as edges in the mesh
2. For each fold operation, identify vertices on one side of the crease
3. Apply a rotation (quaternion) to those vertices around the crease line axis
4. Update the position buffer: `geometry.attributes.position.needsUpdate = true`

**Implementation pattern:**
```javascript
// For a fold along a crease line from point A to point B:
const axis = new THREE.Vector3().subVectors(B, A).normalize();
const quaternion = new THREE.Quaternion().setFromAxisAngle(axis, foldAngle);

for (let i = 0; i < vertexCount; i++) {
  if (isOnFoldSide(vertices[i], creaseLine)) {
    const v = new THREE.Vector3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    v.sub(A);  // translate to crease origin
    v.applyQuaternion(quaternion);  // rotate
    v.add(A);  // translate back
    positions[i*3] = v.x;
    positions[i*3+1] = v.y;
    positions[i*3+2] = v.z;
  }
}
geometry.attributes.position.needsUpdate = true;
```

**For simultaneous folding (Ghassaei approach):**
- Use a spring/constraint solver running per-frame in `useFrame`
- Each crease exerts angular force toward its target fold angle
- Distance constraints prevent stretch, face constraints prevent shear
- This is more physically accurate but computationally heavier
- Could be done in a Web Worker or WASM for performance

### 2.4 Color-Coding Strain with Vertex Colors

**Three.js Lut (Lookup Table) class:**
- Import: `import { Lut } from 'three/addons/math/Lut.js'`
- Built-in colormaps: `rainbow`, `cooltowarm`, `blackbody`, `grayscale`
- For strain: use `cooltowarm` (blue to red) or create custom colormap

**Implementation:**
```javascript
const lut = new Lut('cooltowarm', 512);
lut.setMin(0);    // no strain
lut.setMax(maxStrain);

// For each vertex, compute strain and get color
const colors = new Float32Array(vertexCount * 3);
for (let i = 0; i < vertexCount; i++) {
  const strain = computeVertexStrain(i);
  const color = lut.getColor(strain);
  colors[i * 3] = color.r;
  colors[i * 3 + 1] = color.g;
  colors[i * 3 + 2] = color.b;
}

geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
// Material must have vertexColors: true
```

**In R3F JSX:**
```jsx
<mesh>
  <bufferGeometry>
    <bufferAttribute
      attach="attributes-position"
      array={positions}
      count={vertexCount}
      itemSize={3}
    />
    <bufferAttribute
      attach="attributes-color"
      array={colors}
      count={vertexCount}
      itemSize={3}
    />
  </bufferGeometry>
  <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
</mesh>
```

**Alternative: Shader-based heatmap**
- Custom `ShaderMaterial` with a uniform for strain data
- Map strain to color in fragment shader
- More performant for large meshes, but more complex to implement

---

## 3. Lightweight Rendering Approach for RL

### 3.1 What We Need

| Requirement | Priority | Approach |
|-------------|----------|----------|
| Live view of current fold state | High | Client-side R3F in Gradio |
| Periodic screenshots for training logs | High | `canvas.toDataURL()` or server-side matplotlib |
| Recording folding animation | Medium | CCapture.js / MediaRecorder API |
| Strain heatmap overlay | High | Vertex colors with Lut |
| Crease pattern 2D view | High | SVG (inline or via React component) |

### 3.2 Server-Side Rendering Options

**Option A: matplotlib 3D (Simplest, Recommended for RL Training)**
- `mpl_toolkits.mplot3d` for basic 3D wireframe/surface plots
- Good enough for training logs and periodic state snapshots
- `fig.savefig()` for PNG export
- Works headlessly on any server
- No GPU required

**Option B: Plotly 3D**
- `plotly.graph_objects.Mesh3d` for 3D mesh rendering
- Interactive HTML output embeddable in Gradio
- `fig.write_image()` for static export (requires kaleido)
- Better looking than matplotlib, still no GPU needed

**Option C: Headless Three.js (Node.js)**
- Uses `headless-gl` library for offscreen WebGL
- Full Three.js rendering fidelity
- Significant caveats:
  - Most servers lack GPU, so rendering uses CPU (slow)
  - `headless-gl` has compatibility issues with newer Three.js
  - Requires JSDOM or `three-universal` for DOM mocking
  - Complex setup for questionable benefit
- **Not recommended** unless visual fidelity is critical server-side

**Option D: trimesh + pyrender (Python)**
- `trimesh` for mesh manipulation, `pyrender` for offscreen rendering
- Better 3D quality than matplotlib
- Supports headless via EGL or osmesa
- Good middle ground

**Recommendation: matplotlib for training, Plotly/R3F for demo UI**

### 3.3 Client-Side Rendering (Primary for Demo)

**React + Three.js via R3F:**
- Best visual quality, real-time interactivity
- OrbitControls for rotation, zoom, pan
- Vertex colors for strain heatmap
- Runs in user's browser (GPU-accelerated)

### 3.4 Screenshots from Three.js

**Configuration:**
```jsx
<Canvas gl={{ preserveDrawingBuffer: true, antialias: true, alpha: true }}>
```

**Capture method:**
```javascript
// Inside a component with access to useThree()
const { gl, scene, camera } = useThree();

function captureScreenshot() {
  gl.render(scene, camera);
  const dataURL = gl.domElement.toDataURL('image/png');
  // Send to server or download
  return dataURL;
}
```

**Important:** `toDataURL()` must be called before exiting the current event/frame. The `preserveDrawingBuffer: true` flag is required or you get a black image. Keeping it enabled has minor performance cost.

**High-resolution capture:**
```javascript
// Temporarily increase pixel ratio for hi-res screenshot
const originalPixelRatio = gl.getPixelRatio();
gl.setPixelRatio(window.devicePixelRatio * 2);
gl.setSize(width * 2, height * 2);
gl.render(scene, camera);
const dataURL = gl.domElement.toDataURL('image/png');
gl.setPixelRatio(originalPixelRatio);
gl.setSize(width, height);
```

### 3.5 Recording Folding Animation

**Option A: CCapture.js (Best for frame-perfect capture)**
- **GitHub**: https://github.com/spite/ccapture.js
- Hooks into `requestAnimationFrame`, `Date.now()`, `setTimeout`
- Forces constant time step for stutter-free recording
- Supports WebM, PNG/JPEG tar, GIF
- Usage:
  ```javascript
  const capturer = new CCapture({ format: 'webm', framerate: 30, quality: 95 });
  capturer.start();
  // In render loop:
  capturer.capture(canvas);
  // When done:
  capturer.stop();
  capturer.save(); // triggers download
  ```

**Option B: use-capture (R3F wrapper for CCapture.js)**
- **GitHub**: https://github.com/gsimone/use-capture
- npm: `use-capture`
- Hook-based API for R3F
- Note: GIF export reportedly broken; WebM works
- Usage:
  ```javascript
  const [bind, startRecording] = useCapture({ duration: 4, fps: 30 });
  // <Canvas {...bind} gl={{ preserveDrawingBuffer: true }}>
  ```

**Option C: MediaRecorder API (Browser-native)**
- No library needed
- `canvas.captureStream(30)` + `MediaRecorder`
- Outputs WebM or MP4 (codec support varies by browser)
- Less control over frame timing than CCapture.js
- Usage:
  ```javascript
  const stream = canvas.captureStream(30);
  const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
  recorder.start();
  // Later:
  recorder.stop();
  recorder.ondataavailable = (e) => { /* save e.data blob */ };
  ```

**Option D: Server-side GIF from matplotlib frames**
- Generate PNG frames server-side with matplotlib
- Assemble with `imageio` or `PIL` into GIF
- Lower quality but works without browser
- Good for training log animations

---

## 4. Gradio + React in HuggingFace Spaces

### 4.1 Gradio gr.HTML with Custom JavaScript (Recommended for Hackathon)

Gradio 6 introduced powerful `gr.HTML` customization. This is the fastest path to embedding Three.js in a Gradio app.

**How it works:**
- `html_template`: Handlebars (`{{}}`) or JS expression (`${}`) syntax
- `css_template`: Scoped CSS for the component
- `js_on_load`: JavaScript that runs when the component loads
- `server_functions`: Python functions callable from JS via `await server.func_name()`
- `@children`: Placeholder for embedding other Gradio components

**Three.js in gr.HTML pattern:**
```python
import gradio as gr

class OrigamiViewer(gr.HTML):
    def __init__(self, **kwargs):
        super().__init__(
            html_template="""
                <div id="origami-3d" style="width:100%; height:500px;"></div>
            """,
            js_on_load="""
                // Load Three.js from CDN
                const script = document.createElement('script');
                script.src = 'https://unpkg.com/three@0.160.0/build/three.module.js';
                script.type = 'module';
                // ... set up scene, camera, renderer, mesh
                // Access props.value for fold state data from Python
            """,
            css_template="""
                #origami-3d { border: 1px solid #ccc; border-radius: 8px; }
            """,
            server_functions=[get_fold_state],
            **kwargs
        )
```

**Data flow:**
- Python sets `props.value` with fold state data (JSON)
- JS reads `props.value` and updates the Three.js scene
- JS can call `await server.func_name(args)` to invoke Python functions
- JS can set `props.value = newData` and call `trigger()` to send data back

### 4.2 Gradio Custom Components (Full npm Package)

For a more polished, reusable component:

```bash
gradio cc create OrigamiViewer --template SimpleTextbox
```

This creates:
```
origami_viewer/
  backend/            # Python component code
  frontend/           # Svelte/JS component (can include React/Three.js)
  demo/               # Example app
  pyproject.toml
```

- Frontend is Svelte by default but can import any JS library
- Can include React + R3F as dependencies in the frontend build
- Published to PyPI, installable with `pip install`
- More work but produces a proper reusable component

### 4.3 Gradio Model3D Component (Built-in but Limited)

Gradio has a built-in `gr.Model3D` component:
```python
gr.Model3D(value="model.glb", label="3D View")
```
- Supports GLB, GLTF, OBJ, STL
- Has built-in viewer with rotation/zoom
- **Limitations**: No custom vertex colors, no animation, no strain overlay
- Could work for static final-state display only

### 4.4 Alternative: Static React App + FastAPI Backend

**Architecture:**
```
origami-space/
  Dockerfile
  backend/
    app.py            # FastAPI server (OpenEnv + API endpoints)
  frontend/
    src/
      App.tsx         # React app with R3F
    dist/             # Built static files
```

**FastAPI serves both:**
```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

**Pros:**
- Full React + R3F with all features (OrbitControls, vertex colors, animation)
- Complete control over UI layout
- No Gradio limitations

**Cons:**
- Doesn't use Gradio (may not match hackathon expectations)
- More setup work
- Need to handle WebSocket connection to OpenEnv manually

### 4.5 Alternative: Streamlit + Custom Component

- `streamlit.components.v1.components.html()` can embed HTML/JS
- `streamlit-stl` component exists for 3D model display
- Three.js can be embedded via HTML string injection
- Less mature than Gradio for HF Spaces
- **Not recommended** unless Gradio proves insufficient

### 4.6 Deployment on HuggingFace Spaces

**Docker Space (most flexible):**
```yaml
# README.md header
---
title: Origami RL
sdk: docker
app_port: 7860
---
```

**Static Space (for React-only frontend):**
```yaml
---
title: Origami Viewer
sdk: static
---
```
- Supports build steps: `npm run build`
- Serves built files directly

**Recommended approach**: Docker Space with FastAPI serving both the OpenEnv server and a Gradio UI that uses `gr.HTML` for the Three.js viewer.

---

## 5. Crease Pattern 2D Rendering

### 5.1 Standard Origami Conventions

| Edge Type | Color | Line Style | FOLD Assignment |
|-----------|-------|------------|-----------------|
| Mountain fold | Red | Dashed (dash-dot-dot) | "M" |
| Valley fold | Blue | Dashed | "V" |
| Boundary | Black | Solid | "B" |
| Unassigned | Gray | Dotted | "U" |
| Flat/crease | Gray | Solid thin | "F" |

**Note:** There is no universally standardized convention. The above is the most common in computational origami tools. Ghassaei's simulator uses: red=mountain, blue=valley, black=boundary. Flat-folder uses the same convention for SVG import.

**FOLD format edge assignments:**
- `edges_assignment`: Array of "M", "V", "B", "U", "F"
- `edges_foldAngle`: Array of numbers in [-180, 180] (positive=valley, negative=mountain, 0=flat/boundary)

### 5.2 SVG Rendering (Recommended for 2D View)

**Direct SVG in React:**
```jsx
function CreasePatternSVG({ vertices, edges, assignments }) {
  const getEdgeStyle = (assignment) => {
    switch (assignment) {
      case 'M': return { stroke: '#e74c3c', strokeDasharray: '8,3,2,3', strokeWidth: 2 };
      case 'V': return { stroke: '#3498db', strokeDasharray: '6,3', strokeWidth: 2 };
      case 'B': return { stroke: '#2c3e50', strokeDasharray: 'none', strokeWidth: 2.5 };
      case 'U': return { stroke: '#95a5a6', strokeDasharray: '2,4', strokeWidth: 1 };
      case 'F': return { stroke: '#bdc3c7', strokeDasharray: 'none', strokeWidth: 0.5 };
    }
  };

  return (
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      {edges.map(([v1, v2], i) => {
        const style = getEdgeStyle(assignments[i]);
        return (
          <line
            key={i}
            x1={vertices[v1][0]} y1={vertices[v1][1]}
            x2={vertices[v2][0]} y2={vertices[v2][1]}
            {...style}
          />
        );
      })}
      {vertices.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={0.5} fill="#333" />
      ))}
    </svg>
  );
}
```

### 5.3 SVG in Python (Server-Side, for matplotlib/Gradio)

```python
import svgwrite

def render_crease_pattern_svg(fold_data: dict) -> str:
    """Render FOLD data as SVG string."""
    vertices = fold_data['vertices_coords']
    edges = fold_data['edges_vertices']
    assignments = fold_data.get('edges_assignment', ['U'] * len(edges))

    STYLES = {
        'M': {'stroke': 'red', 'stroke_dasharray': '8,3,2,3', 'stroke_width': 2},
        'V': {'stroke': 'blue', 'stroke_dasharray': '6,3', 'stroke_width': 2},
        'B': {'stroke': 'black', 'stroke_dasharray': 'none', 'stroke_width': 2.5},
        'U': {'stroke': 'gray', 'stroke_dasharray': '2,4', 'stroke_width': 1},
        'F': {'stroke': 'lightgray', 'stroke_dasharray': 'none', 'stroke_width': 0.5},
    }

    dwg = svgwrite.Drawing(size=('400px', '400px'), viewBox='0 0 1 1')
    for (v1, v2), asgn in zip(edges, assignments):
        style = STYLES.get(asgn, STYLES['U'])
        dwg.add(dwg.line(
            start=vertices[v1], end=vertices[v2],
            **style
        ))
    return dwg.tostring()
```

### 5.4 matplotlib for 2D Crease Pattern (Quickest Python Path)

```python
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_crease_pattern(fold_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    vertices = fold_data['vertices_coords']
    edges = fold_data['edges_vertices']
    assignments = fold_data.get('edges_assignment', ['U'] * len(edges))

    STYLES = {
        'M': dict(color='red', linestyle='--', linewidth=2),
        'V': dict(color='blue', linestyle='-.', linewidth=2),
        'B': dict(color='black', linestyle='-', linewidth=2.5),
        'U': dict(color='gray', linestyle=':', linewidth=1),
        'F': dict(color='lightgray', linestyle='-', linewidth=0.5),
    }

    for (v1, v2), asgn in zip(edges, assignments):
        p1, p2 = vertices[v1], vertices[v2]
        style = STYLES.get(asgn, STYLES['U'])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style)

    ax.set_aspect('equal')
    ax.set_title('Crease Pattern')
    return ax
```

### 5.5 D3.js Option (Overkill for Our Needs)

D3.js could render crease patterns with full interactivity (hover tooltips, click-to-fold, zoom/pan), but:
- Adds another large dependency
- SVG-in-React is simpler and sufficient
- D3 is better for complex data visualization, not needed for line drawings
- **Skip D3 unless we need interactive editing of crease patterns**

---

## 6. UI Layout & Component Architecture

### 6.1 Proposed Layout

```
+------------------------------------------------------------------+
|  ORIGAMI RL ENVIRONMENT                          [Material: ▼]   |
+------------------------------------------------------------------+
|                          |                                        |
|   CREASE PATTERN (2D)   |          FOLDED STATE (3D)             |
|                          |                                        |
|   [SVG View]             |    [Three.js R3F Canvas]              |
|   Mountain = red ---     |    OrbitControls (rotate/zoom)        |
|   Valley  = blue -.-     |    Vertex colors = strain heatmap    |
|   Border  = black ___    |    DoubleSide material               |
|                          |                                        |
|   Scale: 1m x 1m         |    Camera: perspective, 45deg        |
|                          |                                        |
+------------------------------------------------------------------+
|                                                                    |
|   FOLD SEQUENCE                 METRICS DASHBOARD                 |
|   [Step 1] [Step 2] ...        Compactness: 0.85                 |
|   [Play] [Pause] [Reset]       Fold count: 12                    |
|   Progress: ████░░ 4/8         Max strain: 0.03                  |
|                                 Validity: PASS                    |
|                                 Deploy ratio: 15.2x              |
|                                                                    |
+------------------------------------------------------------------+
```

### 6.2 Component Breakdown

**For Gradio + gr.HTML approach:**

| Component | Implementation | Data Source |
|-----------|---------------|-------------|
| 2D Crease Pattern | SVG via `gr.HTML` or `gr.Plot` (matplotlib) | `vertices_coords`, `edges_vertices`, `edges_assignment` from FOLD |
| 3D Folded State | Three.js via `gr.HTML` with `js_on_load` | Vertex positions (3D), face indices, strain values |
| Strain Heatmap | Vertex colors on 3D mesh (Lut colormap) | Per-vertex strain from physics engine |
| Fold Sequence | `gr.Slider` + `gr.Button` controls | Fold step index, triggers re-render |
| Metrics Dashboard | `gr.Dataframe` or `gr.JSON` | Dict of metric name -> value |
| Material Selector | `gr.Dropdown` | Enum of material presets |

**For standalone React approach:**

| Component | Library | Notes |
|-----------|---------|-------|
| `<CreasePatternView>` | React SVG | Inline SVG, responsive |
| `<FoldedStateView>` | R3F + drei | Canvas with OrbitControls, BufferGeometry |
| `<StrainOverlay>` | Three.js Lut | Vertex color attribute on mesh |
| `<FoldSequencePlayer>` | React state + R3F useFrame | Timeline slider, play/pause |
| `<MetricsDashboard>` | React (plain HTML/CSS) | Cards with numeric values |
| `<MaterialSelector>` | React select | Updates physics constants |

### 6.3 Data Flow

```
Python Backend (OpenEnv)
    |
    | WebSocket / REST API
    | Sends: { vertices_coords, edges_vertices, edges_assignment,
    |          fold_angles, strain_per_vertex, metrics }
    v
Frontend (Gradio gr.HTML or React App)
    |
    +-- 2D View: vertices_coords[:, :2] + edges -> SVG lines
    |
    +-- 3D View: vertices_coords[:, :3] -> BufferGeometry positions
    |            faces_vertices -> index buffer
    |            strain_per_vertex -> Lut -> vertex colors
    |
    +-- Metrics: metrics dict -> dashboard display
    |
    +-- Animation: fold_angles array per step -> interpolate in useFrame
```

---

## 7. FOLD Format Integration

### 7.1 FOLD JavaScript Library

- **GitHub**: https://github.com/edemaine/fold
- **npm**: `npm install fold`
- **CDN**: Available via unpkg
- **Spec**: https://edemaine.github.io/fold/doc/spec.html

**Key data fields we need:**
```json
{
  "vertices_coords": [[0,0], [1,0], [1,1], [0,1]],
  "edges_vertices": [[0,1], [1,2], [2,3], [3,0], [0,2]],
  "edges_assignment": ["B", "B", "B", "B", "M"],
  "edges_foldAngle": [0, 0, 0, 0, -180],
  "faces_vertices": [[0,1,2], [0,2,3]]
}
```

**Parsing in Python:**
```python
import json

def load_fold(filepath):
    with open(filepath) as f:
        return json.load(f)
    # It's just JSON!
```

**Parsing in JavaScript:**
```javascript
// It's just JSON
const foldData = JSON.parse(fileContents);
// Or use the FOLD library for manipulation:
import FOLD from 'fold';
FOLD.filter.collapseNearbyVertices(foldData, epsilon);
```

### 7.2 Converting Between Python Backend and JS Frontend

The FOLD format is JSON-native, making Python-to-JS data transfer trivial:
- Python: `json.dumps(fold_data)` -> send via WebSocket/API
- JavaScript: `JSON.parse(message)` -> use directly with Three.js

---

## 8. Recommended Implementation Plan

### Phase 1: Minimum Viable Rendering (Day 1)

1. **Python-side 2D crease pattern** with matplotlib
   - `plot_crease_pattern(fold_data)` function
   - Display in Gradio via `gr.Plot`
   - Mountain=red dashed, Valley=blue dash-dot, Boundary=black solid

2. **Python-side 3D wireframe** with matplotlib mplot3d
   - `plot_folded_state(fold_data)` function
   - Basic wireframe of folded vertices
   - Display in Gradio via `gr.Plot`

3. **Metrics as gr.JSON or gr.Dataframe**

### Phase 2: Interactive 3D Viewer (Day 2)

4. **Three.js viewer via gr.HTML**
   - Load Three.js + OrbitControls from CDN in `js_on_load`
   - Receive fold state as `props.value` (JSON)
   - Render mesh with vertex colors for strain
   - OrbitControls for rotation/zoom

5. **SVG crease pattern via gr.HTML**
   - Render 2D crease pattern as inline SVG
   - Update when fold state changes

### Phase 3: Animation & Polish (Day 3)

6. **Fold sequence animation**
   - Slider to step through fold sequence
   - Animate fold angle interpolation
   - Play/pause controls

7. **Screenshot/recording**
   - `canvas.toDataURL()` for screenshots
   - MediaRecorder API for video (simpler than CCapture.js)

8. **Material visual differentiation**
   - Different colors/textures for paper vs mylar vs aluminum
   - Opacity/shininess changes based on material

---

## 9. Key Technical Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Three.js in gr.HTML is janky | Fall back to Plotly 3D mesh or static React app |
| Vertex manipulation perf for large meshes | Limit mesh resolution (100x100 grid max), use Web Worker |
| FOLD format conversion errors | Use FOLD JS library for validation, test with known patterns |
| HF Spaces doesn't support WebGL | All modern browsers support WebGL; provide matplotlib fallback |
| Animation recording in Gradio | Use server-side GIF generation with imageio as fallback |
| OrbitControls conflict with Gradio events | Isolate Three.js canvas, prevent event propagation |

---

## 10. Key Dependencies

### Python (Backend)
```
numpy          # Mesh computation
matplotlib     # 2D/3D fallback rendering
svgwrite       # SVG crease pattern generation (optional)
imageio        # GIF export from frames
plotly         # Interactive 3D (optional fallback)
trimesh        # Mesh utilities (optional)
```

### JavaScript (Frontend, loaded via CDN or npm)
```
three          # 3D rendering engine
@react-three/fiber   # React wrapper (if standalone React app)
@react-three/drei    # Helpers (OrbitControls, etc.)
fold           # FOLD format manipulation
ccapture.js    # Animation recording (optional)
```

### For Gradio gr.HTML approach (no npm needed)
```html
<!-- Load from CDN in js_on_load -->
<script src="https://unpkg.com/three@0.160.0/build/three.module.js" type="module"></script>
<script src="https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js" type="module"></script>
```

---

## Sources

- [Origami Simulator - GitHub](https://github.com/amandaghassaei/OrigamiSimulator)
- [Origami Simulator - Live Demo](https://origamisimulator.org/)
- [OrigamiOdyssey - React + R3F Origami App](https://github.com/robbwdoering/origamiodyssey)
- [georgiee/origami - Three.js Origami Builder](https://github.com/georgiee/origami)
- [flat-folder - Flat-Foldable State Computation](https://github.com/origamimagiro/flat-folder)
- [crease - SVG Crease Pattern Editor](https://github.com/mwalczyk/crease)
- [FOLD File Format - GitHub](https://github.com/edemaine/fold)
- [FOLD Specification](https://edemaine.github.io/fold/doc/spec.html)
- [FOLD npm package](https://www.npmjs.com/package/fold)
- [react-three-fiber - GitHub](https://github.com/pmndrs/react-three-fiber)
- [drei - R3F Helpers](https://github.com/pmndrs/drei)
- [R3F Basic Animations Tutorial](https://r3f.docs.pmnd.rs/tutorials/basic-animations)
- [Three.js Lut Documentation](https://threejs.org/docs/examples/en/math/Lut.html)
- [Three.js Vertex Colors Lookup Table Example](https://threejs.org/examples/webgl_geometry_colors_lookuptable.html)
- [Three.js Heatmap on 3D Model - Forum](https://discourse.threejs.org/t/how-to-creating-heatmap-over-3d-model/52744)
- [Three.js Vertex Color BufferGeometry](https://dustinpfister.github.io/2023/01/20/threejs-buffer-geometry-attributes-color/)
- [Three.js Mesh Modifiers (Bend/Twist)](https://github.com/drawcall/threejs-mesh-modifiers)
- [CCapture.js - Canvas Animation Capture](https://github.com/spite/ccapture.js/)
- [use-capture - R3F Recording Hook](https://github.com/gsimone/use-capture)
- [Videos and GIFs with Three.js](https://janakiev.com/blog/videos-and-gifs-with-threejs/)
- [R3F Screenshot Discussion](https://github.com/pmndrs/react-three-fiber/discussions/2054)
- [Headless Three.js Rendering - Forum](https://discourse.threejs.org/t/headless-rendering/14401)
- [Gradio Custom HTML Components Guide](https://www.gradio.app/guides/custom-HTML-components)
- [Gradio Custom Components in Five Minutes](https://www.gradio.app/guides/custom-components-in-five-minutes)
- [Gradio gr.HTML One-Shot Apps](https://huggingface.co/blog/gradio-html-one-shot-apps)
- [Gradio Model3D Component](https://www.gradio.app/docs/gradio/model3d)
- [HuggingFace Static HTML Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
- [Create Static HF Space with React](https://blog.rednegra.net/2024/10/14/create-a-static-huggingface-space-with-react)
- [Streamlit + R3F Discussion](https://discuss.streamlit.io/t/streamlit-components-wrap-around-react-three-fiber/4749)
- [OriDomi - CSS Paper Folding](http://oridomi.com/)
- [Fast, Interactive Origami Simulation using GPU Computation (Paper)](https://erikdemaine.org/papers/OrigamiSimulator_Origami7/paper.pdf)
