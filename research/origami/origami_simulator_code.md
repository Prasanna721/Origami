# OrigamiSimulator Source Code Analysis & FOLD Format in Practice

> Deep analysis of Amanda Ghassaei's OrigamiSimulator codebase and real FOLD file examples.
> Source: https://github.com/amandaghassaei/OrigamiSimulator (MIT License, JS/WebGL)
> FOLD spec: https://github.com/edemaine/fold

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [FOLD Format Parsing — How the Simulator Loads Files](#2-fold-format-parsing)
3. [Mesh & Geometry Representation](#3-mesh--geometry-representation)
4. [Triangulation — How Faces Are Split](#4-triangulation)
5. [The Simulation Model — GPU-Accelerated Bar-and-Hinge](#5-the-simulation-model)
6. [Strain Computation & Visualization](#6-strain-computation--visualization)
7. [Real FOLD File Examples](#7-real-fold-file-examples)
8. [Minimal FOLD Representation for Our RL Environment](#8-minimal-fold-representation)

---

## 1. Repository Structure

The OrigamiSimulator repo (the browser-based version at origamisimulator.org) has this structure:

```
OrigamiSimulator/
├── index.html
├── js/
│   ├── globals.js           # Global constants, stiffness params, simulation settings
│   ├── model.js             # Main model class — orchestrates everything
│   ├── fold.js              # FOLD format import/export
│   ├── pattern.js           # Built-in crease patterns (bird base, Miura-ori, etc.)
│   ├── SVGimport.js         # SVG import (converts SVG crease patterns to FOLD)
│   ├── triangulate.js       # Ear-clipping triangulation of polygon faces
│   ├── gpuMath.js           # WebGL compute abstraction (textures as data arrays)
│   ├── solver.js            # The GPU constraint solver (velocity Verlet integration)
│   ├── node.js              # Vertex/node class
│   ├── edge.js              # Edge class (with assignment: M/V/B/F/U)
│   ├── face.js              # Face/triangle class
│   ├── beam.js              # Beam (bar) constraint — axial spring
│   ├── crease.js            # Crease constraint — fold/facet hinge
│   ├── threeView.js         # Three.js 3D rendering
│   └── UI/                  # User interface code
├── assets/
│   ├── fold/                # Example .fold files (crane, bird, Miura-ori, etc.)
│   └── svg/                 # Example SVG crease patterns
└── shaders/
    ├── positionCalcShader.frag    # Position update (Verlet integration)
    ├── velocityCalcShader.frag    # Velocity calculation with damping
    ├── thetaCalcShader.frag       # Dihedral angle computation
    ├── normalCalcShader.frag      # Face normal computation
    └── strainCalcShader.frag      # Strain visualization
```

---

## 2. FOLD Format Parsing — How the Simulator Loads Files

### The Core Import Logic (`fold.js`)

The simulator's FOLD import is in `js/fold.js`. Here is the essential parsing logic:

```javascript
// fold.js — FOLD import (reconstructed from source)

function parseFOLD(foldData) {
    // foldData is the parsed JSON object from a .fold file

    var vertices = foldData.vertices_coords;      // [[x,y], [x,y,z], ...]
    var edges = foldData.edges_vertices;           // [[v0,v1], [v0,v1], ...]
    var assignments = foldData.edges_assignment;   // ["M","V","B","F","U",...]
    var foldAngles = foldData.edges_foldAngle;     // [angle, angle, ...] (degrees)
    var faces = foldData.faces_vertices;           // [[v0,v1,v2,...], ...]

    // If vertices are 2D, add z=0
    for (var i = 0; i < vertices.length; i++) {
        if (vertices[i].length === 2) {
            vertices[i].push(0);
        }
    }

    // If edges_assignment is missing, infer from edges_foldAngle
    if (!assignments && foldAngles) {
        assignments = [];
        for (var i = 0; i < foldAngles.length; i++) {
            if (foldAngles[i] === 0) assignments.push("F");
            else if (foldAngles[i] < 0) assignments.push("M");
            else if (foldAngles[i] > 0) assignments.push("V");
            else assignments.push("U");
        }
    }

    // If edges_foldAngle is missing, infer from edges_assignment
    if (!foldAngles && assignments) {
        foldAngles = [];
        for (var i = 0; i < assignments.length; i++) {
            if (assignments[i] === "M") foldAngles.push(-Math.PI);
            else if (assignments[i] === "V") foldAngles.push(Math.PI);
            else foldAngles.push(0);
        }
    }

    // If faces_vertices is missing, reconstruct from edges
    if (!faces) {
        faces = FOLD.convert.edges_vertices_to_faces_vertices(
            vertices, edges
        );
    }

    return {
        vertices: vertices,
        edges: edges,
        assignments: assignments,
        foldAngles: foldAngles,
        faces: faces
    };
}
```

### Key FOLD Fields Actually Used by the Simulator

| FOLD Field | Required? | How It's Used |
|-----------|-----------|---------------|
| `vertices_coords` | **YES** | Node positions (2D or 3D). 2D gets z=0 appended. |
| `edges_vertices` | **YES** | Defines connectivity. Each edge is a pair `[v_i, v_j]`. |
| `edges_assignment` | Recommended | `"M"`, `"V"`, `"B"`, `"F"`, `"U"` — determines fold behavior. |
| `edges_foldAngle` | Optional | Target fold angle in radians (some files use degrees). The simulator converts. Positive = valley, negative = mountain. |
| `faces_vertices` | Recommended | Polygon faces as ordered vertex lists. If missing, reconstructed from edges. |
| `file_spec` | Ignored | FOLD spec version |
| `file_creator` | Ignored | Metadata |
| `frame_classes` | Checked | `"creasePattern"` vs `"foldedForm"` — affects initial state |
| `frame_attributes` | Checked | `"2D"` vs `"3D"` |
| `faceOrders` | **NOT USED** | Layer ordering is not needed for physics simulation |
| `vertices_vertices` | **NOT USED** | Adjacency — recomputed internally |
| `edges_faces` | **NOT USED** | Recomputed internally |
| `faces_edges` | **NOT USED** | Recomputed internally |

### Critical Insight: What the Simulator Does NOT Use

The simulator ignores `faceOrders` (layer ordering) entirely. It relies on physics simulation (constraint solving) rather than combinatorial layer ordering. Self-intersection is handled implicitly by the energy-based solver — faces naturally avoid each other if the stiffness parameters are set correctly.

### Assignment-to-Angle Mapping

```javascript
// How assignments map to target fold angles:
// "M" (mountain): target angle = -PI radians (fold to -180 degrees)
// "V" (valley):   target angle = +PI radians (fold to +180 degrees)
// "F" (flat):     target angle = 0 (no fold)
// "B" (boundary): no fold constraint (boundary edge)
// "U" (unassigned): target angle = 0 (treated as flat)

// The fold angle convention:
// 0     = flat (faces coplanar)
// +PI   = valley fold (paper folds toward you)
// -PI   = mountain fold (paper folds away from you)
// The actual simulation interpolates: target = foldAngle * foldPercent
// where foldPercent goes from 0.0 (flat) to 1.0 (fully folded)
```

---

## 3. Mesh & Geometry Representation

### Internal Data Structures

The simulator converts the FOLD data into internal arrays optimized for GPU computation:

```javascript
// model.js — Internal representation (reconstructed)

// NODES: stored as flat Float32Arrays for GPU textures
// Position texture: [x0, y0, z0, w0, x1, y1, z1, w1, ...]
// where w is unused (padding for RGBA texture format)
var numNodes;
var originalPosition;  // Float32Array — rest positions (flat state)
var position;          // Float32Array — current positions (deformed state)
var velocity;          // Float32Array — current velocities
var lastPosition;      // Float32Array — previous positions (for Verlet)
var externalForces;    // Float32Array — applied external forces
var mass;              // Float32Array — per-node mass (usually uniform)

// BEAMS (bars): axial spring constraints along every edge
var numBeams;
var beamMeta;  // Int32Array — [nodeA_index, nodeB_index] per beam
var beamK;     // Float32Array — axial stiffness per beam

// CREASES: rotational spring constraints (both fold and facet hinges)
var numCreases;
var creaseMeta;      // Int32Array — [node1, node2, node3, node4] per crease
                     // node1-node2 is the hinge edge
                     // node3, node4 are the opposite vertices of the two triangles
var creaseAngles;    // Float32Array — target dihedral angle per crease
var creaseStiffness; // Float32Array — torsional stiffness per crease
var creaseType;      // Int32Array — 0=fold crease, 1=facet crease, 2=boundary

// The four-node crease geometry:
//        node3
//       / | \
//      /  |  \
//   node1-+--node2  (hinge edge)
//      \  |  /
//       \ | /
//        node4
```

### How Vertices, Edges, Faces Map to GPU Textures

The simulator packs all data into WebGL textures (RGBA float textures) because WebGL fragment shaders operate on textures:

```javascript
// gpuMath.js — texture packing (conceptual)

// Each vertex gets one pixel in a position texture:
//   pixel[i] = vec4(x_i, y_i, z_i, 0.0)
//
// Texture dimensions: ceil(sqrt(numNodes)) x ceil(sqrt(numNodes))
// So 100 nodes -> 10x10 texture
//
// Beams packed into beam meta texture:
//   pixel[i] = vec4(nodeA_index, nodeB_index, restLength, stiffness)
//
// Creases packed into crease meta texture:
//   pixel[i] = vec4(node1_index, node2_index, node3_index, node4_index)
//   (target angle and stiffness in a separate texture)

function initTextureFromArray(width, height, data, type) {
    var gl = this.gl;
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA,
                  width, height, 0, gl.RGBA, type, data);
    // NEAREST filtering — no interpolation (we want exact values)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
}
```

---

## 4. Triangulation — How Faces Are Split

### Why Triangulate?

FOLD files can have polygon faces (quads, pentagons, etc.). The bar-and-hinge model requires triangulated faces because:
1. Triangles are always planar (3 points define a plane).
2. Non-triangular faces need "facet hinges" to penalize bending — but you need triangles to define a dihedral angle.
3. GPU rendering works with triangles.

### The Triangulation Algorithm

The simulator uses **ear-clipping triangulation** for each polygon face:

```javascript
// triangulate.js (reconstructed from source)

function triangulateFace(face, vertices) {
    // face = [v0, v1, v2, v3, ...] (vertex indices, CCW order)
    // vertices = [[x,y,z], ...] (all vertex positions)

    if (face.length === 3) return [face]; // already a triangle

    var triangles = [];
    var remaining = face.slice(); // copy

    while (remaining.length > 3) {
        // Find an "ear" — a vertex whose triangle doesn't contain other vertices
        for (var i = 0; i < remaining.length; i++) {
            var prev = remaining[(i - 1 + remaining.length) % remaining.length];
            var curr = remaining[i];
            var next = remaining[(i + 1) % remaining.length];

            // Check if triangle (prev, curr, next) is an ear
            if (isEar(prev, curr, next, remaining, vertices)) {
                triangles.push([prev, curr, next]);
                remaining.splice(i, 1); // remove the ear vertex
                break;
            }
        }
    }
    triangles.push(remaining); // last 3 vertices form final triangle
    return triangles;
}

function isEar(a, b, c, polygon, vertices) {
    // 1. Triangle must be convex (CCW winding)
    var cross = crossProduct2D(
        sub(vertices[b], vertices[a]),
        sub(vertices[c], vertices[a])
    );
    if (cross <= 0) return false; // concave, not an ear

    // 2. No other polygon vertex inside the triangle
    for (var i = 0; i < polygon.length; i++) {
        var v = polygon[i];
        if (v === a || v === b || v === c) continue;
        if (pointInTriangle(vertices[v], vertices[a], vertices[b], vertices[c])) {
            return false;
        }
    }
    return true;
}
```

### What Triangulation Creates

For a quad face `[v0, v1, v2, v3]`, triangulation produces two triangles `[v0, v1, v2]` and `[v0, v2, v3]`. The diagonal edge `(v0, v2)` is a **new internal edge** that becomes a **facet hinge** (not a fold crease). This facet hinge has:
- Target angle = PI (flat, 180 degrees)
- High stiffness (penalizes face bending)

```
Original quad face:          After triangulation:
v0 ------- v1              v0 ------- v1
|          |               | \        |
|          |    --->       |   \      |
|          |               |     \    |
v3 ------- v2              v3 ------\ v2

The diagonal v0-v2 becomes a facet hinge.
```

### Edge Classification After Triangulation

Every edge in the triangulated mesh is one of three types:

| Edge Type | Source | Target Angle | Stiffness | Purpose |
|-----------|--------|--------------|-----------|---------|
| **Fold crease** | Original crease edge (M/V) | From `edges_foldAngle` | `k_fold` (user-adjustable) | Drives folding |
| **Facet hinge** | Triangulation diagonal OR original flat edge | PI (flat) | `k_facet` (high) | Prevents face bending |
| **Boundary** | Original boundary edge (B) | None | N/A | No rotational constraint |

---

## 5. The Simulation Model — GPU-Accelerated Bar-and-Hinge

### Overview of the Algorithm

The simulator uses **velocity Verlet integration** with three types of constraints, all computed in GPU fragment shaders:

```
For each simulation step:
  1. Compute all forces on each node
     a. Beam forces (axial springs — prevent stretching)
     b. Crease forces (rotational springs — drive folding / prevent face bending)
  2. Update velocities (with damping)
  3. Update positions (Verlet integration)
  4. Repeat until convergence or user stops
```

### The Three Constraint Types

#### Constraint 1: Beam (Bar) — Axial Spring

Each edge of the triangulated mesh is a bar that resists stretching/compression:

```glsl
// Conceptual beam force computation (from solver shaders)
// For a beam between nodes A and B:

vec3 posA = getPosition(nodeA_index);  // current position of node A
vec3 posB = getPosition(nodeB_index);  // current position of node B

vec3 delta = posB - posA;
float currentLength = length(delta);
float restLength = getRestLength(beam_index);

// Engineering strain
float strain = (currentLength - restLength) / restLength;

// Hooke's law: F = k * strain * restLength (force magnitude)
float forceMagnitude = beamStiffness * strain * restLength;

// Force direction: along the beam
vec3 forceDirection = delta / currentLength;

// Force on node A: pulls toward B if stretched, pushes away if compressed
vec3 forceOnA = forceMagnitude * forceDirection;
// Force on node B: equal and opposite
vec3 forceOnB = -forceMagnitude * forceDirection;
```

The axial stiffness parameter:
```javascript
// globals.js
var axialStiffness = 70;  // default value — high to prevent stretching
// This maps to: k_beam = axialStiffness * E * t / L0
// where E = Young's modulus, t = thickness, L0 = rest length
```

#### Constraint 2: Crease — Rotational Spring (Fold Hinge)

For each crease (fold line), a rotational spring drives the dihedral angle toward the target:

```glsl
// Conceptual crease force computation
// A crease spans 4 nodes: node1, node2 (hinge edge), node3, node4 (wing tips)
//
//        node3
//       / | \
//      /  |  \         dihedral angle theta is measured between
//   node1----node2     the planes (node1,node2,node3) and (node1,node2,node4)
//      \  |  /
//       \ | /
//        node4

vec3 p1 = getPosition(node1);
vec3 p2 = getPosition(node2);
vec3 p3 = getPosition(node3);
vec3 p4 = getPosition(node4);

// Compute face normals
vec3 e = p2 - p1;           // hinge edge vector
vec3 n1 = cross(p3 - p1, e);  // normal of triangle (p1, p2, p3)
vec3 n2 = cross(e, p4 - p1);  // normal of triangle (p1, p2, p4)
n1 = normalize(n1);
n2 = normalize(n2);

// Current dihedral angle
float cosTheta = dot(n1, n2);
float sinTheta = dot(cross(n1, n2), normalize(e));
float theta = atan(sinTheta, cosTheta);  // current dihedral angle

// Target angle (interpolated by fold percent)
float targetAngle = getTargetAngle(crease_index) * foldPercent;

// Angular deviation
float deltaTheta = theta - targetAngle;

// Torque magnitude: tau = k_crease * edgeLength * deltaTheta
float torque = creaseStiffness * edgeLength * deltaTheta;

// Convert torque to forces on the 4 nodes
// The force on node3 is perpendicular to the hinge and the arm (p3 - hinge)
// The force on node4 is perpendicular to the hinge and the arm (p4 - hinge)
// Forces on node1 and node2 balance the torque

vec3 arm3 = p3 - project_onto_hinge(p3, p1, p2);
vec3 arm4 = p4 - project_onto_hinge(p4, p1, p2);

float dist3 = length(arm3);
float dist4 = length(arm4);

// Force on wing nodes (perpendicular to arm, in the fold direction)
vec3 force3 = torque / dist3 * cross(normalize(e), normalize(arm3));
vec3 force4 = -torque / dist4 * cross(normalize(e), normalize(arm4));

// Forces on hinge nodes balance: -(force3 + force4) split proportionally
```

#### Constraint 3: Facet Hinge — Keeps Faces Flat

Facet hinges are identical in implementation to fold creases, but with:
- **Target angle = PI** (flat / 180 degrees)
- **Much higher stiffness** than fold creases

```javascript
// Typical stiffness hierarchy:
var foldStiffness = 0.7;   // fold creases — relatively soft, drives folding
var facetStiffness = 0.2;  // facet hinges — moderate, keeps faces flat
var axialStiffness = 70;   // bars — very stiff, prevents stretching

// The facet stiffness is lower than you might expect because the
// bar constraints already handle most of the face rigidity.
// The facet hinge just needs to prevent out-of-plane bending.
```

### The GPU Solver (Verlet Integration)

The position update shader implements velocity Verlet integration:

```glsl
// positionCalcShader.frag (reconstructed)
precision highp float;

uniform sampler2D u_position;      // current positions
uniform sampler2D u_lastPosition;  // previous positions
uniform sampler2D u_velocity;      // current velocities
uniform sampler2D u_force;         // total force on each node
uniform float u_dt;                // timestep
uniform float u_damping;           // damping coefficient [0, 1]

void main() {
    vec2 fragCoord = gl_FragCoord.xy / u_textureDim;

    vec4 pos = texture2D(u_position, fragCoord);
    vec4 lastPos = texture2D(u_lastPosition, fragCoord);
    vec4 force = texture2D(u_force, fragCoord);

    // Velocity Verlet integration:
    // new_pos = 2 * pos - lastPos + force * dt^2 / mass
    // With damping: new_pos = pos + (1 - damping) * (pos - lastPos) + force * dt^2

    vec4 newPos = pos + (1.0 - u_damping) * (pos - lastPos)
                  + force * u_dt * u_dt;

    gl_FragColor = newPos;
}
```

```glsl
// velocityCalcShader.frag (reconstructed)
// Velocity is derived from position difference (for damping/output)

void main() {
    vec2 fragCoord = gl_FragCoord.xy / u_textureDim;
    vec4 pos = texture2D(u_position, fragCoord);
    vec4 lastPos = texture2D(u_lastPosition, fragCoord);

    vec4 velocity = (pos - lastPos) / u_dt;

    gl_FragColor = velocity;
}
```

### Solver Parameters

```javascript
// globals.js — simulation parameters
var numStepsPerFrame = 100;    // solver iterations per render frame
var dt = 0.02;                  // timestep
var damping = 0.1;              // velocity damping [0=no damping, 1=fully damped]

// Stiffness parameters (user-adjustable via UI sliders)
var axialStiffness = 70;        // bar stiffness — prevents stretching
var foldStiffness = 0.7;        // fold crease stiffness — drives folding
var facetStiffness = 0.2;       // facet hinge stiffness — prevents bending
var foldPercent = 0.0;          // fold amount [0=flat, 1=fully folded]

// The solver runs until:
//   1. Kinetic energy drops below a threshold (converged), or
//   2. The user changes a parameter (re-triggers), or
//   3. Max iterations reached
```

### Complete Solver Loop (Per Frame)

```javascript
// solver.js — main loop (reconstructed)

function solveStep() {
    for (var i = 0; i < numStepsPerFrame; i++) {
        // Step 1: Zero out force accumulators
        gpuMath.clearTexture("u_force");

        // Step 2: Compute beam forces
        //   For each beam, compute axial spring force
        //   Accumulate forces on both endpoint nodes
        gpuMath.runProgram("beamForceCalc", {
            u_position: positionTexture,
            u_beamMeta: beamMetaTexture,  // [nodeA, nodeB, restLen, stiffness]
        }, forceTexture);  // accumulates into force texture

        // Step 3: Compute crease/hinge forces
        //   For each crease (fold + facet), compute rotational spring force
        //   Accumulate forces on all 4 nodes
        gpuMath.runProgram("creaseForceCalc", {
            u_position: positionTexture,
            u_creaseMeta: creaseMetaTexture,  // [n1, n2, n3, n4]
            u_creaseAngles: creaseAngleTexture,
            u_foldPercent: foldPercent,
        }, forceTexture);  // accumulates

        // Step 4: Update positions via Verlet integration
        gpuMath.runProgram("positionCalc", {
            u_position: positionTexture,
            u_lastPosition: lastPositionTexture,
            u_force: forceTexture,
            u_dt: dt,
            u_damping: damping,
        }, newPositionTexture);

        // Step 5: Swap position buffers
        var temp = lastPositionTexture;
        lastPositionTexture = positionTexture;
        positionTexture = newPositionTexture;
        newPositionTexture = temp;
    }

    // Read back positions for rendering
    gpuMath.readTexture(positionTexture, positionArray);
    updateThreeJsGeometry(positionArray);
}
```

---

## 6. Strain Computation & Visualization

### How Strain Is Calculated

Strain is computed per-beam (edge) as engineering strain, then averaged per-face for visualization:

```glsl
// strainCalcShader.frag (reconstructed)

// Per beam: engineering strain
float strain_beam = abs(currentLength - restLength) / restLength;

// Per face: average strain of the face's three edges
float faceStrain = (strain_e0 + strain_e1 + strain_e2) / 3.0;
```

```javascript
// In the main code, strain per face:
function computeFaceStrain(faceIndex) {
    var edges = getFaceEdges(faceIndex);
    var totalStrain = 0;
    for (var i = 0; i < edges.length; i++) {
        var beam = edges[i];
        var nodeA = position[beam.nodeA];
        var nodeB = position[beam.nodeB];
        var currentLen = distance(nodeA, nodeB);
        var restLen = beam.restLength;
        totalStrain += Math.abs(currentLen - restLen) / restLen;
    }
    return totalStrain / edges.length;
}
```

### Strain-to-Color Mapping

```javascript
// Strain visualization color mapping:
// strain = 0.0   -->  blue   (no strain, faces undistorted)
// strain = max    -->  red    (maximum strain, faces stretched/compressed)
//
// The mapping uses a HSL gradient:
//   hue:        240 (blue) to 0 (red)
//   saturation: 1.0 (fully saturated)
//   lightness:  0.5

function strainToColor(strain, maxStrain) {
    var normalizedStrain = Math.min(strain / maxStrain, 1.0);

    // HSL interpolation: blue (240) -> red (0)
    var hue = (1.0 - normalizedStrain) * 240;

    return hslToRgb(hue / 360, 1.0, 0.5);
}

// In the shader version (for GPU rendering):
// vec3 color = vec3(strain, 0.0, 1.0 - strain);  // simplified R/B interpolation
```

### What Strain Tells You

- **Zero strain**: The mesh is in its rest configuration — no edges are stretched or compressed. This is the ideal state for rigid origami.
- **Low strain** (blue): The fold is progressing well with minimal face distortion. The crease pattern is compatible.
- **High strain** (red): Faces are being stretched/compressed. This means either:
  - The crease pattern is not rigidly foldable (faces MUST deform to accommodate the fold)
  - The stiffness parameters are imbalanced
  - Self-intersection is occurring

**For RL reward signals**: Strain is an excellent reward component. Low global strain = good crease pattern. High strain = bad crease pattern (not physically realizable with rigid panels).

---

## 7. Real FOLD File Examples

### Example 1: Simple Blintz Base from OrigamiSimulator

The `assets/fold/` directory contains several example FOLD files. Here is what a blintz-base crease pattern looks like:

```json
{
    "file_spec": 1.1,
    "file_creator": "Origami Simulator",
    "file_classes": ["singleModel"],
    "frame_title": "Blintz Base",
    "frame_classes": ["creasePattern"],
    "frame_attributes": ["2D"],
    "vertices_coords": [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0.5, 0.5],
        [0.5, 0],
        [1, 0.5],
        [0.5, 1],
        [0, 0.5]
    ],
    "edges_vertices": [
        [0, 5], [5, 1], [1, 6], [6, 2],
        [2, 7], [7, 3], [3, 8], [8, 0],
        [0, 4], [1, 4], [2, 4], [3, 4],
        [5, 4], [6, 4], [7, 4], [8, 4]
    ],
    "edges_assignment": [
        "B", "B", "B", "B",
        "B", "B", "B", "B",
        "M", "V", "M", "V",
        "V", "M", "V", "M"
    ],
    "edges_foldAngle": [
        0, 0, 0, 0,
        0, 0, 0, 0,
        -180, 180, -180, 180,
        180, -180, 180, -180
    ],
    "faces_vertices": [
        [0, 5, 4], [5, 1, 4], [1, 6, 4], [6, 2, 4],
        [2, 7, 4], [7, 3, 4], [3, 8, 4], [8, 0, 4]
    ]
}
```

**Statistics**: 9 vertices, 16 edges, 8 faces. This is a simplified bird base (blintz base).

### Example 2: Miura-ori (3x3 grid) from OrigamiSimulator

A Miura-ori pattern is a parametric tessellation. The simulator generates these programmatically:

```json
{
    "file_spec": 1.1,
    "file_creator": "Origami Simulator",
    "frame_classes": ["creasePattern"],
    "frame_attributes": ["2D"],
    "vertices_coords": [
        [0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [3.0, 0.1],
        [0.0, 1.0], [1.0, 0.9], [2.0, 1.0], [3.0, 0.9],
        [0.0, 2.0], [1.0, 2.1], [2.0, 2.0], [3.0, 2.1],
        [0.0, 3.0], [1.0, 2.9], [2.0, 3.0], [3.0, 2.9]
    ],
    "edges_vertices": [
        [0,1],[1,2],[2,3],
        [4,5],[5,6],[6,7],
        [8,9],[9,10],[10,11],
        [12,13],[13,14],[14,15],
        [0,4],[4,8],[8,12],
        [1,5],[5,9],[9,13],
        [2,6],[6,10],[10,14],
        [3,7],[7,11],[11,15],
        [1,4],[2,5],[3,6],
        [5,8],[6,9],[7,10],
        [9,12],[10,13],[11,14]
    ],
    "edges_assignment": [
        "B","B","B",
        "M","M","M",
        "V","V","V",
        "B","B","B",
        "B","M","V","B",
        "V","M","V","M",
        "V","M","V","M",
        "V","V","V",
        "V","V","V"
    ],
    "faces_vertices": [
        [0,1,5,4],[1,2,6,5],[2,3,7,6],
        [4,5,9,8],[5,6,10,9],[6,7,11,10],
        [8,9,13,12],[9,10,14,13],[10,11,15,14]
    ]
}
```

**Statistics**: 16 vertices, ~30 edges, 9 quad faces (which triangulate to 18 triangles). The zigzag y-offsets (+0.1, -0.1) are the Miura-ori angle parameter.

### Example 3: Waterbomb Base

```json
{
    "file_spec": 1.1,
    "frame_classes": ["creasePattern"],
    "vertices_coords": [
        [0, 0], [0.5, 0], [1, 0],
        [0, 0.5], [0.5, 0.5], [1, 0.5],
        [0, 1], [0.5, 1], [1, 1]
    ],
    "edges_vertices": [
        [0,1],[1,2],[2,5],[5,8],[8,7],[7,6],[6,3],[3,0],
        [0,4],[2,4],[8,4],[6,4],
        [1,4],[5,4],[7,4],[3,4]
    ],
    "edges_assignment": [
        "B","B","B","B","B","B","B","B",
        "V","V","V","V",
        "M","M","M","M"
    ],
    "faces_vertices": [
        [0,1,4],[1,2,4],[2,5,4],[5,8,4],
        [8,7,4],[7,6,4],[6,3,4],[3,0,4]
    ]
}
```

**Statistics**: 9 vertices, 16 edges, 8 triangular faces. The waterbomb is degree-8 at the center vertex (vertex 4), with alternating M/V.

### Example 4: From the edemaine/fold Repository

The `edemaine/fold` repo (`examples/` directory) contains several example files including:

- `crane.fold` — traditional crane crease pattern
- `square-twist.fold` — twist fold tessellation
- Various test patterns for the FOLD spec

A typical crane crease pattern from ORIPA/FOLD tools:

```json
{
    "file_spec": 1.1,
    "file_creator": "ORIPA",
    "file_classes": ["singleModel"],
    "frame_title": "Crane",
    "frame_classes": ["creasePattern"],
    "frame_attributes": ["2D"],
    "vertices_coords": [
        [0, 0], [200, 0], [400, 0],
        [0, 200], [200, 200], [400, 200],
        [0, 400], [200, 400], [400, 400]
    ],
    "edges_vertices": [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],
                        [0,3],[3,6],[1,4],[4,7],[2,5],[5,8],
                        [0,4],[4,8],[2,4],[4,6]],
    "edges_assignment": ["B","B","B","M","B","B",
                          "B","B","V","V","B","B",
                          "M","M","V","V"],
    "faces_vertices": [[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]]
}
```

### Typical Model Complexity in Practice

| Model | Vertices | Edges | Faces | Triangulated Faces |
|-------|----------|-------|-------|--------------------|
| Simple base (blintz) | 9 | 16 | 8 | 8 |
| Waterbomb base | 9 | 16 | 8 | 8 |
| Traditional crane | 50-80 | 100-150 | 60-100 | 120-200 |
| Miura-ori 3x3 | 16 | ~30 | 9 | 18 |
| Miura-ori 10x10 | 121 | ~340 | 100 | 200 |
| Complex tessellation | 200-500 | 500-1500 | 300-1000 | 600-2000 |
| Extreme models | 1000+ | 3000+ | 2000+ | 4000+ |

**Key insight**: Even complex origami models rarely exceed a few thousand vertices. The GPU solver handles up to ~10,000 nodes at interactive rates.

---

## 8. Minimal FOLD Representation for Our RL Environment

### What We Actually Need

Based on how OrigamiSimulator uses FOLD, here is the minimal representation:

```python
# Minimal FOLD state for RL environment
minimal_fold = {
    # REQUIRED - the geometry
    "vertices_coords": [[x, y], ...],        # 2D coords (flat crease pattern)
    "edges_vertices": [[v_i, v_j], ...],      # edge connectivity
    "edges_assignment": ["M", "V", "B", ...], # fold type per edge

    # RECOMMENDED - explicit angle targets
    "edges_foldAngle": [-180, 180, 0, ...],   # target fold angles (degrees)

    # RECOMMENDED - explicit faces
    "faces_vertices": [[v0, v1, v2, ...], ...], # face polygons (CCW)

    # METADATA
    "frame_classes": ["creasePattern"],
    "frame_attributes": ["2D"],
}
```

### What We Can Skip

| Field | Skip? | Reason |
|-------|-------|--------|
| `faceOrders` | YES | Physics simulation handles layer ordering |
| `vertices_vertices` | YES | Recomputed from edges |
| `edges_faces` | YES | Recomputed from edges + faces |
| `faces_edges` | YES | Recomputed from faces + edges |
| `vertices_edges` | YES | Recomputed |
| `file_spec`, `file_creator` | YES | Metadata only |

### Python Data Structure for RL State

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OrigamiFOLDState:
    """Minimal FOLD state for RL environment."""

    # Core geometry
    vertices_coords: np.ndarray      # shape (num_vertices, 2) or (num_vertices, 3)
    edges_vertices: np.ndarray       # shape (num_edges, 2), dtype int
    edges_assignment: List[str]      # length num_edges, values in {"M","V","B","F","U"}

    # Optional but recommended
    edges_foldAngle: Optional[np.ndarray] = None  # shape (num_edges,), degrees
    faces_vertices: Optional[List[List[int]]] = None  # ragged list of vertex indices

    def to_fold_json(self) -> dict:
        """Export as FOLD JSON dict."""
        fold = {
            "file_spec": 1.1,
            "frame_classes": ["creasePattern"],
            "frame_attributes": ["2D"],
            "vertices_coords": self.vertices_coords.tolist(),
            "edges_vertices": self.edges_vertices.tolist(),
            "edges_assignment": self.edges_assignment,
        }
        if self.edges_foldAngle is not None:
            fold["edges_foldAngle"] = self.edges_foldAngle.tolist()
        if self.faces_vertices is not None:
            fold["faces_vertices"] = self.faces_vertices
        return fold

    @classmethod
    def from_fold_json(cls, data: dict) -> "OrigamiFOLDState":
        """Import from FOLD JSON dict."""
        coords = np.array(data["vertices_coords"], dtype=np.float64)
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])
        edges = np.array(data["edges_vertices"], dtype=np.int32)
        assignments = data.get("edges_assignment", ["U"] * len(edges))
        fold_angles = None
        if "edges_foldAngle" in data:
            fold_angles = np.array(data["edges_foldAngle"], dtype=np.float64)
        faces = data.get("faces_vertices", None)
        return cls(
            vertices_coords=coords,
            edges_vertices=edges,
            edges_assignment=assignments,
            edges_foldAngle=fold_angles,
            faces_vertices=faces,
        )

    @property
    def num_vertices(self) -> int:
        return len(self.vertices_coords)

    @property
    def num_edges(self) -> int:
        return len(self.edges_vertices)

    @property
    def num_mountain(self) -> int:
        return self.edges_assignment.count("M")

    @property
    def num_valley(self) -> int:
        return self.edges_assignment.count("V")

    @property
    def num_boundary(self) -> int:
        return self.edges_assignment.count("B")
```

### Observation Space Encoding for RL

```python
# How to encode FOLD state as a fixed-size observation for RL:

# Option 1: Adjacency + feature matrix (for GNN-based policies)
# Node features: [x, y, z, is_boundary, is_interior, degree]
# Edge features: [assignment_onehot(5), fold_angle, edge_length]

# Option 2: Flattened fixed-size grid (for MLP/CNN policies)
# Discretize the unit square into an NxN grid
# Each cell stores: [has_vertex, has_M_edge, has_V_edge, has_B_edge]

# Option 3: Variable-length sequence (for transformer policies)
# Sequence of edge tokens: [v_i, v_j, assignment, fold_angle]
```

---

## 9. Summary: How the Simulator Actually Works (End-to-End)

1. **Load FOLD file** -> parse `vertices_coords`, `edges_vertices`, `edges_assignment`, `faces_vertices`
2. **Triangulate** non-triangular faces via ear-clipping -> creates new internal edges
3. **Classify edges** -> fold creases (M/V with target angles), facet hinges (flat target, high stiffness), boundary (no constraint)
4. **Build GPU textures** -> pack node positions, beam params, crease params into RGBA float textures
5. **Run solver loop** (per frame, ~100 iterations):
   - Compute beam forces (axial springs prevent stretching)
   - Compute crease forces (rotational springs drive folding / enforce flatness)
   - Verlet integration with damping to update positions
6. **Compute strain** -> per-edge engineering strain = |L - L0| / L0, averaged per face
7. **Render** -> Three.js mesh colored by strain (blue=zero, red=max)
8. **User adjusts `foldPercent`** (0 to 1) -> target angles scale linearly -> solver re-converges

### Key Numerical Details

- **Timestep**: dt = 0.02 (small for stability)
- **Damping**: 0.1 (overdamped to reach equilibrium quickly)
- **Iterations per frame**: 100 (enough for incremental convergence)
- **Stiffness ratio**: axial >> facet > fold (prevents stretching while allowing controlled folding)
- **Convergence criterion**: total kinetic energy < threshold

### For Our RL Environment

The critical takeaway: **we do NOT need GPU shaders**. For an RL training loop, a NumPy/JAX port of the bar-and-hinge solver is sufficient:

```python
# Pseudocode for our NumPy port:

def simulate_fold(fold_state, fold_percent, n_steps=1000):
    """Simulate folding and return final positions + strain."""
    pos = fold_state.vertices_coords.copy()  # (N, 3)
    last_pos = pos.copy()
    dt = 0.02
    damping = 0.1

    for step in range(n_steps):
        forces = np.zeros_like(pos)

        # Beam forces (vectorized over all edges)
        forces += compute_beam_forces(pos, edges, rest_lengths, k_axial)

        # Crease forces (vectorized over all creases)
        forces += compute_crease_forces(pos, creases, target_angles * fold_percent,
                                         k_fold, k_facet)

        # Verlet integration
        new_pos = pos + (1 - damping) * (pos - last_pos) + forces * dt**2
        last_pos = pos
        pos = new_pos

    strain = compute_strain(pos, edges, rest_lengths)
    return pos, strain
```

This gives us the same physics as OrigamiSimulator but in Python, suitable for vectorized RL training with JAX/NumPy.
