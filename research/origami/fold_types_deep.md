# Origami Fold Types, Bases, and Operations: Complete Action Space for RL Environment

> Deep research compilation for building an origami reinforcement learning environment.
> Covers: primitive operations, fold taxonomy, bases, crane sequence, tessellation patterns,
> Huzita-Justin axioms, and complexity analysis.

---

## Table of Contents

1. [Huzita-Justin Axioms (The Primitive Action Space)](#1-huzita-justin-axioms---the-primitive-action-space)
2. [All Fold Types (Compound Operations)](#2-all-fold-types---compound-operations)
3. [Origami Bases (Starting Configurations)](#3-origami-bases---starting-configurations)
4. [Crane (Tsuru) Fold Sequence](#4-crane-tsuru-complete-fold-sequence)
5. [Compression/Packing Origami Patterns](#5-compressionpacking-origami-patterns)
6. [Yoshizawa-Randlett Notation System](#6-yoshizawa-randlett-notation-system)
7. [Fold Sequence Complexity](#7-fold-sequence-complexity)
8. [Flat-Foldability Theorems](#8-flat-foldability-theorems)
9. [RL Environment Design Implications](#9-rl-environment-design-implications)

---

## 1. Huzita-Justin Axioms -- The Primitive Action Space

These 7 axioms define ALL possible single-fold operations. Any single fold you can make on a
piece of paper corresponds to exactly one of these axioms. They are the **true primitive
action space** for origami.

First discovered by Jacques Justin (1986), rediscovered by Humiaki Huzita (1991), with Axiom 7
found by Koshiro Hatori (2001) and Robert J. Lang independently.

### Axiom 1: Fold Through Two Points
- **Input**: Two distinct points p1 and p2
- **Output**: A unique fold (crease line) that passes through both points
- **Geometric equivalent**: Drawing a line through two points (straightedge operation)
- **Solutions**: Exactly 1
- **Parameters**: `(p1_x, p1_y, p2_x, p2_y)`

### Axiom 2: Point-to-Point Fold
- **Input**: Two distinct points p1 and p2
- **Output**: A unique fold that places p1 onto p2
- **Geometric equivalent**: Constructing the perpendicular bisector of the segment p1-p2
- **Solutions**: Exactly 1
- **Parameters**: `(p1_x, p1_y, p2_x, p2_y)`

### Axiom 3: Line-to-Line Fold
- **Input**: Two lines l1 and l2
- **Output**: A fold that places l1 onto l2
- **Geometric equivalent**: Bisecting the angle between two lines
- **Solutions**: 1 if lines are parallel (perpendicular bisector), 2 if lines intersect (two angle bisectors)
- **Parameters**: `(l1_point, l1_direction, l2_point, l2_direction)`

### Axiom 4: Fold Through Point Perpendicular to Line
- **Input**: A point p1 and a line l1
- **Output**: A unique fold perpendicular to l1 that passes through p1
- **Geometric equivalent**: Constructing a perpendicular through a point
- **Solutions**: Exactly 1
- **Parameters**: `(p1_x, p1_y, l1_point, l1_direction)`

### Axiom 5: Point-to-Line Through Point
- **Input**: Two points p1, p2 and a line l1
- **Output**: A fold that places p1 onto l1 and passes through p2
- **Geometric equivalent**: Finding a tangent to a parabola from an external point (solves quadratic)
- **Solutions**: 0, 1, or 2
- **Parameters**: `(p1_x, p1_y, p2_x, p2_y, l1_point, l1_direction)`

### Axiom 6: Two Points onto Two Lines (The Beloch Fold)
- **Input**: Two points p1, p2 and two lines l1, l2
- **Output**: A fold that simultaneously places p1 onto l1 AND p2 onto l2
- **Geometric equivalent**: Finding a common tangent to two parabolas (solves cubic equations!)
- **Solutions**: 0, 1, 2, or 3
- **Parameters**: `(p1_x, p1_y, p2_x, p2_y, l1_point, l1_direction, l2_point, l2_direction)`
- **Significance**: This is what gives origami more power than compass-and-straightedge. It can
  trisect angles and double cubes (both impossible with compass/straightedge).
  Named after Margherita Beloch, who showed in 1936 that this fold solves general cubic equations.

### Axiom 7: Point-to-Line Perpendicular to Line
- **Input**: One point p and two lines l1, l2
- **Output**: A fold that places p onto l1 and is perpendicular to l2
- **Geometric equivalent**: Constructing a perpendicular to a line that maps a point onto another line
- **Solutions**: 0 or 1
- **Parameters**: `(p_x, p_y, l1_point, l1_direction, l2_point, l2_direction)`

### Summary Table: Axiom Action Parameters

| Axiom | Inputs | Max Solutions | Solves | Key Use |
|-------|--------|---------------|--------|---------|
| O1 | 2 points | 1 | Linear | Line through two points |
| O2 | 2 points | 1 | Linear | Perpendicular bisector |
| O3 | 2 lines | 2 | Linear | Angle bisector |
| O4 | 1 point + 1 line | 1 | Linear | Perpendicular through point |
| O5 | 2 points + 1 line | 2 | Quadratic | Point onto line via point |
| O6 | 2 points + 2 lines | 3 | **Cubic** | Simultaneous alignment |
| O7 | 1 point + 2 lines | 1 | Quadratic | Perpendicular alignment |

### Mathematical Power

- **Compass + straightedge**: Solves up to degree-2 (quadratic) equations
- **Origami (Axioms 1-7)**: Solves up to degree-3 (cubic) equations
- **Multi-fold origami** (simultaneous folds): Can solve higher-degree equations

---

## 2. All Fold Types -- Compound Operations

While the Huzita-Justin axioms are the mathematical primitives, origami practice uses
**compound fold types** -- named sequences that combine one or more axiom applications
and layer manipulations. These are the operations an origami folder actually thinks in.

### 2.1 Valley Fold (Tani-ori)

**The most fundamental fold.**

- **Geometry**: Paper is folded toward the viewer along a straight crease line. Creates a
  V-shaped cross-section (concave when viewed from above).
- **Dihedral angle**: gamma > 0 (positive fold angle between face normals)
- **Flat-folded state**: gamma = +pi (180 degrees)
- **Notation**: Dashed line (---) with filled arrowhead showing direction of motion
- **Parameters**:
  - `fold_line`: defined by a point and direction vector (or two points)
  - `fold_angle`: 0 to pi (typically pi for flat fold)
  - `layers_affected`: which layers of paper are being folded
- **Axiom mapping**: Corresponds to any of Axioms 1-7 depending on how the fold line is determined
- **Decomposition**: Single atomic operation

### 2.2 Mountain Fold (Yama-ori)

**The complement of valley fold.**

- **Geometry**: Paper is folded away from the viewer along a straight crease line. Creates an
  inverted-V cross-section (convex when viewed from above).
- **Dihedral angle**: gamma < 0 (negative fold angle)
- **Flat-folded state**: gamma = -pi (-180 degrees)
- **Notation**: Dot-dash line (-.-.-) with hollow single-sided arrowhead
- **Parameters**: Same as valley fold
- **Relationship to valley**: A mountain fold is geometrically identical to a valley fold viewed
  from the other side. Flipping the paper converts all mountains to valleys and vice versa.
- **Decomposition**: Single atomic operation (equivalent to: turn paper over + valley fold + turn back)

### 2.3 Inside Reverse Fold

**Used extensively for heads, tails, beaks, and feet in animal models.**

- **Geometry**: A pointed flap (at least 2 layers) is opened and the tip is pushed inward,
  reversing the direction of the central crease while creating two new creases on either side.
- **What happens**: The mountain fold along the spine of the flap is reversed (becomes valley)
  between two new valley folds that diverge from a point on the original spine.
- **Parameters**:
  - `flap_to_reverse`: which flap/point
  - `fold_line_angle`: angle of the new crease relative to the flap spine
  - `fold_depth`: how far down the spine the reversal point sits
- **Decomposition**: 2 valley folds + 1 mountain-to-valley conversion = 3 crease changes
- **Prerequisite state**: Requires an existing folded flap with a central crease

### 2.4 Outside Reverse Fold

**The mirror complement of inside reverse fold.**

- **Geometry**: A pointed flap is opened and the tip is wrapped around the outside, reversing
  the central crease while creating two new creases.
- **What happens**: The valley fold along the spine becomes mountain, and two mountain folds
  diverge from a point on the spine. The flap wraps over the outside.
- **Parameters**: Same as inside reverse fold
- **Decomposition**: 2 mountain folds + 1 valley-to-mountain conversion = 3 crease changes
- **Notation**: Mountain fold lines on near layer, valley on far layer, push arrow

### 2.5 Squash Fold

**Opens a flap and flattens it symmetrically.**

- **Geometry**: A flap with at least 2 layers has its closed edge opened. A radial fold from
  the closed point bisects the flap. The flap is pressed flat, creating two adjacent flaps
  from one.
- **What happens**: One existing crease becomes a fold, a new crease bisects the original flap,
  and the flap is flattened into a diamond/square shape.
- **Parameters**:
  - `flap_to_squash`: which flap
  - `bisecting_line`: the line that will become the new center (typically the symmetry axis)
  - `direction`: which side to flatten toward
- **Decomposition**: 1 new valley fold (bisector) + opening + flattening = compound operation
- **Common use**: Preliminary base -> bird base transition; creating diamond shapes from triangular flaps

### 2.6 Petal Fold

**The signature fold of the bird base. Creates a long, narrow flap from a wider one.**

- **Geometry**: Starting with two connected flaps (each with 2+ layers), two radial folds
  are made from the open point so that the open edges lie along a reference crease. The top
  layer is lifted and folded upward while the sides collapse inward.
- **What happens**: A point is elongated by folding two edges to a center line, then lifting
  the top layer while the creases collapse. Essentially two symmetrically-placed rabbit ears
  executed simultaneously.
- **Parameters**:
  - `reference_crease`: the center line to fold edges toward
  - `flap_edges`: the two edges that will be brought to the center
  - `lift_direction`: up or down
- **Decomposition**: 2 valley folds (edges to center) + 1 valley fold (top layer lift) +
  2 mountain folds (collapse) = 5 simultaneous crease changes
- **Common use**: Central operation in creating the bird base from preliminary base

### 2.7 Rabbit Ear Fold

**Creates a triangular flap that stands up from the surface.**

- **Geometry**: Starting with a triangular region, fold the angle bisectors from two corners
  on the same side of a reference diagonal. The resulting triangular flap is folded flat to
  one side.
- **What happens**: Three creases meet at a point -- two valley folds (bisectors) and one
  mountain fold (the ridge of the raised triangle). The excess paper forms a small triangular
  flap.
- **Parameters**:
  - `reference_crease`: the diagonal or edge used as the base
  - `bisector_1_angle`: angle of first bisector fold
  - `bisector_2_angle`: angle of second bisector fold
  - `flap_direction`: which side the resulting flap folds to
- **Decomposition**: 2 valley folds + 1 mountain fold = 3 simultaneous creases
- **Common use**: Fish base construction, creating narrow triangular points

### 2.8 Sink Fold (Three Variants)

**Pushes a point or corner into the interior of the model. The most difficult standard fold.**

#### 2.8a Open Sink
- **Geometry**: A corner point is pushed inward, and the paper around it is opened and
  reflattened. The surrounding paper forms a waterbomb-base-like configuration around
  the sunken area.
- **What happens**: Creases around the point are reversed (mountains become valleys and
  vice versa). The paper can be opened flat during the process.
- **Parameters**:
  - `point_to_sink`: which vertex/corner
  - `sink_depth`: how far to push in (defined by a crease line around the point)
  - `crease_pattern`: the polygon of creases that defines the sink boundary
- **Decomposition**: Multiple crease reversals; the paper is opened, re-creased, re-collapsed
- **Difficulty**: Intermediate

#### 2.8b Closed Sink
- **Geometry**: Same goal as open sink, but the paper layers cannot be separated during the
  operation. The corner is pushed in while the paper remains closed.
- **What happens**: The point inverts without opening. Paper layers become deeply intertwined.
  All axial creases around the point become mountain folds (key difference from open sink).
- **Parameters**: Same as open sink
- **Decomposition**: Simultaneous reversal of multiple creases without opening
- **Difficulty**: Advanced -- requires significant force; layers lock together

#### 2.8c Spread Sink (Spread Squash)
- **Geometry**: A closed flap or point is pushed in while the surrounding paper spreads
  outward and flattens. The sinked analog of the squash fold.
- **What happens**: Creates a wide, flat area around the flap's base instead of a long point
- **Parameters**: Same as open sink + spread direction
- **Decomposition**: Sink + flatten/spread
- **Difficulty**: Advanced

### 2.9 Unsink Fold (Two Variants)

**The inverse of sink -- pops a sunken point back out.**

#### 2.9a Open Unsink
- Makes a concave pocket convex without fully unfolding
- The opposite of an open sink

#### 2.9b Closed Unsink
- Inverts a closed sink without opening the paper
- Extremely difficult -- requires pulling (not pushing) hidden paper
- Involves simultaneously folding a locking flap hidden inside

### 2.10 Crimp Fold

**A reverse fold applied to an edge rather than a point.**

- **Geometry**: Opens a section of paper, applies a valley fold, then a mountain fold with
  some paper between them, creating a zigzag profile.
- **What happens**: Two parallel or near-parallel creases are created, with the paper between
  them forming a step. The edge changes direction.
- **Parameters**:
  - `crimp_location`: where along the edge
  - `fold_line_1`: first crease (valley)
  - `fold_line_2`: second crease (mountain)
  - `crimp_width`: distance between the two creases
  - `crimp_angle`: angle of direction change
- **Decomposition**: 1 valley fold + 1 mountain fold applied simultaneously to a multi-layer section
- **Relationship**: Similar to pleat fold but applied to a folded edge (multiple layers);
  like an inside reverse fold but without full reversal
- **Common use**: Creating feet, zigzag shapes, direction changes in legs

### 2.11 Pleat Fold (Accordion Fold)

**The simplest compound fold -- alternating valleys and mountains.**

- **Geometry**: A series of parallel or near-parallel alternating valley and mountain folds.
  Creates a zigzag/accordion profile.
- **What happens**: Paper forms parallel ridges and valleys, like a fan or accordion.
- **Parameters**:
  - `fold_lines[]`: array of parallel crease lines
  - `fold_types[]`: alternating valley/mountain assignments
  - `pleat_width`: distance between consecutive creases (can be uniform or varying)
  - `num_pleats`: number of folds
- **Decomposition**: n alternating valley + mountain folds
- **Common use**: Adding detail (pleats in clothing), creating segmented forms, fan shapes

### 2.12 Swivel Fold

**A loosely-defined fold where one flap "swivels" around a pivot point.**

- **Geometry**: A flap of paper rotates around a specific vertex or point while a connected
  flap or edge is dragged around that pivot. One fold inevitably causes another.
- **What happens**: Simultaneous folding of multiple connected regions around a pivot point.
  One fold "trails" another.
- **Parameters**:
  - `pivot_point`: the vertex around which paper swivels
  - `primary_fold_line`: the main fold being executed
  - `trailing_fold_line`: the induced fold
  - `swivel_angle`: angle of rotation
- **Decomposition**: 2+ simultaneous folds (one driving, one or more trailing)
- **Note**: Loosely defined; many different configurations qualify as "swivel folds"
- **Common use**: Shaping flaps, creating offset angles, bird tails

### Summary: Fold Type Taxonomy

```
ATOMIC FOLDS (single crease, corresponds to 1 axiom application):
  |-- Valley Fold (gamma > 0)
  |-- Mountain Fold (gamma < 0)

COMPOUND FOLDS (multiple simultaneous creases):
  |-- Reverse Folds
  |   |-- Inside Reverse Fold (3 crease changes)
  |   |-- Outside Reverse Fold (3 crease changes)
  |
  |-- Squash Fold (2-3 crease changes)
  |
  |-- Petal Fold (5 simultaneous crease changes)
  |
  |-- Rabbit Ear Fold (3 simultaneous creases)
  |
  |-- Sink Folds
  |   |-- Open Sink (multiple reversals, paper opens)
  |   |-- Closed Sink (multiple reversals, paper stays closed)
  |   |-- Spread Sink (sink + flatten)
  |
  |-- Unsink Folds
  |   |-- Open Unsink (reverse of open sink)
  |   |-- Closed Unsink (reverse of closed sink)
  |
  |-- Crimp Fold (2 simultaneous folds on multi-layer edge)
  |
  |-- Pleat Fold (n alternating valley/mountain folds)
  |
  |-- Swivel Fold (2+ coupled folds around pivot)

NON-FOLD OPERATIONS (manipulations):
  |-- Turn Over (flip paper)
  |-- Rotate (in-plane rotation)
  |-- Inflate / Open (spread layers apart, puff out)
  |-- Unfold (reverse a previous fold)
  |-- Cut (kirigami -- not standard origami)
```

### Fold Parameter Space

For an RL environment, each fold can be parameterized as:

```
Action = {
  fold_type:     enum[valley, mountain, reverse_in, reverse_out, squash, petal,
                      rabbit_ear, sink_open, sink_closed, spread_sink,
                      unsink_open, unsink_closed, crimp, pleat, swivel,
                      turn_over, rotate, inflate, unfold]

  -- For atomic folds (valley/mountain):
  fold_line:     (point: vec2, direction: vec2)  -- or (point1, point2)
  fold_angle:    float [0, pi]                   -- how far to fold
  layers:        bitset                          -- which layers to fold

  -- For compound folds, additional params:
  target_flap:   flap_id                         -- which flap to operate on
  depth:         float                           -- for reverse/sink: how deep
  angle:         float                           -- for reverse/crimp: fold angle
  direction:     enum[left, right, up, down]     -- fold direction preference
  width:         float                           -- for crimp/pleat: spacing

  -- For manipulation operations:
  rotation_angle: float                          -- for rotate
  axis:          enum[horizontal, vertical]      -- for turn_over
}
```

---

## 3. Origami Bases -- Starting Configurations

Bases are standard intermediate forms that serve as starting points for families of models.
**Yes, bases are shortcuts for common fold sequences.** They represent well-known crease
patterns that produce useful distributions of flaps and points.

### 3.1 Preliminary Base (Square Base)

- **Shape**: Multi-layered diamond/square standing on a corner
- **Flaps**: 4 flaps (2 front, 2 back)
- **Points**: 1 closed point at top, 4 open edges at bottom
- **Construction from flat square**:
  1. Valley fold in half horizontally (1 fold)
  2. Valley fold in half vertically (1 fold)
  3. Unfold both (0 folds -- return to flat with creases)
  4. Valley fold diagonally both ways (2 folds, unfold)
  5. Collapse: push sides in using existing creases (simultaneous collapse)
- **Total atomic folds**: ~4 creases + 1 collapse = ~5 operations
- **Crease pattern**: 2 perpendicular valley diagonals + 2 perpendicular mountain edge-bisectors
- **Key relationship**: Inverse of Waterbomb base (same creases, swapped mountain/valley)
- **Gateway to**: Bird base, Frog base, Flower base

### 3.2 Waterbomb Base

- **Shape**: Flat triangle with multiple layers
- **Flaps**: 4 flaps (2 front, 2 back)
- **Points**: 4 points at base, closed edges at sides
- **Construction from flat square**:
  1. Valley fold both diagonals (2 folds, unfold)
  2. Mountain fold horizontally and vertically (2 folds, unfold)
  3. Collapse into triangle form
- **Total atomic folds**: ~4 creases + 1 collapse = ~5 operations
- **Crease pattern**: 2 perpendicular mountain diagonals + 2 perpendicular valley edge-bisectors
- **Key relationship**: Inverse of Preliminary base (same creases, swapped mountain/valley)
- **Gateway to**: Waterbomb (balloon), waterbomb tessellations, some Frog base variants

### 3.3 Kite Base

- **Shape**: Kite-shaped flat form
- **Flaps**: 1 main point, 2 small triangular flaps
- **Construction from flat square**:
  1. Valley fold diagonal (1 fold, unfold -- reference crease)
  2. Valley fold two adjacent edges to lie on the diagonal (2 folds)
- **Total atomic folds**: 3 (simplest base)
- **Gateway to**: Simple animals (dog face, cat face), Kite -> Fish base progression

### 3.4 Fish Base

- **Shape**: Diamond shape with 4 points (two long, two short)
- **Flaps**: 4 points usable for fins/legs/petals
- **Construction from flat square**:
  1. Start with kite base (3 folds)
  2. Rabbit ear fold on one end (3 simultaneous creases)
  3. Rabbit ear fold on other end (3 simultaneous creases)
  4. Fold resulting flaps down (2 folds)
- **Total atomic folds**: ~8-11 operations
- **Crease pattern**: Two rabbit ears against diagonal reference creases on opposite corners
- **Gateway to**: Fish models, some flower models

### 3.5 Bird Base (Crane Base)

- **Shape**: Long diamond with 4 narrow flaps
- **Flaps**: 4 long, narrow flaps (2 front, 2 back)
- **Points**: All 4 original corners become elongated points
- **Construction from flat square**:
  1. Fold preliminary base (~5 operations)
  2. Kite-fold front flaps: fold left and right edges to center line (2 valley folds)
  3. Fold top triangle down over the kite folds (1 valley fold, then unfold)
  4. Unfold the kite folds (restore)
  5. Petal fold: lift bottom point up using existing creases (1 petal fold = ~5 crease changes)
  6. Turn over
  7. Repeat steps 2-5 on back side (mirror)
- **Total atomic folds**: ~18-22 operations from flat square
- **Key significance**: The most versatile and widely-used origami base
- **Gateway to**: Crane, many birds, dragon, horse, and hundreds of other models

### 3.6 Frog Base

- **Shape**: 4 long narrow flaps radiating from center (more symmetrical than bird base)
- **Flaps**: 4 long flaps + 4 shorter flaps
- **Construction from flat square**:
  1. Fold preliminary base (~5 operations)
  2. Squash fold each of the 4 flaps (4 squash folds)
  3. Petal fold each of the 4 resulting diamonds (4 petal folds)
- **Total atomic folds**: ~25-30 operations from flat square
- **Gateway to**: Frog, lily, iris, and other 4-legged/4-petal models

### 3.7 Windmill Base

- **Shape**: 4 triangular flaps arranged like a pinwheel
- **Flaps**: 4 flaps that can rotate in a windmill pattern
- **Construction from flat square**:
  1. Fold and unfold both diagonals and both midlines (4 creases)
  2. Fold all 4 edges to center (4 valley folds) -- this is the "blintz fold"
  3. Unfold
  4. Fold 2 opposite edges to center (2 valley folds)
  5. Pull out trapped corners and flatten (2 squash-like operations)
- **Total atomic folds**: ~12-15 operations
- **Gateway to**: Windmill, some modular units, windmill-derived models

### Base Relationship Map

```
Flat Square
  |
  +---> Kite Base ---------> Fish Base
  |        (3 folds)            (8-11 folds)
  |
  +---> Preliminary Base ---> Bird Base ------> [Crane, Birds, etc.]
  |        (5 folds)            (18-22 folds)
  |                        |
  |                        +-> Frog Base -----> [Frog, Lily, etc.]
  |                              (25-30 folds)
  |
  +---> Waterbomb Base -----> [Waterbomb/Balloon, etc.]
  |        (5 folds)
  |
  +---> Blintz Base --------> Windmill Base -> [Windmill, etc.]
  |        (8 folds)            (12-15 folds)
  |
  +---> Book Fold / Cupboard Fold / etc.
```

---

## 4. Crane (Tsuru) Complete Fold Sequence

The traditional origami crane (orizuru) is THE canonical origami model, and the best
benchmark for an RL environment. Here is the complete fold-by-fold sequence from flat
square to finished crane.

### Phase 1: Create Crease Pattern (Pre-creasing)

| Step | Operation | Fold Type | Fold Line | Result |
|------|-----------|-----------|-----------|--------|
| 1 | Fold square in half diagonally (corner to corner) | Valley fold | Main diagonal (bottom-left to top-right) | Triangle |
| 2 | Unfold | Unfold | -- | Square with diagonal crease |
| 3 | Fold in half on other diagonal | Valley fold | Other diagonal (bottom-right to top-left) | Triangle |
| 4 | Unfold | Unfold | -- | Square with X crease |
| 5 | Fold in half horizontally | Valley fold | Horizontal midline | Rectangle |
| 6 | Unfold | Unfold | -- | Square with X + horizontal crease |
| 7 | Fold in half vertically | Valley fold | Vertical midline | Rectangle |
| 8 | Unfold | Unfold | -- | Square with full crease pattern (X + cross) |

### Phase 2: Collapse into Preliminary Base

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 9 | Collapse: push left and right edges inward while folding top down | Simultaneous collapse | Diagonals become valley folds, midlines become mountain folds | Preliminary base (4-layer diamond) |

### Phase 3: Kite Folds (Front)

| Step | Operation | Fold Type | Fold Line | Result |
|------|-----------|-----------|-----------|--------|
| 10 | Fold left edge of top layer to center line | Valley fold | Left edge to center crease | Left kite flap |
| 11 | Fold right edge of top layer to center line | Valley fold | Right edge to center crease | Kite shape |
| 12 | Fold top triangle down over kite flaps | Valley fold | Horizontal line at top of kite flaps | Triangle folded over |
| 13 | Unfold step 12 | Unfold | -- | Kite shape with horizontal crease |
| 14 | Unfold steps 10-11 | Unfold | -- | Diamond with crease guides |

### Phase 4: Front Petal Fold

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 15 | Lift bottom point of top layer upward using existing creases; sides collapse inward | Petal fold | Bottom point lifts to top; left and right edges fold to center simultaneously | Front petal fold complete -- one long narrow flap pointing up |

### Phase 5: Repeat on Back

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 16 | Turn model over | Turn over | Flip along vertical axis | Back is now front |
| 17 | Fold left edge to center line | Valley fold | Left edge to center | Left kite flap |
| 18 | Fold right edge to center line | Valley fold | Right edge to center | Kite shape |
| 19 | Fold top triangle down | Valley fold | Horizontal line at top | Triangle folded over |
| 20 | Unfold step 19 | Unfold | -- | Kite with crease |
| 21 | Unfold steps 17-18 | Unfold | -- | Diamond with creases |
| 22 | Petal fold: lift bottom point up, collapse sides in | Petal fold | Same as step 15 on back | Bird base complete |

### Phase 6: Narrow the Legs

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 23 | Fold left flap (front layer) edge to center | Valley fold | Left edge to centerline | Narrower left flap |
| 24 | Fold right flap (front layer) edge to center | Valley fold | Right edge to centerline | Narrower right flap |
| 25 | Turn over | Turn over | -- | Back side |
| 26 | Fold left flap edge to center | Valley fold | Left edge to centerline | Narrower left flap (back) |
| 27 | Fold right flap edge to center | Valley fold | Right edge to centerline | Narrower right flap (back) |

### Phase 7: Form Neck and Tail

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 28 | Fold left bottom flap (front+back layers) upward along angle to form neck | Inside reverse fold | Fold line at ~60-70 degrees from vertical | Neck points upward at angle |
| 29 | Fold right bottom flap upward to form tail | Inside reverse fold | Fold line at ~60-70 degrees from vertical | Tail points upward |

### Phase 8: Form Head and Finish

| Step | Operation | Fold Type | Details | Result |
|------|-----------|-----------|---------|--------|
| 30 | Fold tip of neck downward to form head/beak | Inside reverse fold | Small fold near tip of neck, ~30 degrees | Head with beak |
| 31 | Gently pull wings apart from body and press bottom to create 3D body | Open/Inflate | Separate wing layers, crease body | Finished crane |

### Crane Fold Statistics

| Metric | Count |
|--------|-------|
| **Total numbered steps** | 31 |
| **Valley folds** | 14 |
| **Unfolds** | 6 |
| **Petal folds** | 2 (each = ~5 atomic crease changes) |
| **Inside reverse folds** | 3 (each = ~3 atomic crease changes) |
| **Turn over** | 2 |
| **Collapse** | 1 (= ~4 simultaneous crease changes) |
| **Open/inflate** | 1 |
| **Total atomic crease changes** | ~40-45 |
| **Unique fold types used** | 5 (valley, petal, inside reverse, collapse, inflate) |

---

## 5. Compression/Packing Origami Patterns

These are tessellation and pattern-based folds used in engineering, not traditional origami
art. Critical for understanding the space of possible flat-to-compact transformations.

### 5.1 Miura-ori Fold

- **Inventor**: Koryo Miura (1970)
- **Geometry**: Tessellation of parallelograms with alternating mountain/valley creases.
  In one direction, creases are straight lines with mirror-reflected parallelograms.
  In the other direction, creases zigzag, with parallelograms translated across creases.
- **Parameters**:
  - Parallelogram angle (alpha): the acute angle of the parallelogram, typically 55-85 degrees
  - Panel width and height
  - Number of panels in each direction
- **Rigid-foldable**: YES -- can be folded with completely rigid (non-bending) panels
- **Degrees of freedom**: 1 (single DOF per unit cell). The entire sheet deploys/compacts
  with a single motion.
- **Compression**: Folds flat in BOTH directions simultaneously. Unfolds by pulling opposite
  corners apart in a single motion.
- **Poisson's ratio**: Always NEGATIVE (auxetic material). When you pull it in one direction,
  it expands in the perpendicular direction too.
- **Compression ratio**: Depends on number of panels; theoretically approaches thickness-only
  when fully compressed. A sheet can compress to approximately (n_panels * t) where t is
  material thickness.
- **Applications**: Satellite solar panel arrays (Space Flyer Unit, 1995), metamaterials,
  deployable shelters, foldable maps
- **Crease pattern**: Regular grid of mountain and valley folds at specific angles

### 5.2 Waterbomb Tessellation

- **Geometry**: Repeating waterbomb base units tessellated across a surface. Each unit has
  degree-6 vertices with alternating mountain/valley folds.
- **Parameters**:
  - Base unit size
  - Grid dimensions
  - Mountain/valley assignment per crease
- **Rigid-foldable**: Partially (depends on configuration)
- **Compression ratio**: ~3:1 (switching ratio)
- **Properties**: Can form curved surfaces; used in origami-inspired robots; basis for
  many adaptive structures
- **Applications**: Soft robotics, adaptive surfaces, energy absorption

### 5.3 Ron Resch Pattern

- **Inventor**: Ron Resch (patent 1968)
- **Geometry**: Folded equilateral triangles arranged in periodic radial formation.
  Six triangles around a central point compress together to form a flat-surfaced hexagon.
- **Parameters**:
  - Triangle size
  - Grid pattern (triangular or square variant)
  - Fold depth
- **Rigid-foldable**: No (requires panel bending)
- **Compression ratio**: Up to 50:1 theoretical, ~6:1 practical (limited by creep)
- **Specific elastic compression modulus**: 15-365 MPa/kg for standard designs;
  novel variants reach 594-926 MPa/kg
- **Properties**: Excellent energy absorption; creates two flat surfaces (top and bottom)
  with triangular columns between them
- **Applications**: Impact damping, packaging, architectural surfaces, sandwich panel cores

### 5.4 Flasher Pattern

- **Geometry**: Central polygon (typically hexagon or octagon) whose edges connect to
  extending panels, each with identical crease geometry. Panels spiral around the
  central polygon when folded.
- **Parameters**:
  - Central polygon type and size
  - Number of extending panel rings
  - Spiral angle
  - Panel thickness (critical for rigid implementations)
- **Compression ratio**: Stowed-to-deployed diameter ratio of ~9.2:1 for typical designs;
  area ratio much higher (square of diameter ratio, ~85:1)
- **Rigid-foldable**: With modifications (membrane hinges or diagonal folding to accommodate
  panel thickness)
- **Degrees of freedom**: 1 (deploys/compacts with single rotational motion)
- **Applications**: Space solar arrays (250+ kW), solar sails, deployable reflectors,
  NASA deployable structures
- **Key challenge**: Accommodating real material thickness; zero-thickness models don't
  directly translate

### 5.5 Kresling Pattern

- **Inventor**: Biruta Kresling
- **Geometry**: Tessellated triangles forming a thin cylindrical structure. Diagonal fold
  lines create helical creases around the cylinder.
- **Parameters**:
  - Number of polygon sides (n, typically 6-8)
  - Cylinder radius
  - Triangle aspect ratio
  - Folding angles (beta and gamma relative to horizontal)
- **Rigid-foldable**: NO (requires panel bending/deformation for compression)
- **Degrees of freedom**: Coupled -- compression induces twist (axial-torsional coupling)
- **Compression ratio**: ~1.5:1 switching ratio
- **Bistability**: Key property -- can snap between extended and compressed states.
  Geometrical parameters can be tuned for mono-, bi-, or multistability.
- **Properties**: Compression-twist coupling, negative stiffness regions, energy storage
- **Applications**: Soft robots, mechanical metamaterials, deployable tubes, vibration isolation,
  energy harvesting

### 5.6 Yoshimura Pattern

- **Geometry**: Diamond/rhombus pattern around a cylinder, creating a Yoshimura buckle
- **Rigid-foldable**: NO
- **Properties**: Natural buckling pattern of thin cylindrical shells under axial compression
- **Applications**: Crushable energy absorbers, deployable structures

### Pattern Comparison Table

| Pattern | Rigid-Foldable | DOF | Compression Ratio | Bistable | Key Property |
|---------|---------------|-----|-------------------|----------|--------------|
| Miura-ori | Yes | 1 | High (thickness-limited) | No | Auxetic (negative Poisson's ratio) |
| Waterbomb | Partial | Multi | ~3:1 | No | Curved surfaces possible |
| Ron Resch | No | Multi | ~6-50:1 | No | Excellent energy absorption |
| Flasher | Modified yes | 1 | ~9.2:1 (diameter) | No | Spiral deployment |
| Kresling | No | Coupled | ~1.5:1 | YES | Compression-twist coupling |
| Yoshimura | No | Multi | Moderate | No | Natural buckling mode |

---

## 6. Yoshizawa-Randlett Notation System

The standard international notation system for origami diagrams, created by Akira Yoshizawa
(1954) and formalized by Samuel Randlett and Robert Harbin (1961).

### 6.1 Line Types

| Line Style | Meaning | Description |
|------------|---------|-------------|
| Dashed line `------` | Valley fold | Paper folds toward you |
| Dot-dash line `-.-.-.` | Mountain fold | Paper folds away from you |
| Thin solid line | Existing crease | A crease already made in a previous step |
| Thick solid line | Paper edge | The boundary of the paper |
| Dotted line `......` | X-ray line | Hidden edge or crease visible through layers |

### 6.2 Arrow Types

| Arrow | Meaning | Visual |
|-------|---------|--------|
| Filled/split arrowhead, curved stem | Valley fold (fold toward you) | Shows rotation path of paper |
| Hollow single-sided arrowhead | Mountain fold (fold away) | Hooks behind moving flap |
| Double-sided hollow arrowheads | Fold and unfold | Make crease, then return |
| Double-sided hollow (on existing fold) | Unfold only | Reverse an existing fold |
| Arrow with loop in stem | Turn paper over | Flip horizontally or vertically |
| Hollow cleft-tailed arrow | Push / Apply pressure | Used for sinks, inflations, reverses |
| Arrow curving around layers | Hooking arrow | Shows which specific layers move |
| Arrow touching down multiple times | Fold over and over | Repeated folding |

### 6.3 Special Symbols

| Symbol | Meaning |
|--------|---------|
| Circle with fraction + curved arrows | Rotate in plane (e.g., 1/4 turn) |
| Box with step range + count | Repeat steps (e.g., "steps 5-8, x4") |
| Open circle on paper | Hold here / grip point |
| Perpendicular marks on edges | Equal distances |
| Arc marks on angles | Equal angles |
| Heavy circle around area | Cut-away view (shows hidden layers) |
| Zigzag line (edge view) | Crimp/pleat layer diagram |
| Stylized eye symbol | Next view location/angle |
| Scissors symbol | Cut (kirigami) |
| Puff/cloud symbol | Inflate / blow air |

### 6.4 Compound Fold Notation

| Fold Type | Notation Components |
|-----------|-------------------|
| Inside reverse | Mountain line (near layer) + valley line (far layer) + push arrow + valley motion arrow |
| Outside reverse | Paired mountain/valley lines + hooked arrows showing opposite directions |
| Squash | Valley fold line + push arrow + opening indicator |
| Petal | 2 mountain lines + 1 valley line + lift arrow |
| Rabbit ear | 3 valley lines (bisectors) + 1 mountain line + flap direction arrow |
| Sink | Mountain fold line + push arrow (hollow for open sink; dots at corners for closed) |
| Crimp | Edge-view zigzag diagram + paired fold lines |

---

## 7. Fold Sequence Complexity

### 7.1 Complexity Classification (OrigamiUSA Standard)

| Level | Steps | Time | Fold Types Used |
|-------|-------|------|-----------------|
| Simple | 1-16 | 5-15 min | Valley and mountain folds only |
| Low Intermediate | 10-20 | 10-20 min | + reverse folds |
| Intermediate | 17-30 | 15-40 min | + squash, petal folds |
| High Intermediate | 25-50 | 20-60 min | + sink folds, complex shaping |
| Complex | 40-100 | 1-4 hours | All fold types, multi-step collapses |
| Super Complex | 100-1000+ | Hours to weeks | Nested sinks, 10+ simultaneous creases |

### 7.2 Model Complexity Examples

| Model | Approximate Folds | Level | Key Operations |
|-------|-------------------|-------|----------------|
| Paper airplane | 5-7 | Simple | Valley folds only |
| Fortune teller | 8 | Simple | Valley folds, turn over |
| Waterbomb (balloon) | 10-12 | Simple | Valley, mountain, inflate |
| Jumping frog | 15-18 | Low Intermediate | Valley, mountain, pleat |
| Crane (tsuru) | 30-31 | Intermediate | Valley, petal, inside reverse, inflate |
| Lily/Iris | 35-45 | Intermediate | Valley, petal, curl |
| Traditional frog | 40-50 | High Intermediate | Valley, petal, sink, reverse |
| Dragon | 60-100 | Complex | All fold types |
| Kawasaki's rose | 40-60 | Complex | Twist folds, curl, 3D shaping |
| Kamiya's Ryujin 3.5 (dragon) | 1000+ | Super Complex | Everything, including nested sinks |

### 7.3 Complexity Scaling

The relationship between fold count and model complexity is **super-linear**:

- Each fold adds to the **state space** exponentially (more layers, more possible future folds)
- Compound folds (petal, sink) each involve 3-10 atomic crease changes
- **Layer count** grows exponentially: after n flat folds, up to 2^n layers in some regions
- **Branching factor**: At each step, the number of possible next folds depends on the
  current crease pattern, number of exposed edges, and layer configuration
- **Collapse operations**: Some steps require 4-10+ creases to change simultaneously,
  making them hard to decompose into atomic actions for RL

### 7.4 Computational Complexity Results

- **Flat-foldability** (single vertex): Polynomial -- checkable via Kawasaki's and Maekawa's theorems
- **Flat-foldability** (multi-vertex, global): **NP-complete** (proven by Bern and Hayes, 1996)
- **Simple fold sequences**: Can be verified in polynomial time
- **Optimal fold sequences** (minimum folds to reach a target): Unknown complexity class,
  likely intractable for complex models

---

## 8. Flat-Foldability Theorems

These theorems constrain what configurations are VALID in the state space -- they define the
physics/rules of the environment.

### 8.1 Kawasaki's Theorem (Kawasaki-Justin Theorem)

**At a single flat-foldable vertex, the alternating sum of consecutive sector angles equals zero.**

```
alpha_1 - alpha_2 + alpha_3 - alpha_4 + ... = 0
```

Equivalently: if you partition the angles around a vertex into two alternating subsets,
each subset sums to exactly 180 degrees.

- **Applies to**: Single vertex, flat-foldable patterns
- **Necessary and sufficient**: For single-vertex flat foldability (combined with Maekawa's)
- **Discovered by**: Kawasaki, Robertson, and Justin (late 1970s - early 1980s)

### 8.2 Maekawa's Theorem (Maekawa-Justin Theorem)

**At every flat-foldable vertex, the number of mountain folds and valley folds differ by exactly 2.**

```
|M - V| = 2
```

Where M = number of mountain folds and V = number of valley folds at that vertex.

- **Corollary**: The total number of creases at a flat-foldable vertex must be even
- **Corollary**: Since M - V = +/- 2, and M + V = total creases, you can derive M and V
  given the total number of creases

### 8.3 Additional Constraints

- **Two-colorability**: A flat-foldable crease pattern divides the paper into regions that
  can be two-colored (like a checkerboard) such that regions sharing a crease get different colors
- **No self-intersection**: Paper cannot pass through itself during folding
- **Layer ordering**: At any point in a flat-folded model, the layers must have a consistent
  stacking order that respects fold connectivity
- **Global flat-foldability**: NP-complete for general crease patterns (Bern & Hayes, 1996)

### 8.4 Implications for RL Environment

These theorems define the **validity constraints** for the environment:
- After each fold action, the resulting crease pattern must satisfy Kawasaki's and Maekawa's
  theorems at every vertex for flat foldability
- The layer ordering must remain consistent (no self-intersection)
- Invalid actions (violating these constraints) should be either prevented or penalized

---

## 9. RL Environment Design Implications

### 9.1 Action Space Options

Based on the research, there are three natural levels for defining the action space:

#### Option A: Axiom-Level (Pure Primitives)
```
Action = HuzitaJustinAxiom(axiom_number, input_points, input_lines)
```
- **Size**: 7 axiom types x continuous parameters = continuous action space
- **Pros**: Mathematically complete, provably covers all possible single folds
- **Cons**: Very low-level; compound folds like petal fold require many steps;
  the agent must discover compound operations on its own

#### Option B: Named Fold Level (Origami Operations)
```
Action = FoldOperation(fold_type, target, parameters)
```
- **Size**: ~15-19 fold types x continuous parameters
- **Pros**: Matches how humans think about origami; compound folds are single actions;
  faster learning due to higher-level primitives
- **Cons**: Not mathematically minimal; some folds are hard to parameterize (sink folds);
  harder to implement a general simulator

#### Option C: Hybrid (Recommended)
```
Action = {
  level: [axiom | compound]
  if axiom: (axiom_id, params...)
  if compound: (fold_type, flap_id, params...)
}
```
- **Pros**: Agent can use either low-level or high-level actions; macro-actions speed up
  learning while atomic actions enable novel folds
- **Cons**: Larger action space; need to implement both levels

### 9.2 State Representation

The state needs to capture:
1. **Crease pattern**: Graph of vertices and creases with mountain/valley assignments
2. **Layer ordering**: At every point, which layers are on top
3. **3D configuration**: Current fold angles (dihedral angles at each crease)
4. **Paper boundary**: The current outline of the folded paper

### 9.3 Reward Shaping Considerations

- **Target matching**: Compare current crease pattern to target model's crease pattern
- **Base achievement**: Intermediate rewards for reaching known bases
- **Fold validity**: Penalty for invalid folds (violating Kawasaki/Maekawa)
- **Efficiency**: Bonus for fewer total folds
- **Symmetry**: Reward for symmetric fold patterns (most models are symmetric)

### 9.4 Key Challenges

1. **Continuous action space**: Fold lines are defined by continuous parameters (position, angle)
2. **Variable-length sequences**: Different models need different numbers of folds (5-1000+)
3. **State explosion**: Layer count grows exponentially with folds
4. **Physical constraints**: Paper cannot self-intersect; fold validity is non-trivial to check
5. **3D reasoning**: Many operations (reverse folds, sinks) require reasoning about 3D geometry
   even though the result is flat
6. **Simultaneous creases**: Compound folds change 3-10 creases at once -- this is hard to
   decompose for a step-by-step environment

### 9.5 Suggested Starting Point

For a first implementation, consider:
- **Action space**: Valley fold + Mountain fold + Turn over + Unfold (4 operations with
  continuous fold-line parameters)
- **Target**: Simple models only (paper airplane, fortune teller, waterbomb)
- **State**: 2D crease pattern + layer count map
- **Expansion path**: Add reverse fold -> squash fold -> petal fold -> sink fold as the
  agent masters simpler operations

---

## Sources

- [Huzita-Justin Axioms - Robert J. Lang](https://langorigami.com/article/huzita-justin-axioms/)
- [Huzita-Hatori Axioms - Wikipedia](https://en.wikipedia.org/wiki/Huzita%E2%80%93Hatori_axioms)
- [The Huzita-Justin Axioms - Origami Math](https://orimath.wordpress.com/2021/08/02/the-huzita-hatori-axioms/)
- [One, Two, and Multi-Fold Origami Axioms - Alperin & Lang](https://langorigami.com/wp-content/uploads/2015/09/o4_multifold_axioms.pdf)
- [Origami Diagramming Conventions - Robert J. Lang](https://langorigami.com/article/origami-diagramming-conventions/)
- [Yoshizawa-Randlett System - Wikipedia](https://en.wikipedia.org/wiki/Yoshizawa%E2%80%93Randlett_system)
- [Fold Hierarchy and Origin of Origami Symbols - British Origami](https://www.britishorigami.org/cp-lister-list/fold-hierarchy-and-origin-of-origami-symbols/)
- [Origami Bases - Wikibooks](https://en.wikibooks.org/wiki/Origami/Techniques/Model_bases)
- [Origami Techniques Practice - Wikibooks](https://en.wikibooks.org/wiki/Origami/Techniques/Practice)
- [Kawasaki's Theorem - Wikipedia](https://en.wikipedia.org/wiki/Kawasaki's_theorem)
- [Maekawa's Theorem - Wikipedia](https://en.wikipedia.org/wiki/Maekawa's_theorem)
- [Flat Foldability - Abrashi Origami School](https://abrashiorigami.com/maekawa-justin-and-kawasaki-justin-theorems/)
- [Mathematics of Paper Folding - Wikipedia](https://en.wikipedia.org/wiki/Mathematics_of_paper_folding)
- [Miura Fold - Wikipedia](https://en.wikipedia.org/wiki/Miura_fold)
- [Geometry of Miura-folded Metamaterials - PNAS](https://www.pnas.org/doi/10.1073/pnas.1217998110)
- [Origami Engineering - Misseroni et al.](https://www.daraio.caltech.edu/publications/Misseroni_et_at_2024.pdf)
- [Flasher Deployable Arrays - NASA](https://ntrs.nasa.gov/citations/20150004060)
- [Analysis of Origami Flasher-Inspired Structures - MIT](https://dspace.mit.edu/bitstream/handle/1721.1/156648/bai-janebai-sb-meche-2024-thesis.pdf)
- [Kresling Origami Mechanics - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022509624000966)
- [Kresling Bistable Behavior - Cai & Deng](https://www.researchgate.net/publication/276173838_Bistable_Behavior_of_the_Cylindrical_Origami_Structure_With_Kresling_Pattern)
- [Freeform Origami Tessellations - Tomohiro Tachi](https://origami.c.u-tokyo.ac.jp/~tachi/cg/FreeformOrigamiTessellationsTachi2013ASME.pdf)
- [OrigamiUSA Difficulty Guidelines](https://origamiusa.org/difficulty)
- [Origami Complexity Framework - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2095034921000489)
- [Sink Fold - Abrashi Origami School](https://abrashiorigami.com/sink-fold/)
- [Swivel Fold - Abrashi Origami School](https://abrashiorigami.com/swivel-fold/)
- [Petal Fold - Abrashi Origami School](https://abrashiorigami.com/petal-fold/)
- [Rabbit Ear - Origami Book](https://rabbitear.org/book/origami.html)
- [From Fold to Function: Dynamic Origami Simulation](https://arxiv.org/html/2511.10580)
- [Automating Rigid Origami Design - IJCAI](https://www.ijcai.org/proceedings/2023/0645.pdf)
- [OrigamiSpace: Benchmarking LLMs in Origami](https://arxiv.org/html/2511.18450v1)
- [Valley and Mountain Folds - British Origami](https://www.britishorigami.org/cp-resource/valley-mountain-folds/)
- [Origami Crane Tutorial - Origami.me](https://origami.me/crane/)
- [5 Essential Origami Bases - OrigamiZen](https://origamizen.com/5-essential-origami-bases-every-folder-should-master/)
