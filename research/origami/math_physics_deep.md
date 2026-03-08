# Deep Research: Mathematics, Physics, and Engineering Science of Origami

> Theoretical foundations for building an origami simulation engine.
> Compiled from academic literature, computational geometry research, and engineering publications.

---

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Mechanics / Physics of Folding](#2-mechanics--physics-of-folding)
3. [Computational Origami Theory](#3-computational-origami-theory)
4. [Engineering Metrics](#4-engineering-metrics)
5. [Key References and Sources](#5-key-references-and-sources)

---

## 1. Mathematical Foundations

### 1.1 Flat-Foldability: What Makes a Crease Pattern Flat-Foldable?

A crease pattern is **flat-foldable** if it can be folded from a flat sheet of paper into a completely flat (2D) final state. This is the central question of combinatorial origami. A flat fold is formally modeled as a **piecewise isometric map** from a 2D region to itself, equipped with:

1. A **crease pattern** (a planar graph embedded in the paper).
2. A **mountain-valley (MV) assignment** labeling each crease as mountain (M) or valley (V).
3. A **layer ordering** that specifies which regions of paper lie above which others in the folded state.

For a crease pattern to be flat-foldable, ALL of the following must hold simultaneously:

- **Local flat-foldability at every vertex** (Kawasaki-Justin condition).
- **Valid MV assignment** consistent with Maekawa-Justin and Big-Little-Big constraints at every vertex.
- **Global layer ordering** that is consistent (no paper passes through itself).
- **No self-intersection** of the paper during the folding motion (for physical realizability).

**Critical distinction:** A crease pattern can be *locally* flat-foldable at every vertex yet fail to be *globally* flat-foldable. Local conditions are necessary but not sufficient for multi-vertex patterns.

**Two-colorability:** Every flat-foldable crease pattern is **two-colorable** -- the faces of the crease pattern can be colored with two colors such that no two adjacent faces share the same color. This follows from the fact that each crease reverses the orientation of the paper.

---

### 1.2 Kawasaki-Justin Theorem

#### Exact Statement

**Kawasaki-Justin Theorem (Single-Vertex Flat-Foldability, Necessary and Sufficient):**
Let a single vertex in a crease pattern have 2n crease lines meeting at it, creating consecutive sector angles alpha_1, alpha_2, ..., alpha_{2n} (summing to 2*pi). The crease pattern is flat-foldable at this vertex *if and only if*:

```
alpha_1 - alpha_2 + alpha_3 - alpha_4 + ... + alpha_{2n-1} - alpha_{2n} = 0
```

**Equivalent formulation (the 180-degree form):**

```
alpha_1 + alpha_3 + alpha_5 + ... + alpha_{2n-1} = pi   (= 180 degrees)
alpha_2 + alpha_4 + alpha_6 + ... + alpha_{2n}   = pi   (= 180 degrees)
```

That is, the sum of every other angle equals pi. This form applies when the paper is initially flat (zero angular defect).

**Immediate corollary:** The number of crease lines at a flat-foldable vertex must be **even** (2n lines creating 2n sectors).

#### History

This theorem was discovered independently by:
- Toshikazu Kawasaki (1989)
- Stuart Robertson (late 1970s)
- Jacques Justin (1986)

Hence it is properly called the **Kawasaki-Justin** theorem. The necessity was proven by all three; the sufficiency (that the angle condition is also sufficient for a *single* vertex) was first stated explicitly by Thomas Hull (1994).

#### Proof Idea

**Necessity:** When paper is folded flat, each crease reverses the paper's orientation. Place the first crease along the positive x-axis. In the flat-folded state, the paper accumulates rotation: after the first sector (angle alpha_1) comes the first crease, which reverses orientation. The second sector rotates by alpha_2 in the reversed direction. After traversing all 2n sectors and 2n creases, we must return to the starting direction. The net rotation is:

```
alpha_1 - alpha_2 + alpha_3 - ... - alpha_{2n} = 0
```

If this sum is not zero, the paper cannot close back on itself, so it cannot fold flat.

**Sufficiency:** Given a crease pattern satisfying the angle condition, an explicit flat folding can be constructed via an **accordion fold** (crimp fold). Choose index i such that the partial alternating sum is minimized. Starting from sector alpha_{2i+1}, alternately assign mountain and valley folds, placing each angular wedge below previous folds. At each step until the final fold, the accordion fold avoids self-intersection.

#### How to Compute

For a simulation engine:

```python
def check_kawasaki(angles):
    """
    angles: list of sector angles [alpha_1, ..., alpha_2n] in radians,
            measured consecutively around the vertex, summing to 2*pi.
    Returns True if Kawasaki-Justin condition is satisfied.
    """
    if len(angles) % 2 != 0:
        return False  # Must have even number of creases
    alternating_sum = sum(a * ((-1)**i) for i, a in enumerate(angles))
    return abs(alternating_sum) < EPSILON

# Equivalent check:
def check_kawasaki_180(angles):
    odd_sum = sum(angles[i] for i in range(0, len(angles), 2))
    even_sum = sum(angles[i] for i in range(1, len(angles), 2))
    return abs(odd_sum - math.pi) < EPSILON and abs(even_sum - math.pi) < EPSILON
```

**For multi-vertex patterns:** Apply Kawasaki-Justin at *every* interior vertex. This gives a necessary condition for local flat-foldability, but is NOT sufficient for global flat-foldability.

---

### 1.3 Maekawa-Justin Theorem

#### Exact Statement

**Maekawa-Justin Theorem:** At any interior vertex of a flat-foldable crease pattern, if M is the number of mountain folds and V is the number of valley folds, then:

```
M - V = +/- 2
```

#### Proof Idea

Consider the cross-section near a flat-folded vertex. The paper forms a flat polygon when viewed edge-on. Each mountain fold contributes an interior angle of 360 degrees (the paper wraps over), and each valley fold contributes an interior angle of 0 degrees (the paper tucks under), or vice versa depending on viewing side.

The sum of interior angles of an n-gon is (n-2) * 180 degrees. With n = M + V creases:

```
0 * V + 360 * M = (M + V - 2) * 180
360M = 180M + 180V - 360
180M - 180V = -360
M - V = -2
```

Viewing from the other side reverses mountain and valley, giving M - V = +2. Therefore |M - V| = 2.

**Corollary:** Since M + V = 2n (total creases) and M - V = +/-2, we get M = n+1, V = n-1 (or vice versa). There is always **one more mountain than valley** (or one more valley than mountain).

**Corollary:** The total number of creases at a flat-foldable vertex must be even (consistent with Kawasaki-Justin).

#### How to Compute

```python
def check_maekawa(assignment):
    """
    assignment: list of 'M' or 'V' for each crease at a vertex.
    Returns True if Maekawa condition holds.
    """
    M = assignment.count('M')
    V = assignment.count('V')
    return abs(M - V) == 2

def enumerate_maekawa_assignments(n_creases):
    """
    Generate all MV assignments satisfying Maekawa's theorem.
    n_creases must be even (= 2k for some k).
    Returns assignments with (k+1) mountains and (k-1) valleys, or vice versa.
    """
    from itertools import combinations
    k = n_creases // 2
    assignments = []
    for n_mountains in [k + 1, k - 1]:
        for mountain_positions in combinations(range(n_creases), n_mountains):
            assignment = ['V'] * n_creases
            for pos in mountain_positions:
                assignment[pos] = 'M'
            assignments.append(tuple(assignment))
    return assignments
```

**Note:** Maekawa's theorem gives an upper bound on valid MV assignments. Not all assignments satisfying M - V = +/-2 are actually flat-foldable; additional constraints (Big-Little-Big, non-crossing) must also be checked.

---

### 1.4 Big-Little-Big Angle Theorem

#### Exact Statement

**Big-Little-Big Lemma (BLB):** In a single-vertex crease pattern with sector angles alpha_1, ..., alpha_{2n}, if a sector angle alpha_i is a **strict local minimum** (i.e., alpha_{i-1} > alpha_i < alpha_{i+1}, with indices modulo 2n), then the two crease lines bounding that sector must have **opposite MV parities** -- one must be mountain and the other must be valley.

#### Extended Form (Equal Angles)

- If a local minimum consists of an **even** number of consecutive equal sector angles, the number of bordering M and V creases must differ by one.
- If a local minimum consists of an **odd** number of consecutive equal sector angles, the number of bordering M and V creases must be equal.

#### Proof Idea

If both bounding creases had the same MV parity (both mountain or both valley), the larger flanking sectors would fold onto the same side of the paper, causing the paper to overlap and self-intersect through the small sector. The only way to avoid this collision is to fold one crease as mountain and the other as valley, so the larger sectors fold to opposite sides.

More formally: consider the cross-section near the vertex after flat folding. The small sector must be "sandwiched" between its neighbors. If both adjacent creases have the same parity, the two large sectors would stack on the same side, creating a geometric impossibility (paper would need to pass through itself).

#### Significance for Simulation

The BLB lemma provides a powerful **pruning rule** for MV assignment search:

1. Find all local-minimum sectors.
2. For each, constrain the bounding creases to have opposite MV parity.
3. This dramatically reduces the search space before attempting full layer-ordering.

```python
def apply_blb_constraints(angles, n_creases):
    """
    Returns a dict of constraints: {(crease_i, crease_j): 'opposite'}
    for creases bounding local-minimum sectors.
    """
    constraints = {}
    for i in range(n_creases):
        prev = (i - 1) % n_creases
        next_ = (i + 1) % n_creases
        if angles[i] < angles[prev] and angles[i] < angles[next_]:
            # Creases i and (i+1) % n bound sector i
            c_left = i        # crease on the left of sector i
            c_right = (i + 1) % n_creases  # crease on the right
            constraints[(c_left, c_right)] = 'opposite'
    return constraints
```

#### Role in Single-Vertex Foldability Algorithm

The complete algorithm for single-vertex flat-foldability:

1. Check **Kawasaki-Justin**: alternating angle sum = 0.
2. Enumerate MV assignments satisfying **Maekawa-Justin**: |M - V| = 2.
3. Filter by **Big-Little-Big**: opposite parity at local-minimum sectors.
4. For remaining candidates, verify no self-intersection via a **layer ordering** check (crimp-and-uncrimp algorithm).

This runs in **polynomial time** for single vertices (O(2^n) candidate assignments, but BLB pruning makes it tractable in practice for typical crease counts).

---

### 1.5 Layer Ordering Problem

#### What Is It?

After a crease pattern is folded flat, different regions (facets) of the paper overlap. The **layer ordering** is a function that assigns each facet a position in a vertical stack wherever facets overlap. The ordering must satisfy:

1. **Consistency**: If facet A is above B, and B is above C, then A must be above C (transitivity).
2. **Non-crossing at creases**: Two facets sharing a crease line must be adjacent in the layer order along that crease (no other facet can be sandwiched between them at the crease).
3. **Taco-taco constraint**: When two creases cross in the folded image, the layers from one crease cannot interleave arbitrarily with layers from the other.
4. **Taco-tortilla constraint**: A crease folding layers cannot have an unfolded facet blocking its closure.

#### Why NP-Hard?

**Bern and Hayes (1996)** proved that determining whether a valid layer ordering exists for a given crease pattern with MV assignment is **NP-complete**.

**Proof technique**: Reduction from **Not-All-Equal 3-SAT (NAE-3SAT)**:
- A **pleat** (two parallel creases creating a fold) encodes a binary variable. The layer ordering of the pleat (which side goes on top) represents the variable's truth value.
- A **clause gadget** is constructed where three pleats intersect. The gadget can be folded flat if and only if the three incoming pleat states are not all equal.
- NAE-3SAT asks whether a Boolean formula in which each clause requires that its three literals not all have the same value is satisfiable. Since NAE-3SAT is NP-complete, flat-foldability is NP-complete.

**Note on proof history**: A gap in the original 1996 proof went undetected for 20 years until Akitaya et al. (2016) repaired and strengthened it.

#### Approximation and Tractability

- **Fixed-parameter tractable (FPT)**: Flat foldability is FPT parameterized by the **ply** (maximum number of overlapping layers) and the **treewidth** of the cell adjacency graph. Time complexity: O((p!)^{O(w)} * n^2) where p = ply, w = treewidth, n = number of creases.
- **Simple patterns**: For single-vertex patterns, the cell adjacency graph is a cycle (treewidth 1), so the algorithm is polynomial.
- **Map folding (1D)**: Folding a 1D strip of paper with prescribed creases is solvable in O(n^2) time, but the 2D map folding problem remains open for general m x n grids.
- **Heuristic approaches**: In practice, constraint propagation (Jason Ku's algorithm) works well:
  1. Determine local layer orders from MV assignments at each crease.
  2. Propagate implications (transitivity).
  3. Recursively guess-and-check remaining ambiguities.
  4. Backtrack on contradiction.

---

### 1.6 Gaussian Curvature in Origami

#### Fundamental Property

Paper is a **developable surface**: it has **zero Gaussian curvature** everywhere. Gaussian curvature is the product of the two principal curvatures (K = kappa_1 * kappa_2). For paper:
- At smooth (unfolded) points: the paper can bend in one direction but not two simultaneously, so at least one principal curvature is zero, giving K = 0.
- At crease lines: the paper has a discontinuity in the surface normal (a "fold"), but the Gaussian curvature of each face on either side remains zero. The crease itself concentrates curvature into a 1D line.
- At vertices: this is where things get interesting.

#### Why Zero Everywhere Except at Vertices

The **Gauss-Bonnet theorem** applied to origami: for a closed region of paper containing a vertex, the integral of Gaussian curvature over the region equals the **angular defect** at the vertex:

```
integral(K dA) = 2*pi - sum(sector_angles_at_vertex)
```

For a flat-foldable vertex (Kawasaki-Justin satisfied), the sector angles sum to 2*pi, so the angular defect is zero. The vertex has zero **discrete Gaussian curvature** -- consistent with the paper being developable.

For a **non-flat** origami vertex (like a partially folded vertex), the angular defect can be nonzero. This is equivalent to the vertex having concentrated Gaussian curvature, which means the paper near that vertex is behaving like a **cone** (positive defect) or a **saddle** (negative defect). However, this is only possible if the paper is being stretched or compressed at the molecular level -- real paper resists this strongly.

**Practical implication for simulation**: The faces of an origami model must remain **developable** (zero Gaussian curvature) at all times. Any deformation that introduces Gaussian curvature into a face is physically unrealistic unless the paper is being plastically deformed. This is why faces in rigid origami are modeled as perfectly rigid panels with zero curvature.

#### Connection to Rigid Origami

In rigid origami, faces must remain planar throughout folding. If triangulated faces are used, planarity is automatic (any three points define a plane). For non-triangular faces, enforcing planarity is a constraint that must be explicitly maintained, and violation of it implies non-zero Gaussian curvature -- physically forbidden for stiff materials.

---

### 1.7 Rigid Origami

#### Definition

**Rigid origami** is origami where:
- Each face of the crease pattern remains **perfectly rigid** (planar, undeformed) during folding.
- All deformation is concentrated at the crease lines, which act as **revolute joints** (hinges).
- The folding is a **continuous motion** from the flat state to the folded state.

This is in contrast to "regular" origami where paper can bend, curve, and deform continuously.

#### What Makes It Different

| Property | Regular Origami | Rigid Origami |
|----------|----------------|---------------|
| Faces | Can bend and curve | Must remain perfectly planar |
| Deformation | Distributed across paper | Concentrated at creases only |
| Degrees of freedom | Very high (infinite-dimensional) | Finite (one fold angle per crease) |
| Model | Continuous thin shell | Linkage mechanism |
| Kinematic analogy | Flexible sheet | System of rigid panels + hinges |
| Gaussian curvature of faces | Can be nonzero transiently | Always zero |

#### Kinematic Model

A rigid origami vertex with n creases is equivalent to a **spherical linkage** with n links and n revolute joints. The fold angles (rho_1, ..., rho_n) are the joint variables. The constraint is that the linkage must close:

```
R_1(rho_1) * R_2(rho_2) * ... * R_n(rho_n) = I   (identity matrix)
```

where R_i(rho_i) is the rotation matrix for the i-th crease by fold angle rho_i around the crease line direction.

For a **degree-4 vertex** (4 creases, the most common in rigid origami patterns like Miura-ori), the spherical linkage has **1 degree of freedom** (generically). This means specifying one fold angle determines all others -- the entire pattern folds as a 1-DOF mechanism.

#### Key Rigid Origami Patterns

- **Miura-ori**: Degree-4 vertices, 1-DOF, negative Poisson's ratio, widely used in engineering.
- **Yoshizawa pattern**: Various fold angles possible.
- **Waterbomb base**: Degree-6 vertices, more complex kinematics.
- **Kresling pattern**: Cylindrical, bistable, used in deployable structures.

#### Rigid Foldability Conditions

Not all crease patterns are rigidly foldable. Conditions:
1. **Kawasaki-Justin** must hold at every vertex (necessary for flat-foldability).
2. The **loop closure equations** (spherical linkage constraints) must have continuous solutions from the flat state.
3. **Compatibility**: At each interior vertex, the fold angles of shared creases between adjacent vertices must agree.

For degree-4 vertices, explicit analytical conditions exist. For higher-degree vertices and multi-vertex patterns, numerical methods are required.

---

### 1.8 Single-Vertex Origami

This is the **simplest and fully understood** case of origami mathematics.

#### Configuration Space

For a single vertex with 2n creases satisfying Kawasaki-Justin, the **configuration space** (set of valid folded states) is fully characterized:

- **MV assignments**: The number of valid MV assignments can be computed by recursive formulas (Hull). For n creases with sector angles alpha_1, ..., alpha_{2n}, the count C(alpha_0, ..., alpha_{2n-1}) follows a recursion based on identifying the smallest sector and "crimping" it.

- **Layer orderings**: For each valid MV assignment, the layer ordering is essentially unique (determined by the accordion fold construction from the sufficiency proof of Kawasaki-Justin).

#### Complete Algorithm (Polynomial Time)

```
Input: sector angles alpha_1, ..., alpha_{2n}
1. Check Kawasaki-Justin: alternating sum = 0?
   If no -> not flat-foldable.
2. For each candidate MV assignment satisfying Maekawa (|M-V| = 2):
   a. Check Big-Little-Big constraints.
   b. Simulate crimp folding to verify no self-intersection.
3. Output all valid (MV assignment, layer ordering) pairs.
```

This is solvable in **polynomial time** because:
- Maekawa limits the number of mountain folds to n+1 or n-1 (binomial choices).
- BLB dramatically prunes the search.
- The crimp algorithm processes creases sequentially.

#### Rigid Single-Vertex

For rigid origami, a single degree-2n vertex is a spherical n-bar linkage. Its configuration space is a smooth manifold whose dimension and topology depend on the sector angles. For degree-4 (the generic case), it is generically a 1-dimensional manifold (a curve) -- a single parameter (fold angle of one crease) determines the rest.

---

### 1.9 Multi-Vertex Origami: Where Complexity Explodes

#### The Fundamental Difficulty

Multi-vertex origami is qualitatively harder than single-vertex because:

1. **Global consistency**: Creases shared between vertices must have consistent fold angles and MV assignments at both endpoints.
2. **Layer ordering becomes global**: The layer ordering must be consistent across the entire pattern, not just locally at each vertex.
3. **Rigid compatibility**: In rigid origami, the fold angles at one vertex constrain fold angles at neighboring vertices, creating a coupled system of nonlinear equations.

#### Complexity Results (Multi-Vertex)

| Problem | Complexity |
|---------|-----------|
| Local flat-foldability (Kawasaki at each vertex) | Polynomial (check each vertex independently) |
| MV assignment for flat fold | **NP-complete** (Bern-Hayes 1996) |
| Layer ordering for flat fold | **NP-complete** (Bern-Hayes 1996) |
| Flat foldability (combined) | **NP-complete** |
| Rigid foldability (continuous motion) | **NP-complete** for arbitrary crease subsets |
| Reconfiguration between flat states | **PSPACE-complete** |
| Counting flat-folded states | **#P-complete** |
| Self-similar infinite patterns | **Undecidable** |
| Flat origami with optional creases | **Turing complete** (can simulate Rule 110) |

#### Why Multi-Vertex is Hard: Intuition

Each vertex locally constrains its creases, but these constraints propagate through the pattern. A pleat at one vertex forces a layer ordering that may conflict with a pleat at a distant vertex. The interconnection structure creates a constraint satisfaction problem equivalent to SAT.

**Bern-Hayes gadgets**:
- **Wire gadget**: A pleat (two parallel creases) propagates a binary signal (which layer is on top).
- **NOT gadget**: A single crease crossing a pleat flips the signal.
- **NAE clause gadget**: Three pleats meeting at a region that can fold flat iff the signals are not all equal.
- These gadgets encode NAE-3SAT, proving NP-completeness.

#### Practical Approaches for Simulation

Despite NP-completeness in the worst case, many practical origami patterns are tractable:

1. **Low ply**: Most artistic origami has bounded ply (< 10 layers). FPT algorithms with ply as parameter are efficient.
2. **Tree-like structure**: Many patterns have cell adjacency graphs with low treewidth, enabling dynamic programming.
3. **Constraint propagation**: Start from boundary conditions and propagate MV/layer constraints. Many patterns resolve without backtracking.
4. **Physics-based simulation**: Avoid the combinatorial problem entirely by simulating the continuous folding process with forces and constraints.

---

## 2. Mechanics / Physics of Folding

### 2.1 What Physically Happens When Paper Is Folded

Paper is a composite material: a network of **cellulose fibers** (typically 1-3 mm long, 20-40 micrometers in diameter) bonded together by hydrogen bonds, with pores filled with air, fillers (clay, calcium carbonate), and sizing agents.

#### Microscale Mechanics of a Fold

When paper is folded:

1. **Elastic bending** (initial phase): The outer fibers are stretched in tension, inner fibers compressed. At the fold line, the paper experiences its maximum curvature. The bending stress follows approximately:
   ```
   sigma = E * y / R
   ```
   where E = Young's modulus, y = distance from neutral axis, R = radius of curvature.

2. **Fiber rearrangement**: At moderate fold sharpness, fibers begin to slide relative to each other, breaking hydrogen bonds between fibers. This is an irreversible microstructural change.

3. **Plastic deformation** (sharp fold): For a tight crease, the inner fibers buckle and permanently deform. The outer fibers may fracture or debond. The cellulose microfibrils within individual fibers undergo plastic kinking. The fold line experiences **localized plastic failure**.

4. **Residual stress**: After unfolding, the permanently deformed fibers create internal stresses that give the paper a "memory" of the fold -- it preferentially returns to the folded state. This is the physical basis of crease memory.

#### Key Parameters

| Parameter | Typical Value (copy paper) |
|-----------|---------------------------|
| Thickness | 80-120 micrometers |
| Young's modulus (MD) | 3-6 GPa |
| Young's modulus (CD) | 1-3 GPa |
| Bending stiffness | 5-15 mN*m |
| Tensile strength | 20-80 MPa |
| Strain at break | 1-5% |

MD = machine direction, CD = cross direction. Paper is anisotropic due to preferential fiber alignment during manufacturing.

---

### 2.2 Stress Concentration at Fold Lines

#### The Problem

A crease line is a **stress concentrator**. The radius of curvature at a sharp fold approaches zero, which would imply infinite stress according to linear elasticity:

```
sigma_max = E * t / (2R)
```

where t = paper thickness, R = radius of curvature at the fold.

In reality, three mechanisms prevent infinite stress:
1. **Finite fold radius**: Even a "sharp" fold has a radius of curvature of ~0.1-0.5 mm.
2. **Plastic yielding**: The material yields before reaching the elastic stress limit.
3. **Fiber debonding**: The network structure allows local failures that redistribute stress.

#### Modeling Approach

For simulation purposes, the stress concentration at a fold is typically not modeled explicitly. Instead:

1. **Ideal crease model**: The crease is a geometric line of zero width. All deformation is captured by the fold angle. The crease has a **rest angle** (the angle it naturally tends toward) and a **torsional stiffness** (resistance to deviating from the rest angle).

2. **Compliant crease model**: The crease has a finite width w. Within this width, the paper has reduced stiffness (due to plastic damage). The crease zone is modeled as a thin strip with modified material properties.

3. **Damage model**: The crease stiffness changes as a function of folding history. Repeated folding reduces the rest angle toward the fully-folded state and may reduce stiffness.

---

### 2.3 Elastic Energy Stored in a Fold

The elastic energy of an origami structure comes from three sources:

#### 1. Crease Folding Energy

Each crease acts as a **torsional spring**. The energy stored in a crease of length L with fold angle rho (deviation from rest angle rho_0):

```
E_crease = (1/2) * kappa * L * (rho - rho_0)^2
```

where kappa is the torsional stiffness per unit length (units: N*m/m = N).

For a crease at its rest angle, E_crease = 0. Energy increases quadratically as the fold angle deviates from rest.

#### 2. Panel Bending Energy

If panels are not perfectly rigid, they can bend. The bending energy density of a thin plate is:

```
e_bend = (B/2) * (kappa_1^2 + kappa_2^2 + 2*nu*kappa_1*kappa_2)
```

where B = Et^3/(12(1-nu^2)) is the flexural rigidity, kappa_1 and kappa_2 are principal curvatures, and nu is Poisson's ratio.

For a developable (zero Gaussian curvature) deformation, one principal curvature is zero, simplifying to:

```
e_bend = (B/2) * kappa^2
```

#### 3. Stretching Energy

If panels are stretched or compressed in-plane:

```
e_stretch = (Et/2) * (epsilon_xx^2 + epsilon_yy^2 + 2*nu*epsilon_xx*epsilon_yy + 2*(1-nu)*epsilon_xy^2) / (1 - nu^2)
```

In rigid origami, stretching energy is zero by assumption. In compliant origami, it is typically much smaller than bending energy for thin sheets.

#### The Key Ratio: Origami Length Scale

The competition between panel bending and crease folding defines a characteristic length:

```
L* = B / kappa
```

where B = flexural rigidity of the panel, kappa = torsional stiffness of the crease per unit length.

- If panel size >> L*: the creases are relatively soft, and the structure behaves like **rigid origami** (panels stay flat, all deformation at creases).
- If panel size << L*: the creases are relatively stiff, and the panels bend significantly -- the structure behaves as a **flexible shell**.

For most paper origami, panel size >> L*, so rigid origami is a good approximation.

---

### 2.4 Bending Stiffness vs. Fold Angle Relationship

The moment-angle relationship of a crease is approximately **linear** for small deviations from the rest angle:

```
M = kappa * L * (rho - rho_0)
```

where M is the restoring moment, L is crease length, rho is current fold angle, rho_0 is rest angle.

**Nonlinear effects at large deviations:**

1. **Geometric nonlinearity**: For large fold angles, the effective lever arm changes, and the relationship between moment and angle becomes nonlinear.
2. **Material nonlinearity**: At large deformations, the crease material (damaged paper fibers) exhibits nonlinear stress-strain behavior.
3. **Contact**: When the fold approaches 0 degrees (fully folded) or 360 degrees, the paper layers contact each other, introducing a hard constraint.

**For simulation**, a piecewise model is often used:

```
M(rho) = kappa * L * (rho - rho_0)                    for |rho - rho_0| < rho_threshold
M(rho) = kappa * L * rho_threshold * sign(rho - rho_0)  for |rho - rho_0| >= rho_threshold
```

More sophisticated models use exponential or polynomial stiffening near full closure.

**Accuracy Note**: The linear spring model is suggested for use when the folding angle deviation is less than approximately 45 degrees from rest. Beyond this range, nonlinear corrections become significant.

---

### 2.5 Crease Mechanics: Permanent Effects

A crease permanently alters paper properties:

#### Rest Angle Shift (Plasticity)

Before folding: rest angle = pi (flat, 180 degrees).
After folding to angle rho_fold and releasing:

```
rho_0_new = pi - alpha * (pi - rho_fold)
```

where alpha is a plasticity parameter (0 = no plastic deformation, 1 = perfect memory of the fold). For sharp folds in standard paper, alpha is approximately 0.7-0.9.

The rest angle shifts toward the folded state. This is why folded paper stays folded -- the physical basis of origami.

#### Stiffness Reduction

The torsional stiffness of a crease decreases with repeated folding:

```
kappa_n = kappa_0 * (1 - beta * log(n + 1))
```

where kappa_n is stiffness after n fold cycles, kappa_0 is initial stiffness, and beta is a material-dependent degradation parameter.

This models the progressive breaking of fiber bonds with repeated folding.

#### Anisotropy

A crease introduces local anisotropy: the paper is much weaker in bending across the crease line than along it. In-plane stiffness across the crease is also reduced.

---

### 2.6 Thickness Accommodation

#### The Problem

Mathematical origami assumes zero-thickness paper. Real materials have thickness t > 0, which creates problems:

1. **Geometric interference**: When paper folds, the inner layers have shorter paths than outer layers. For a fold angle theta, the path length difference is approximately t * theta.
2. **Accumulation**: In multi-layer folds, thickness accumulates, causing significant geometric errors.
3. **Strain**: Thick materials experience significant bending strain at folds: epsilon = t/(2R).
4. **Kinematics**: Thick panels cannot fold the same way as zero-thickness paper -- the kinematics change.

#### Thickness Accommodation Techniques

**1. Offset Panel Technique:**
Panels are offset from the mathematical (zero-thickness) fold surface by t/2. Each panel sits on one side of the mathematical surface. Creases are placed at the edges of offset panels, creating gaps.

- Preserves kinematics of the zero-thickness pattern.
- Introduces gaps between panels at valley folds.
- Simple to implement but creates asymmetry.

**2. Tapered Panel Technique:**
Panels are tapered (thinned) near crease lines, maintaining the mathematical surface at the center of each panel while allowing folding at the edges.

- Preserves the same kinematic folding motion as the zero-thickness system.
- Requires material removal.
- Suitable for manufacturing.

**3. Double Crease Technique:**
A single mathematical crease is replaced by two parallel creases separated by a distance proportional to thickness. The strip between them accommodates the volume.

- Simple and effective.
- Changes the crease pattern geometry slightly.
- Common in engineering applications.

**4. Membrane Hinge Technique:**
Thick panels are connected by thin flexible membranes (living hinges). The membrane serves as the hinge while the panels remain rigid and thick.

- Clean separation of rigid panels and flexible hinges.
- Widely used in manufactured products (e.g., plastic containers with living hinges).
- The membrane can be different material from the panels.

**5. Rolling Contact Technique:**
Panels are connected by rolling contact elements that accommodate thickness through synchronized rolling motion.

- Achieves true rigid foldability with thick panels.
- Complex mechanism but kinematically exact.

---

### 2.7 Spring-Hinge Model

The **spring-hinge model** is the simplest mechanical model of origami:

#### Model Description

- Each **face** of the crease pattern is a rigid panel.
- Each **crease** is a **torsional spring** (revolute joint with spring resistance).
- The spring has:
  - A **rest angle** rho_0 (the angle the crease naturally tends toward).
  - A **torsional stiffness** kappa (resistance to deviation from rest angle).
  - A **damping coefficient** c (energy dissipation during dynamic motion).

#### Equations of Motion

For a dynamic simulation:

```
I * d^2(rho)/dt^2 + c * d(rho)/dt + kappa * (rho - rho_0) = tau_external
```

where I is the moment of inertia, c is damping, kappa is torsional stiffness, and tau_external is any externally applied torque.

For quasi-static simulation (slow folding), inertia is negligible:

```
c * d(rho)/dt + kappa * (rho - rho_0) = tau_external
```

#### Advantages

- Simple and computationally efficient.
- Captures the essential 1-DOF-per-crease kinematics.
- Easy to implement.

#### Limitations

- Cannot capture face bending (faces are perfectly rigid).
- Cannot capture stretching or shearing.
- Crease stiffness is a single scalar (no width, no distributed behavior).
- Poor accuracy for compliant origami where face bending is significant.

---

### 2.8 Bar-and-Hinge Model (Ghassaei's Approach)

The **bar-and-hinge model** is a more sophisticated mechanical model that captures three types of deformation while remaining computationally efficient.

#### Model Description

Each face of the crease pattern is triangulated (if not already triangular). The model then consists of:

1. **Bars (edges)**: Each edge of the triangulated mesh is a bar with **axial stiffness**. Bars resist stretching and compression. This prevents the faces from stretching or shearing.

2. **Fold hinges (at creases)**: Each crease line has a **torsional spring** connecting the two adjacent faces. This captures the crease folding behavior with a target fold angle and stiffness.

3. **Facet hinges (within faces)**: Each pair of triangles sharing a non-crease edge has a **torsional spring** with high stiffness and a rest angle of pi (flat). This penalizes face bending, keeping faces approximately flat while allowing small deformations.

#### Three Deformation Modes

| Mode | Element | Stiffness | Physical meaning |
|------|---------|-----------|-----------------|
| Stretching/shearing | Bars | Axial stiffness (Et) | In-plane deformation of faces |
| Face bending | Facet hinges | Flexural rigidity (B ~ Et^3) | Out-of-plane bending of faces |
| Crease folding | Fold hinges | Torsional stiffness (kappa) | Folding at prescribed crease lines |

#### Energy Formulation

Total energy:

```
E_total = E_bar + E_facet + E_fold

E_bar   = sum_bars   (1/2) * k_axial * (L - L_0)^2
E_facet = sum_facets (1/2) * k_facet * l * (theta - pi)^2
E_fold  = sum_folds  (1/2) * k_fold  * l * (rho - rho_target)^2
```

where L, L_0 = current and rest length of bars; theta = dihedral angle at facet edges; rho, rho_target = current and target fold angle at creases; l = edge length; and k values are stiffness parameters.

#### Stiffness Parameters

From Schenk and Guest's formulation:

```
k_axial = E * t * w / L_0          (bar axial stiffness)
k_facet = E * t^3 / (12 * (1-nu^2))  (per unit length, facet bending)
k_fold  = kappa                     (per unit length, crease torsion)
```

where E = Young's modulus, t = thickness, w = tributary width, nu = Poisson's ratio.

#### Compliant Crease Extension

For creases with finite width w_c, Ghassaei et al. extended the model:
- The crease zone is represented by **7 nodes, 12 bars, and 8 rotational springs**.
- Two rows of rotational springs (not one) capture the distributed curvature in the crease zone.
- Additional bars capture torsional and extensional deformation within the crease.

This extension is critical for predicting **bistability** and **multistability** in origami structures, which the simplified model misses.

#### Solution Method

The bar-and-hinge model is solved using **nonlinear static analysis**:

1. Define target fold angles (rho_target) as a function of a folding parameter t in [0, 1].
2. Increment t in small steps.
3. At each step, minimize E_total using Newton-Raphson iteration:
   - Compute forces: F_i = -partial(E_total)/partial(x_i)
   - Compute stiffness matrix: K_ij = partial^2(E_total)/partial(x_i)partial(x_j)
   - Update positions: delta_x = K^{-1} * F
4. Check convergence.

#### GPU-Accelerated Dynamic Variant (Origami Simulator)

Amanda Ghassaei's browser-based Origami Simulator uses a dynamic variant:
- All constraints (bars, facet hinges, fold hinges) generate forces.
- Positions are updated via numerical integration (Euler or Verlet).
- **Verlet integration**: x_{n+1} = 2*x_n - x_{n-1} + F*dt^2/m
- GPU fragment shaders compute forces in parallel for fast performance.
- Damping prevents oscillation; system converges to static equilibrium.
- All creases fold simultaneously (not sequentially).

---

## 3. Computational Origami Theory

### 3.1 Tractability Landscape

| Problem | Complexity | Notes |
|---------|-----------|-------|
| Single-vertex flat foldability (Kawasaki check) | **O(n)** | Sum alternating angles |
| Single-vertex MV assignment | **Polynomial** | Recursive crimp algorithm |
| Single-vertex layer ordering | **Polynomial** | Determined by MV assignment |
| Multi-vertex local flat foldability | **Polynomial** | Check Kawasaki at each vertex independently |
| Multi-vertex MV assignment | **NP-complete** | Bern-Hayes 1996 |
| Multi-vertex layer ordering | **NP-complete** | Bern-Hayes 1996 |
| Multi-vertex flat foldability (combined) | **NP-complete** | MV + layer ordering together |
| Flat foldability (bounded ply + bounded treewidth) | **FPT**: O((p!)^{O(w)} n^2) | Tractable for simple patterns |
| Map folding (1xn strip) | **O(n^2)** | |
| Map folding (2xn grid) | **Polynomial** | |
| Map folding (general mxn) | **Open** | Conjectured NP-hard |
| Rigid foldability (continuous motion, all creases) | **Weakly NP-complete** | |
| Rigid foldability (arbitrary crease subsets) | **Strongly NP-complete** | |
| Reconfiguration between flat states | **PSPACE-complete** | |
| Counting flat-folded states | **#P-complete** | |
| Flat origami with optional creases | **Turing complete** | Simulates Rule 110 cellular automaton |
| Self-similar infinite pattern foldability | **Undecidable** | coRE-complete |

### 3.2 Single-Vertex Flat Foldability: Polynomial Time

**Algorithm (Hull's Crimp Algorithm):**

```
Input: Sector angles alpha_1, ..., alpha_{2n} at a vertex.
Output: All valid (MV assignment, layer ordering) pairs, or "not flat-foldable."

1. Check Kawasaki-Justin: sum of odd-indexed angles = pi?
   If not, return "not flat-foldable."

2. Find the smallest sector angle alpha_min.
   (If tie, choose any.)

3. The two creases bounding alpha_min must have OPPOSITE MV parity
   (Big-Little-Big lemma).

4. "Crimp" the smallest sector: merge it with the adjacent sectors.
   The new sector has angle = alpha_{i-1} - alpha_i + alpha_{i+1}
   (which equals the original angle alpha_{i-1} or alpha_{i+1} minus the "crimped" difference).

5. This reduces the vertex to a (2n-2)-crease vertex.
   Recurse.

6. Base case: 2 creases, angles pi, pi. Trivially foldable.
```

**Time complexity:** O(n^2) in the straightforward implementation, O(n log n) with priority queues.

**Counting valid MV assignments:** Hull developed recursive formulas C(alpha_0,...,alpha_{2n-1}) for the number of valid MV assignments, based on the crimp structure. The count depends on angle multiplicities and can range from 2 (minimum, for generic angles) to 2^{n-1} (maximum, for equal angles).

### 3.3 Multi-Vertex Flat Foldability: NP-Complete

**The Bern-Hayes Construction (1996):**

The reduction is from **NAE-3SAT** (Not-All-Equal 3-Satisfiability):
- Given a Boolean formula in CNF where each clause has 3 literals.
- NAE-3SAT asks: is there an assignment such that no clause has all three literals the same value?

**Gadgets:**

1. **Wire (pleat)**: Two parallel creases forming a pleat. Binary state = which layer goes on top. This encodes a Boolean variable.

2. **Signal propagation**: The pleat's layer ordering propagates along the crease lines, maintaining the binary state.

3. **Turn/crossover**: Pleats can turn corners and cross over each other using specific crease configurations.

4. **NAE clause gadget**: Three pleats meeting at a junction. The junction can fold flat if and only if the three incoming layer orderings are NOT all equal. This directly encodes an NAE-3SAT clause.

5. **Assembly**: Given an NAE-3SAT formula, construct a crease pattern with one pleat per variable and one clause gadget per clause, connected by wire gadgets. The crease pattern is flat-foldable iff the NAE-3SAT formula is satisfiable.

Since NAE-3SAT is NP-complete, and the reduction is polynomial, multi-vertex flat foldability is NP-complete.

**Implications for simulation**: There is no known polynomial-time algorithm for deciding flat-foldability of general crease patterns (unless P = NP). Practical simulation must use heuristics, constraint propagation, or physics-based approaches.

### 3.4 Rigid Foldability: How to Check

#### Degree-4 Vertices (Analytical)

For a degree-4 vertex with sector angles (alpha, beta, gamma, delta) where alpha + beta + gamma + delta = 2*pi and Kawasaki-Justin holds (alpha + gamma = beta + delta = pi):

The fold angles (rho_1, rho_2, rho_3, rho_4) are related by analytical expressions. For a Miura-ori type vertex:

```
tan(rho_2/2) = -(cos((alpha-beta)/2) / cos((alpha+beta)/2)) * tan(rho_1/2)
tan(rho_3/2) = -(cos((alpha+delta)/2) / cos((alpha-delta)/2)) * tan(rho_1/2)
tan(rho_4/2) = -(cos((alpha-beta)/2) / cos((alpha+beta)/2)) * tan(rho_3/2)
```

These give explicit fold-angle relationships, showing the 1-DOF nature.

#### General Vertices (Numerical)

For a vertex of degree n, the rigid foldability condition is the **loop closure equation** of the equivalent spherical linkage:

```
R(e_1, rho_1) * R(e_2, rho_2) * ... * R(e_n, rho_n) = I
```

where R(e_i, rho_i) is rotation by angle rho_i around axis e_i (the crease direction).

This is a system of nonlinear equations in the fold angles. The number of degrees of freedom is n - 3 (for a spherical linkage with n links).

**Solution methods:**
1. **Newton-Raphson**: Iteratively solve the constraint equations starting from the flat state.
2. **Continuation methods**: Trace the solution curve as a parameter (e.g., one fold angle) varies.
3. **Dual quaternion method**: Model rotations using dual quaternions for improved numerical stability. The QRS method decomposes fold angles into quaternion representations for multi-vertex systems.

#### Multi-Vertex Rigid Foldability

For a pattern with multiple vertices:
1. Each vertex imposes loop closure constraints.
2. Shared creases between vertices must have the same fold angle.
3. The combined system of constraints defines the configuration space.

**Heuristic algorithms** (e.g., by Tachi) adjust vertex positions of initially non-rigid patterns to make them rigidly foldable. The configuration is represented by fold angles, and the folding trajectory is computed by projecting the desired motion onto the constraint manifold.

### 3.5 Self-Intersection Detection

Self-intersection detection is critical for validating origami configurations. Two types:

#### 1. Discrete Self-Intersection (Flat Folds)

For flat-folded states, self-intersection is detected by checking **layer ordering consistency**:
- **Taco-taco constraint**: Two crossing creases create a "taco-taco" configuration. The layer ordering of the four resulting regions must be consistent (no interleaving that requires paper to pass through itself).
- **Taco-tortilla constraint**: A crease adjacent to a non-creased region must not have the flat region blocking its closure.
- **Transitivity**: If A > B and B > C in layer order, then A > C.

Jason Ku's algorithm:
1. Compute the overlap graph (which facets overlap in the folded state).
2. For each pair of overlapping facets, determine ordering constraints from MV assignments at shared boundaries.
3. Propagate implications via transitivity.
4. Check for contradictions (cycles in the ordering).

#### 2. Continuous Self-Intersection (During Folding)

During the folding motion, faces may collide transiently even if the final state is valid.

**Approaches:**
- **Bounding volume hierarchies (BVH)**: Enclose each face in a bounding box. Test for overlap using a BVH tree. If bounding boxes overlap, test the actual triangles for intersection.
- **Swept volume**: Track the volume swept by each face during a folding step. Check for overlaps between swept volumes.
- **Incremental testing**: At each time step, check all face pairs for geometric intersection using triangle-triangle intersection tests (Moller's algorithm or similar).

**Computational cost**: O(n^2) per time step for brute-force pairwise testing, O(n log n) with spatial acceleration structures.

### 3.6 Fold-and-Cut Theorem

#### Statement

**Fold-and-Cut Theorem (Demaine, Demaine, Lubiw, 1998):** Every pattern of straight-line cuts in a flat piece of paper (i.e., any planar straight-line graph drawn on the paper) can be achieved by folding the paper flat and making a single complete straight cut.

In other words: any shape (or collection of shapes) bounded by straight edges can be cut from a single sheet of paper with one straight cut, provided the paper is first folded appropriately.

#### Two Proof Methods

**1. Straight Skeleton Method (Demaine, Demaine, Lubiw):**
- Compute the **straight skeleton** of the planar graph (a medial-axis-like structure formed by shrinking the faces of the graph at unit speed).
- The straight skeleton edges become crease lines.
- Add perpendicular creases to make the pattern flat-foldable.
- The resulting crease pattern, when folded, aligns all cut edges along a single line, enabling the single cut.

**2. Disk Packing Method (Bern, Demaine, Eppstein, Hayes):**
- Pack disks into the faces of the graph, tangent to each edge.
- The **Apollonius problem** (finding circles tangent to three given circles) is used to fill gaps.
- The disk packing defines a decomposition into **molecules** (regions between disks).
- **Universal molecules** are constructed for each region, providing crease patterns that fold each molecule flat.
- The assembly of all molecules gives the complete crease pattern.

#### Universal Molecule

A **universal molecule** is a crease pattern for a convex polygonal region with prescribed fold directions on its boundary such that:
1. It folds flat.
2. All boundary edges align along a single line after folding.

The universal molecule construction uses:
- The **perpendicular folds** from each boundary edge.
- **Rabbit ear** triangulations for triangular regions.
- **Recursive decomposition** for higher polygons.

This is a key primitive for computational origami design tools.

### 3.7 Universal Molecule Approach

The universal molecule approach is the computational backbone of the fold-and-cut construction and is more broadly used in computational origami design.

**Algorithm:**

1. **Input**: A planar straight-line graph G (the desired crease/cut pattern).
2. **Compute the straight skeleton** of G. This produces a tree-like structure inside each face.
3. **Decompose** the pattern into convex regions using the skeleton.
4. **For each convex region**, construct a universal molecule:
   a. Assign fold directions (M/V) on the boundary based on the desired folding.
   b. Add internal creases: perpendicular folds from vertices, diagonal folds for triangulation.
   c. Verify flat-foldability of the molecule.
5. **Assemble** all molecules. The internal creases of adjacent molecules must be compatible.
6. **Output**: Complete crease pattern with MV assignment.

**Implementation challenges:**
- The straight skeleton can have degenerate cases (vertices of degree > 3).
- Disk packing requires solving Apollonius problems (circle tangent to three circles), which has up to 8 solutions per instance.
- Numerical precision is critical: small errors in skeleton computation propagate to invalid crease patterns.
- Software tools like **TreeMaker** (Robert Lang) and **ORIPA** implement variants of this approach.

---

## 4. Engineering Metrics

### 4.1 Deployment Ratio

#### Definition

The **deployment ratio** (or **packaging ratio**) quantifies how much an origami structure can expand from its folded (stowed) state to its deployed state:

```
DR = L_deployed / L_stowed
```

where L is a characteristic dimension (length, area, or volume, depending on application).

**Variants:**

- **Linear deployment ratio**: ratio of deployed length to stowed length (1D).
- **Areal deployment ratio**: ratio of deployed area to stowed area (2D): DR_area = A_deployed / A_stowed.
- **Volumetric deployment ratio**: ratio of deployed volume to stowed volume (3D).

#### How It's Measured in Real Engineering

1. **Satellite solar arrays**: Deployment ratio measured as deployed array area divided by launch fairing cross-section area. Typical: 10:1 to 50:1 for advanced designs.
2. **Antenna reflectors**: Ratio of deployed aperture diameter to stowed package diameter. Typical: 5:1 to 20:1.
3. **Emergency shelters**: Ratio of floor area deployed to storage volume. Typical: 20:1 to 100:1.

#### Theoretical Limits

For a Miura-ori pattern with m x n cells:
```
DR_linear ~ m * sin(theta)   (in one direction)
DR_area ~ m * n * sin(theta)  (area ratio)
```
where theta is the fold angle parameter. As theta approaches 0, the deployment ratio increases, but the stowed package thickness also increases due to layer accumulation.

**Practical limit**: Deployment ratio is bounded by the number of layers (ply) and the material thickness. For n layers of thickness t, the minimum stowed dimension is ~ n*t.

### 4.2 Structural Stiffness of Folded State

Origami structures exhibit **tunable stiffness** that depends on the fold state:

#### Stiffness Modulation

- **Flat (unfolded) state**: Maximum in-plane stiffness, minimum out-of-plane stiffness (it is just a flat sheet).
- **Partially folded**: Stiffness varies continuously with fold angle. The structure gains out-of-plane stiffness (3D geometry provides structural depth) while losing some in-plane stiffness.
- **Fully folded (compact)**: Complex stiffness behavior depending on pattern; often the stiffest configuration due to maximum layer stacking.

#### Miura-ori Stiffness

The effective in-plane stiffness of a Miura-ori pattern scales as:

```
K_x ~ E*t * (cos(theta)^2 / sin(theta))   (along corrugation direction)
K_y ~ E*t * sin(theta)                      (perpendicular to corrugation)
```

where theta is the fold angle. Note that K_x diverges as the fold closes (theta -> 0), while K_y vanishes -- this is the auxetic behavior.

The **out-of-plane bending stiffness** scales approximately as:

```
B_eff ~ E*t * h^2
```

where h is the effective structural depth (related to fold amplitude).

#### Self-Locking

Some origami patterns exhibit **self-locking**: at certain fold angles, geometric constraints prevent further motion without significant force. The structure transitions from a mechanism (zero-stiffness mode) to a structure (finite stiffness in all directions).

### 4.3 Fatigue at Fold Lines

#### The Problem

Repeated folding and unfolding causes progressive damage at crease lines:

1. **Cycle 1**: Initial plastic deformation creates the crease. Fiber bonds break, fibers permanently deform.
2. **Cycles 2-10**: Crease softens, rest angle shifts further. The crease becomes "trained."
3. **Cycles 10-100**: Gradual stiffness degradation. Fiber fracture accumulates.
4. **Cycles 100-1000**: Significant weakening. Risk of crack propagation from the crease into the faces.
5. **Cycles 1000+**: Material failure. The paper tears along the fold line.

#### Fatigue Life

The fatigue life depends on:
- **Material**: Paper (100-1000 cycles), polymers (10^4-10^6), metals (10^3-10^5), shape memory alloys (10^6+).
- **Fold angle amplitude**: Larger angle changes = faster fatigue.
- **Fold radius**: Sharper folds = higher strain = faster fatigue.
- **Loading rate**: Faster folding = more heat generation = accelerated degradation.

#### Modeling Fatigue

For simulation, a simple fatigue model:

```
kappa_N = kappa_0 * (1 - D(N))
D(N) = (N / N_f)^p

where:
  kappa_N = stiffness after N cycles
  kappa_0 = initial stiffness
  N_f = cycles to failure
  p = material exponent (typically 0.5-2)
  D = damage variable (0 = undamaged, 1 = failed)
```

When D(N) = 1, the crease has failed (torn through).

#### Design for Fatigue Resistance

- **Living hinges**: Use a different, more fatigue-resistant material at the crease (e.g., polypropylene).
- **Wider creases**: Distribute the deformation over a wider zone, reducing peak strain.
- **Reduced fold amplitude**: Design for partial folding rather than full 180-degree folds.
- **Material selection**: Elastomers and thermoplastic elastomers offer the best fatigue resistance.

### 4.4 Bistability

#### Definition

An origami structure is **bistable** if it has two distinct stable equilibrium configurations separated by an energy barrier. The structure can "snap" between the two states.

#### Mechanism

Bistability arises from the competition between:
1. **Crease energy**: Creases prefer their rest angles.
2. **Face bending energy**: Faces resist bending.
3. **Geometric constraints**: The topology of the pattern constrains the configuration space.

When the energy landscape has two local minima (valleys) separated by a saddle point (hill), the structure is bistable.

#### Example: Kresling Pattern

The Kresling pattern (a cylindrical origami pattern) is naturally bistable:
- **State 1**: Extended (tall cylinder).
- **State 2**: Compressed (short, twisted cylinder).
- **Transition**: Requires overcoming an energy barrier (the faces must bend temporarily during the snap-through).

The bistability of the Kresling pattern can be tuned by adjusting:
- Number of sides (polygon order).
- Fold angle of the pattern.
- Material stiffness ratio (crease vs. face).

#### Modeling Bistability

The bar-and-hinge model with compliant creases (Ghassaei's extended model) can capture bistability. The key is:
1. Include face bending energy (facet hinges), not just crease folding energy.
2. Use the compliant crease model (finite crease width) to capture torsional and extensional deformation in the crease zone.
3. Trace the energy landscape as a function of a loading parameter to find multiple stable equilibria.

**Critical**: The simplified spring-hinge model (rigid faces + torsional springs at creases) typically CANNOT predict bistability because it lacks face bending energy. The competition between face and crease energies is essential for bistability.

### 4.5 Shape Memory in Origami Structures

#### Crease-Level Shape Memory

Individual creases exhibit shape memory due to plastic deformation:
- After folding and unfolding, the crease "remembers" the fold angle.
- Upon re-folding, the crease preferentially returns to its previously folded state.
- The memory improves with repeated folding cycles.

This is the basis of the **Miura-ori map**: a pre-folded Miura-ori pattern can be unfolded to flat and easily re-folded by pushing two opposite corners together. The crease memory guides the paper back to the correct folded state.

#### Material-Level Shape Memory

**Shape memory alloys (SMAs)** and **shape memory polymers (SMPs)** can be used to create origami structures with active shape memory:

- **SMA origami**: Creases made from nitinol wire. At low temperature, the wire is flexible and the structure can be folded flat. Upon heating, the SMA transitions to its austenite phase and contracts, deploying the structure to its memorized 3D shape.
- **SMP origami**: The entire sheet is a shape memory polymer. Heated above T_g, the polymer is soft and can be folded. Cooled below T_g, the polymer hardens in the folded state. Re-heating above T_g triggers autonomous deployment to the flat (or programmed 3D) state.

#### For Simulation

Shape memory can be modeled by making the rest angle a function of temperature:

```
rho_0(T) = rho_programmed                   for T < T_transition
rho_0(T) = rho_deployed                     for T > T_transition
rho_0(T) = interpolation in transition zone
```

### 4.6 Auxetic Behavior (Negative Poisson's Ratio)

#### Definition

A material or structure has **auxetic** behavior if it exhibits a **negative Poisson's ratio**: when stretched in one direction, it expands in the perpendicular direction (instead of contracting as normal materials do).

#### Origami Auxetic Patterns

**Miura-ori** is the canonical example of an auxetic origami pattern:

```
nu_xy = -1   (Poisson's ratio for in-plane deformation of Miura-ori)
```

When a Miura-ori sheet is pulled in the x-direction (along the corrugation), it also expands in the y-direction. This is a purely geometric effect arising from the kinematics of the fold pattern.

**Physical explanation**: As the Miura-ori unfolds in one direction, the fold angle changes globally (it is a 1-DOF mechanism). This global angle change causes the pattern to simultaneously unfold in the perpendicular direction.

#### Tunable Poisson's Ratio

By modifying the pattern geometry, the Poisson's ratio can be tuned:

- **Standard Miura-ori**: nu = -1 (isotropic auxetic in the folding mode).
- **Modified Miura-ori** (varying parallelogram angles): nu can range from negative to positive.
- **Reentrant patterns** (Tachi-Miura polyhedra, zigzag modifications): can achieve strongly negative Poisson's ratios (nu < -1).
- **Hybrid patterns**: Combining different unit cells can create programmable Poisson's ratio fields.

#### Engineering Applications

1. **Impact absorption**: Auxetic structures densify under impact (all directions compress simultaneously), making them excellent energy absorbers.
2. **Morphing surfaces**: Auxetic sheets can conform to doubly-curved surfaces (saddle shapes) without wrinkling, because the negative Poisson's ratio accommodates the required in-plane deformation.
3. **Deployable structures**: The simultaneous expansion in all directions enables compact packaging and uniform deployment.
4. **Medical stents**: Auxetic origami tubes expand radially when stretched axially, useful for vascular stents.

#### For Simulation

The effective Poisson's ratio of an origami pattern can be computed from the kinematic equations:

```
nu_eff = -(d(epsilon_y)/d(epsilon_x))

where epsilon_x, epsilon_y are the effective in-plane strains
computed from the fold-angle-dependent geometry.
```

For Miura-ori with parallelogram angle phi and fold angle theta:

```
nu_xy = -( (cos(theta) * tan(phi))^2 + sin(theta)^2 ) / (cos(theta)^2 * tan(phi)^2 + sin(theta)^2)
```

This gives nu_xy = -1 for the standard Miura-ori (symmetric case), but can vary for asymmetric variants.

---

## 5. Key References and Sources

### Foundational Texts

- **Demaine, E.D. and O'Rourke, J.** (2007). *Geometric Folding Algorithms: Linkages, Origami, Polyhedra*. Cambridge University Press. -- The definitive monograph on computational origami.
- **Lang, R.J.** (2011). *Origami Design Secrets: Mathematical Methods for an Ancient Art*. 2nd ed. A K Peters/CRC Press. -- Practical computational design methods including TreeMaker.
- **Hull, T.C.** (2020). *Origametry: Mathematical Methods in Paper Folding*. Cambridge University Press. -- Modern mathematical treatment.

### Key Papers

- **Bern, M. and Hayes, B.** (1996). "The complexity of flat origami." *Proceedings of the 7th ACM-SIAM Symposium on Discrete Algorithms (SODA)*. -- Proved NP-completeness of flat foldability.
- **Akitaya, H.A. et al.** (2016). "Box pleating is hard." *Proceedings of the 16th Japan Conference on Discrete and Computational Geometry and Graphs*. -- Repaired and strengthened Bern-Hayes proof.
- **Hull, T.C. and Zakharevich, I.** (2025). "Flat origami is Turing complete." *arXiv:2309.07932v4*. -- Showed flat origami with optional creases can simulate universal computation.
- **Schenk, M. and Guest, S.D.** (2011). "Origami folding: A structural engineering approach." *Origami 5: Fifth International Meeting of Origami Science, Mathematics, and Education*. -- Bar-and-hinge model foundations.
- **Ghassaei, A. et al.** (2018). "Fast, interactive origami simulation using GPU computation." *Origami 7*. -- GPU-accelerated bar-and-hinge implementation (origamisimulator.org).
- **Tachi, T.** (2009). "Simulation of rigid origami." *Origami 4: Fourth International Meeting of Origami Science, Mathematics, and Education*. -- Rigid origami simulator.
- **Demaine, E.D., Demaine, M.L., and Lubiw, A.** (1998). "Folding and one straight cut suffice." *Proceedings of the 10th ACM-SIAM Symposium on Discrete Algorithms*. -- Fold-and-cut theorem.
- **Liu, K. and Paulino, G.H.** (2017). "Nonlinear mechanics of non-rigid origami: an efficient computational approach." *Proceedings of the Royal Society A*, 473(2206). -- Advanced bar-and-hinge mechanics.

### Computational Complexity

- **Stern, A. and Hull, T.** (2025). "Computational Complexities of Folding." *arXiv:2410.07666*. -- Comprehensive survey of complexity results including FPT, PSPACE, #P, and undecidability results.
- **Hull, T.C.** (1994). "On the mathematics of flat origamis." *Congressus Numerantium*, 100, 215-224. -- Sufficiency of Kawasaki's condition, counting MV assignments.

### Engineering and Mechanics

- **Filipov, E.T., Liu, K., Tachi, T., Schenk, M., and Paulino, G.H.** (2017). "Bar and hinge models for scalable analysis of origami." *International Journal of Solids and Structures*, 124, 26-45. -- Comprehensive bar-and-hinge formulation.
- **Lang, R.J. et al.** (2018). "A review of thickness-accommodation techniques in origami-inspired engineering." *Applied Mechanics Reviews*, 70(1), 010805. -- Survey of thickness methods.
- **Silverberg, J.L. et al.** (2014). "Using origami design principles to fold reprogrammable mechanical metamaterials." *Science*, 345(6197), 647-650. -- Programmable mechanical properties.
- **Yasuda, H. and Yang, J.** (2015). "Reentrant origami-based metamaterials with negative Poisson's ratio and bistability." *Physical Review Letters*, 114(18), 185502. -- Auxetic and bistable origami.
- **Lechenault, F., Thiria, B., and Adda-Bedia, M.** (2014). "Mechanical response of a creased sheet." *Physical Review Letters*, 112, 244301. -- Elastic theory of creases.

### Software Tools

- **Origami Simulator** (Ghassaei): https://origamisimulator.org/ -- Browser-based GPU-accelerated simulator.
- **Rigid Origami Simulator** (Tachi): https://origami.c.u-tokyo.ac.jp/~tachi/software/ -- Rigid origami kinematic simulator.
- **ORIPA** (Mitani): Crease pattern editor with flat-foldability checks.
- **TreeMaker** (Lang): Computational origami design from stick figures.
- **Rabbit Ear** (Kraft): JavaScript library for computational origami with Kawasaki/Maekawa solvers and layer ordering.

---

## Appendix A: Huzita-Hatori Axioms (Origami Constructive Power)

The **seven Huzita-Hatori axioms** define all possible single-fold operations in origami and establish that origami construction is strictly more powerful than compass-and-straightedge construction.

| Axiom | Description | Geometric Power |
|-------|-------------|----------------|
| O1 | Given two points, fold a line through both. | Line through 2 points (same as straightedge) |
| O2 | Given two points, fold one onto the other. | Perpendicular bisector |
| O3 | Given two lines, fold one onto the other. | Angle bisector |
| O4 | Given a point and a line, fold a perpendicular through the point. | Perpendicular through a point |
| O5 | Given two points and a line, fold one point onto the line through the other. | Solves quadratic equations |
| O6 | Given two points and two lines, fold each point onto its line simultaneously. | **Solves cubic equations** |
| O7 | Given a point and two lines, fold the point onto one line with the fold perpendicular to the other. | Perpendicular fold onto a line |

**Axiom O6 is the key**: It allows origami to solve cubic equations, which compass and straightedge cannot do. This enables:
- **Angle trisection** (impossible with compass and straightedge).
- **Doubling the cube** (impossible with compass and straightedge).
- Construction of regular heptagon (7-sided polygon).

These axioms were discovered by Jacques Justin (1986), rediscovered by Humiaki Huzita (1991), with axiom O7 found by Koshiro Hatori (2001) and independently by Robert Lang.

---

## Appendix B: Summary of Key Formulas for Simulation Engine Implementation

### Geometric Validation

```
Kawasaki-Justin:    sum_{i odd} alpha_i = pi
Maekawa-Justin:     |M - V| = 2
Big-Little-Big:     if alpha_{i-1} > alpha_i < alpha_{i+1},
                    then MV(crease_i) != MV(crease_{i+1})
```

### Energy Model (Bar-and-Hinge)

```
E_total = E_bar + E_facet + E_fold

E_bar   = sum (1/2) * k_a * (L - L0)^2
E_facet = sum (1/2) * k_f * l * (theta - pi)^2
E_fold  = sum (1/2) * k_c * l * (rho - rho_target)^2

k_a = E*t*w/L0    (axial stiffness)
k_f = E*t^3/(12*(1-nu^2))  (per unit length, flexural)
k_c = crease torsional stiffness (per unit length)
```

### Rigid Origami Kinematics (Degree-4 Vertex)

```
tan(rho_2/2) = -cos((alpha-beta)/2) / cos((alpha+beta)/2) * tan(rho_1/2)
```

### Origami Length Scale

```
L* = B / kappa = (E*t^3/12) / kappa

L* >> panel_size  -->  rigid origami regime
L* << panel_size  -->  flexible shell regime
```

### Deployment Ratio

```
DR = L_deployed / L_stowed
DR_area = A_deployed / A_stowed
```

### Effective Poisson's Ratio (Miura-ori)

```
nu_xy = -((cos(theta)*tan(phi))^2 + sin(theta)^2) /
         (cos(theta)^2*tan(phi)^2 + sin(theta)^2)
```

### Fatigue Damage

```
D(N) = (N / N_f)^p
kappa_N = kappa_0 * (1 - D(N))
```

### Numerical Integration (Verlet, for dynamic simulation)

```
x_{n+1} = 2*x_n - x_{n-1} + F * dt^2 / m
```
