"""Bar-and-hinge origami physics solver.

CPU implementation (numpy) matching OrigamiSimulator's math:
- Beam axial springs (prevent stretching)
- Crease torsional springs (drive folding via creasePercent)
- Face angle stiffness (prevent mesh collapse)
- Verlet integration with adaptive timestep and per-beam damping

Reference: OrigamiSimulator/js/dynamic/dynamicSolver.js + GLSL shaders
"""

from dataclasses import dataclass, field

import numpy as np

from .fold_parser import parse_fold

# --- Default physics parameters (matching OrigamiSimulator globals.js) ---
AXIAL_STIFFNESS = 20.0
CREASE_STIFFNESS = 0.7
PANEL_STIFFNESS = 0.7
FACE_STIFFNESS = 0.2
DAMPING_RATIO = 0.45
DENSITY = 1.0
MAX_STEPS = 500
CONVERGENCE_THRESHOLD = 1e-6


@dataclass
class SimResult:
    """Result of a physics simulation."""

    positions: np.ndarray  # (N, 3) final vertex positions
    converged: bool
    steps_taken: int
    max_strain: float
    total_energy: float


@dataclass
class _Beam:
    """Axial spring between two vertices."""

    v1: int
    v2: int
    rest_length: float
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass
class _Crease:
    """Torsional spring at a fold edge."""

    edge_idx: int
    v1: int  # first vertex of crease edge
    v2: int  # second vertex of crease edge
    n1: int  # opposite vertex on face 1
    n2: int  # opposite vertex on face 2
    face1: np.ndarray  # face 1 vertex indices
    face2: np.ndarray  # face 2 vertex indices
    target_angle: float  # radians (scaled by creasePercent)
    stiffness: float = 0.0
    damping: float = 0.0
    length: float = 0.0


@dataclass
class _Simulation:
    """Internal simulation state."""

    positions: np.ndarray  # (N, 3) current
    prev_positions: np.ndarray  # (N, 3) previous (for Verlet)
    velocities: np.ndarray  # (N, 3)
    masses: np.ndarray  # (N,)
    beams: list[_Beam] = field(default_factory=list)
    creases: list[_Crease] = field(default_factory=list)
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    rest_angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dt: float = 0.01


def simulate(
    fold_data: dict,
    crease_percent: float = 1.0,
    max_steps: int = MAX_STEPS,
    params: dict | None = None,
) -> SimResult:
    """Simulate a FOLD crease pattern and return final 3D positions.

    Args:
        fold_data: FOLD-format dict with vertices, edges, assignments, angles.
        crease_percent: 0.0 = flat, 1.0 = fully folded.
        max_steps: Maximum solver iterations.
        params: Override default physics parameters.

    Returns:
        SimResult with final positions, convergence info, strain, energy.
    """
    p = {
        "axial_stiffness": AXIAL_STIFFNESS,
        "crease_stiffness": CREASE_STIFFNESS,
        "panel_stiffness": PANEL_STIFFNESS,
        "face_stiffness": FACE_STIFFNESS,
        "damping_ratio": DAMPING_RATIO,
        "density": DENSITY,
    }
    if params:
        p.update(params)

    parsed = parse_fold(fold_data)
    sim = _build_simulation(parsed, crease_percent, p)

    converged = False
    total_energy = 0.0
    prev_energy = float("inf")

    for step in range(max_steps):
        energy = _solve_step(sim, p)
        total_energy = energy

        # Check convergence: energy change is negligible OR velocity is tiny
        if step > 50:
            energy_change = abs(energy - prev_energy) / max(abs(prev_energy), 1e-10)
            max_vel = np.max(np.abs(sim.velocities))
            if energy_change < 0.001 and max_vel < 0.01:
                converged = True
                break
            if max_vel < CONVERGENCE_THRESHOLD:
                converged = True
                break
        prev_energy = energy

    # Compute strain
    max_strain = _compute_max_strain(sim, parsed)

    return SimResult(
        positions=sim.positions.copy(),
        converged=converged,
        steps_taken=step + 1,
        max_strain=max_strain,
        total_energy=total_energy,
    )


def _build_simulation(
    parsed: dict, crease_percent: float, params: dict
) -> _Simulation:
    """Build simulation objects from parsed FOLD data."""
    vertices = parsed["vertices"].copy()
    edges = parsed["edges"]
    assignments = parsed["assignments"]
    fold_angles = parsed["fold_angles"]
    faces = parsed["faces"]

    n_verts = len(vertices)

    # Compute masses from face areas
    masses = np.full(n_verts, 1e-4, dtype=np.float64)
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        mass_contrib = area * params["density"] / 3.0
        for vi in face:
            masses[vi] += mass_contrib

    # Build beams (one per edge)
    beams = []
    for i, (v1, v2) in enumerate(edges):
        rest_length = np.linalg.norm(vertices[v2] - vertices[v1])
        if rest_length < 1e-10:
            rest_length = 1e-10
        k = params["axial_stiffness"] / rest_length
        m_min = min(masses[v1], masses[v2])
        d = params["damping_ratio"] * 2.0 * np.sqrt(k * m_min)
        beams.append(_Beam(v1=v1, v2=v2, rest_length=rest_length, stiffness=k, damping=d))

    # Build creases (fold edges with two adjacent faces)
    face_adj = _build_face_adjacency(edges, faces)
    creases = []
    for i, (v1, v2) in enumerate(edges):
        if assignments[i] not in ("M", "V", "F", "U"):
            continue
        adj = face_adj.get((min(v1, v2), max(v1, v2)), [])
        if len(adj) < 2:
            continue

        # Find opposite vertices in the two adjacent faces
        f1, f2 = adj[0], adj[1]
        n1 = _opposite_vertex(faces[f1], v1, v2)
        n2 = _opposite_vertex(faces[f2], v1, v2)
        if n1 is None or n2 is None:
            continue

        edge_len = np.linalg.norm(vertices[v2] - vertices[v1])
        if assignments[i] in ("M", "V"):
            k = params["crease_stiffness"] * edge_len
        else:
            k = params["panel_stiffness"] * edge_len
        d = params["damping_ratio"] * 2.0 * np.sqrt(abs(k))

        target = fold_angles[i] * crease_percent

        creases.append(
            _Crease(
                edge_idx=i,
                v1=v1,
                v2=v2,
                n1=n1,
                n2=n2,
                face1=faces[f1],
                face2=faces[f2],
                target_angle=target,
                stiffness=k,
                damping=d,
                length=edge_len,
            )
        )

    # Compute rest triangle angles for face stiffness
    rest_angles = np.zeros((len(faces), 3), dtype=np.float64)
    for fi, face in enumerate(faces):
        rest_angles[fi] = _triangle_angles(
            vertices[face[0]], vertices[face[1]], vertices[face[2]]
        )

    # Compute adaptive timestep: dt = 0.9 / (2π × max_freq)
    max_freq = 0.0
    for beam in beams:
        m_min = min(masses[beam.v1], masses[beam.v2])
        if m_min > 0:
            freq = np.sqrt(beam.stiffness / m_min)
            max_freq = max(max_freq, freq)
    if max_freq > 0:
        dt = 0.9 / (2.0 * np.pi * max_freq)
    else:
        dt = 0.005

    return _Simulation(
        positions=vertices.copy(),
        prev_positions=vertices.copy(),
        velocities=np.zeros((n_verts, 3), dtype=np.float64),
        masses=masses,
        beams=beams,
        creases=creases,
        faces=faces,
        rest_angles=rest_angles,
        dt=dt,
    )


def _solve_step(sim: _Simulation, params: dict) -> float:
    """One simulation timestep. Returns total energy."""
    n = len(sim.positions)
    forces = np.zeros((n, 3), dtype=np.float64)
    total_energy = 0.0

    # 1. Beam axial forces
    for beam in sim.beams:
        p1 = sim.positions[beam.v1]
        p2 = sim.positions[beam.v2]
        delta = p2 - p1
        dist = np.linalg.norm(delta)
        if dist < 1e-12:
            continue

        direction = delta / dist
        stretch = dist - beam.rest_length
        f_spring = beam.stiffness * stretch

        # Damping
        v_rel = sim.velocities[beam.v2] - sim.velocities[beam.v1]
        v_along = np.dot(v_rel, direction)
        f_damp = beam.damping * v_along

        f_total = (f_spring + f_damp) * direction
        forces[beam.v1] += f_total
        forces[beam.v2] -= f_total

        total_energy += 0.5 * beam.stiffness * stretch * stretch

    # 2. Crease torsional forces
    for crease in sim.creases:
        theta = _dihedral_angle(sim.positions, crease)
        if theta is None:
            continue

        torque = crease.stiffness * (crease.target_angle - theta)
        total_energy += 0.5 * crease.stiffness * (crease.target_angle - theta) ** 2

        # Distribute torque as forces on the 4 nodes
        _apply_crease_forces(forces, sim.positions, crease, torque)

    # 3. Face angle stiffness
    if params["face_stiffness"] > 0 and len(sim.faces) > 0:
        for fi, face in enumerate(sim.faces):
            angles = _triangle_angles(
                sim.positions[face[0]],
                sim.positions[face[1]],
                sim.positions[face[2]],
            )
            diff = angles - sim.rest_angles[fi]
            _apply_face_forces(
                forces, sim.positions, face, diff, params["face_stiffness"]
            )

    # 4. Verlet integration with position-based damping
    dt = sim.dt
    dt2 = dt * dt
    # Damping factor: 0 = no damping, 1 = full damping
    # Applied as: new_pos = pos + (1-damp) * (pos - prev_pos) + acc * dt²
    damp = params["damping_ratio"] * 0.5  # scale down for Verlet stability
    new_positions = np.zeros_like(sim.positions)

    for i in range(n):
        if sim.masses[i] < 1e-12:
            new_positions[i] = sim.positions[i]
            continue
        acc = forces[i] / sim.masses[i]
        # Clamp acceleration to prevent explosion
        acc_mag = np.linalg.norm(acc)
        if acc_mag > 500.0:
            acc = acc * (500.0 / acc_mag)

        velocity = sim.positions[i] - sim.prev_positions[i]
        new_positions[i] = (
            sim.positions[i] + (1.0 - damp) * velocity + acc * dt2
        )

    # Update velocities
    sim.velocities = (new_positions - sim.positions) / max(dt, 1e-12)

    # Update positions
    sim.prev_positions = sim.positions.copy()
    sim.positions = new_positions

    return total_energy


def _dihedral_angle(
    positions: np.ndarray, crease: _Crease
) -> float | None:
    """Compute dihedral angle using face normals (OrigamiSimulator convention).

    Uses the same formula as OrigamiSimulator's thetaCalcShader:
      x = dot(normal1, normal2)
      y = dot(cross(normal1, crease_vector), normal2)
      theta = atan2(y, x)

    Convention: flat = 0, valley fold = positive, mountain fold = negative.
    """
    p1 = positions[crease.v1]
    p2 = positions[crease.v2]

    edge = p2 - p1
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-12:
        return None
    edge_unit = edge / edge_len

    # Compute face normals from triangle vertices
    normal1 = _face_normal(positions, crease.face1)
    normal2 = _face_normal(positions, crease.face2)
    if normal1 is None or normal2 is None:
        return None

    # OrigamiSimulator thetaCalc formula
    x = np.dot(normal1, normal2)
    y = np.dot(np.cross(normal1, edge_unit), normal2)
    theta = np.arctan2(y, x)

    return theta


def _face_normal(positions: np.ndarray, face: np.ndarray) -> np.ndarray | None:
    """Compute unit normal of a triangular face."""
    a = positions[face[0]]
    b = positions[face[1]]
    c = positions[face[2]]
    normal = np.cross(b - a, c - a)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-12:
        return None
    return normal / n_len


def _apply_crease_forces(
    forces: np.ndarray,
    positions: np.ndarray,
    crease: _Crease,
    torque: float,
) -> None:
    """Distribute crease torque as forces on the 4 crease nodes.

    Uses face normals and moment arms matching OrigamiSimulator's
    velocityCalcShader. Forces on opposite vertices are along
    the face normal direction, scaled by torque / moment_arm.
    """
    p1 = positions[crease.v1]
    p2 = positions[crease.v2]
    pn1 = positions[crease.n1]
    pn2 = positions[crease.n2]

    edge = p2 - p1
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-12:
        return
    edge_unit = edge / edge_len

    # Face normals (from actual triangle geometry)
    normal1 = _face_normal(positions, crease.face1)
    normal2 = _face_normal(positions, crease.face2)
    if normal1 is None or normal2 is None:
        return

    # Moment arms: perpendicular distance from opposite vertex to crease edge
    w1 = pn1 - p1
    w1_along = np.dot(w1, edge_unit)
    w1_perp = w1 - w1_along * edge_unit
    h1 = np.linalg.norm(w1_perp)

    w2 = pn2 - p1
    w2_along = np.dot(w2, edge_unit)
    w2_perp = w2 - w2_along * edge_unit
    h2 = np.linalg.norm(w2_perp)

    if h1 < 1e-12 or h2 < 1e-12:
        return

    # Projection coefficients (where perpendicular from tip meets edge)
    coef1 = np.clip(w1_along / edge_len, 0.0, 1.0)
    coef2 = np.clip(w2_along / edge_len, 0.0, 1.0)

    # Forces on opposite vertices along their face normals
    # Torque > 0 means increase dihedral angle (valley fold)
    # n1 gets pushed along face1 normal, n2 gets pushed opposite to face2 normal
    f_n1 = (torque / h1) * normal1
    f_n2 = -(torque / h2) * normal2

    forces[crease.n1] += f_n1
    forces[crease.n2] += f_n2

    # Reaction forces on edge vertices (Newton's 3rd law, distributed by projection)
    forces[crease.v1] -= f_n1 * (1.0 - coef1) + f_n2 * (1.0 - coef2)
    forces[crease.v2] -= f_n1 * coef1 + f_n2 * coef2


def _triangle_angles(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Compute three interior angles of triangle abc."""
    ab = b - a
    ac = c - a
    bc = c - b

    ab_len = np.linalg.norm(ab)
    ac_len = np.linalg.norm(ac)
    bc_len = np.linalg.norm(bc)

    angles = np.zeros(3, dtype=np.float64)

    if ab_len > 1e-12 and ac_len > 1e-12:
        cos_a = np.clip(np.dot(ab, ac) / (ab_len * ac_len), -1.0, 1.0)
        angles[0] = np.arccos(cos_a)

    if ab_len > 1e-12 and bc_len > 1e-12:
        cos_b = np.clip(np.dot(-ab, bc) / (ab_len * bc_len), -1.0, 1.0)
        angles[1] = np.arccos(cos_b)

    if ac_len > 1e-12 and bc_len > 1e-12:
        cos_c = np.clip(np.dot(-ac, -bc) / (ac_len * bc_len), -1.0, 1.0)
        angles[2] = np.arccos(cos_c)

    return angles


def _apply_face_forces(
    forces: np.ndarray,
    positions: np.ndarray,
    face: np.ndarray,
    angle_diff: np.ndarray,
    stiffness: float,
) -> None:
    """Apply face stiffness forces to maintain triangle angles.

    Follows OrigamiSimulator's gradient-based approach.
    """
    a, b, c = positions[face[0]], positions[face[1]], positions[face[2]]

    # Compute face normal
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-12:
        return
    normal /= n_len

    ab_len = np.linalg.norm(ab)
    ac_len = np.linalg.norm(ac)
    bc = c - b
    bc_len = np.linalg.norm(bc)

    # Angle gradients (rotation perpendicular to edge, in plane of face)
    scaled_diff = angle_diff * stiffness

    # Force on vertex a (from angles A, B, C)
    if ac_len > 1e-12:
        cross_ac = np.cross(normal, ac / ac_len) / ac_len
    else:
        cross_ac = np.zeros(3)

    if ab_len > 1e-12:
        cross_ab = np.cross(normal, ab / ab_len) / ab_len
    else:
        cross_ab = np.zeros(3)

    if bc_len > 1e-12:
        cross_bc = np.cross(normal, bc / bc_len) / bc_len
    else:
        cross_bc = np.zeros(3)

    forces[face[0]] -= scaled_diff[0] * (cross_ac - cross_ab)
    forces[face[0]] -= scaled_diff[1] * cross_ab
    forces[face[0]] += scaled_diff[2] * cross_ac

    forces[face[1]] += scaled_diff[0] * cross_ab
    forces[face[1]] -= scaled_diff[1] * (cross_bc + cross_ab)
    forces[face[1]] -= scaled_diff[2] * cross_bc

    forces[face[2]] -= scaled_diff[0] * cross_ac
    forces[face[2]] += scaled_diff[1] * cross_bc
    forces[face[2]] += scaled_diff[2] * (cross_ac + cross_bc)


def _build_face_adjacency(
    edges: np.ndarray, faces: np.ndarray
) -> dict[tuple[int, int], list[int]]:
    """Map each edge (sorted vertex pair) to list of face indices."""
    adj: dict[tuple[int, int], list[int]] = {}

    for fi, face in enumerate(faces):
        n = len(face)
        for j in range(n):
            v1, v2 = int(face[j]), int(face[(j + 1) % n])
            key = (min(v1, v2), max(v1, v2))
            if key not in adj:
                adj[key] = []
            adj[key].append(fi)

    return adj


def _opposite_vertex(face: np.ndarray, v1: int, v2: int) -> int | None:
    """Find the vertex in a face that is NOT v1 or v2."""
    for v in face:
        if v != v1 and v != v2:
            return int(v)
    return None


def _compute_max_strain(sim: _Simulation, parsed: dict) -> float:
    """Compute maximum axial strain across all beams."""
    max_strain = 0.0
    for beam in sim.beams:
        dist = np.linalg.norm(sim.positions[beam.v2] - sim.positions[beam.v1])
        if beam.rest_length > 1e-12:
            strain = abs(dist - beam.rest_length) / beam.rest_length
            max_strain = max(max_strain, strain)
    return max_strain
