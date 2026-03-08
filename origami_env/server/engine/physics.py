"""Bar-and-hinge physics solver — NumPy port of Ghassaei's GPU solver.

Matches OrigamiSimulator's force model:
  - Axial bar springs with per-beam velocity damping
  - Crease/facet dihedral hinges with projection coefficients
  - Face stiffness (triangle angle preservation)
  - Adaptive timestep from natural frequency analysis

Uses **dimensionless** stiffness parameters (like OrigamiSimulator's sliders)
so the simulation works correctly regardless of coordinate scale.
"""
from __future__ import annotations

import math

import numpy as np

from .paper import PaperState


# ── Default solver parameters (dimensionless, matching OrigamiSimulator sliders) ──
# OrigamiSimulator uses dimensionless stiffness that self-scale with geometry.
# k_beam = K_AXIAL / L0,  k_crease = K_FOLD * edgeLen,  mass = area/3
DEFAULT_K_AXIAL = 1.0            # axial bar stiffness (dimensionless)
DEFAULT_K_FOLD = 0.7             # fold crease stiffness (dimensionless)
DEFAULT_K_FACET = 0.7            # facet hinge stiffness (dimensionless)
DEFAULT_K_FACE = 0.2             # face (triangle angle) stiffness (dimensionless)
DEFAULT_DT = 0.1                 # base timestep (adaptive overrides this)
DEFAULT_DAMPING = 0.45           # per-beam critical damping ratio
DEFAULT_VELOCITY_DAMPING = 0.99  # global velocity decay per step (OrigamiSimulator default)
DEFAULT_N_STEPS = 200
DEFAULT_CONVERGENCE_THRESHOLD = 1e-10
MAX_FORCE_MAG = 5.0              # clamp individual force magnitude


def simulate(
    paper: PaperState,
    fold_percent: float = 1.0,
    n_steps: int = DEFAULT_N_STEPS,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
) -> PaperState:
    """
    Run bar-and-hinge physics simulation with gradual fold ramping.

    Like OrigamiSimulator, ramps crease percent from 0→target over the
    simulation to prevent instability from large angle jumps.

    Four constraint types:
      1. BAR (axial spring) — every edge resists stretching, with per-beam damping
      2. FACET HINGE — triangulation diagonals keep faces flat
      3. FOLD HINGE — crease edges drive toward target fold angle
      4. FACE — triangle interior angles resist shearing
    """
    pos = paper.vertices_coords.copy()
    last_pos = pos.copy()

    # Build constraint lists
    triangles = paper.triangulated_faces
    vertex_masses = _compute_vertex_masses(paper, triangles)
    beams = _build_beams(paper, vertex_masses, damping)
    creases = _build_creases(paper)
    face_constraints = _build_face_constraints(paper, triangles)

    if not beams and not creases and not face_constraints:
        paper.strain_per_vertex = compute_strain(
            pos, paper.edges_vertices, paper.rest_lengths
        )
        return paper

    dt = _compute_adaptive_dt(beams, vertex_masses, dt)
    vd = DEFAULT_VELOCITY_DAMPING

    for step in range(n_steps):
        # Ramp fold_percent gradually from 0 → target over first half of steps
        ramp = min(1.0, (step + 1) / (n_steps * 0.5)) * fold_percent

        forces = _compute_forces(pos, last_pos, dt, beams, creases, face_constraints, ramp)

        # Clamp total force per vertex
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        mask = force_mags > MAX_FORCE_MAG
        if np.any(mask):
            forces *= np.where(mask, MAX_FORCE_MAG / (force_mags + 1e-12), 1.0)

        # Verlet integration with global velocity damping
        # OrigamiSimulator: nextPos = F*dt²/m + (1+d)*pos - d*lastPos
        accel = forces / np.maximum(vertex_masses, 1e-15)[:, np.newaxis]
        new_pos = accel * (dt * dt) + (1.0 + vd) * pos - vd * last_pos
        last_pos = pos.copy()
        pos = new_pos

        if not np.all(np.isfinite(pos)):
            pos = last_pos
            break

        # Convergence check (only after ramp is complete)
        if ramp >= fold_percent:
            kinetic_energy = np.sum((pos - last_pos) ** 2)
            if kinetic_energy < DEFAULT_CONVERGENCE_THRESHOLD:
                break

    paper.vertices_coords = pos
    paper.strain_per_vertex = compute_strain(pos, paper.edges_vertices, paper.rest_lengths)
    paper.energy = _compute_energy_breakdown(pos, beams, creases, face_constraints, fold_percent, paper)
    return paper


def _compute_forces(
    pos: np.ndarray,
    last_pos: np.ndarray,
    dt: float,
    beams: list[tuple],
    creases: list[tuple],
    face_constraints: list[tuple],
    fold_percent: float,
) -> np.ndarray:
    """Compute all physics forces for one timestep."""
    forces = np.zeros_like(pos)
    velocity = (pos - last_pos) / dt if dt > 1e-15 else np.zeros_like(pos)

    # ── Beam forces (axial springs + per-beam damping) ──
    for (a, b, L0, k, d) in beams:
        delta = pos[b] - pos[a]
        L = np.linalg.norm(delta)
        if L < 1e-12:
            continue
        F_spring = k * delta * (1.0 - L0 / L)
        F_spring = np.clip(F_spring, -MAX_FORCE_MAG, MAX_FORCE_MAG)
        delta_v = velocity[b] - velocity[a]
        F_total = F_spring + d * delta_v
        forces[a] += F_total
        forces[b] -= F_total

    # ── Crease forces (dihedral angle springs) ──
    for crease in creases:
        (n1, n2, n3, n4, target_angle, k, ctype, h1, h2,
         coef1_n3, coef1_n4, coef2_n3, coef2_n4) = crease

        actual_target = target_angle * fold_percent if ctype == "fold" else target_angle
        theta = _compute_dihedral_angle(pos[n1], pos[n2], pos[n3], pos[n4])
        delta_theta = actual_target - theta
        # Wrap to [-π, π] to take shortest rotation path
        delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi

        edge_vec = pos[n2] - pos[n1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue

        ang_force = k * edge_len * delta_theta
        ang_force = np.clip(ang_force, -MAX_FORCE_MAG, MAX_FORCE_MAG)

        normal1 = np.cross(pos[n3] - pos[n1], pos[n2] - pos[n1])
        n1_len = np.linalg.norm(normal1)
        normal2 = np.cross(pos[n2] - pos[n1], pos[n4] - pos[n1])
        n2_len = np.linalg.norm(normal2)
        if n1_len < 1e-12 or n2_len < 1e-12:
            continue
        normal1 /= n1_len
        normal2 /= n2_len

        e_unit = edge_vec / edge_len
        arm3 = pos[n3] - pos[n1] - np.dot(pos[n3] - pos[n1], e_unit) * e_unit
        arm4 = pos[n4] - pos[n1] - np.dot(pos[n4] - pos[n1], e_unit) * e_unit
        dist3 = np.linalg.norm(arm3)
        dist4 = np.linalg.norm(arm4)
        if dist3 < 1e-12 or dist4 < 1e-12:
            continue

        forces[n3] += (ang_force / dist3) * normal1
        forces[n4] += (ang_force / dist4) * normal2
        forces[n1] += -ang_force * (coef1_n3 / dist3 * normal1 + coef1_n4 / dist4 * normal2)
        forces[n2] += -ang_force * (coef2_n3 / dist3 * normal1 + coef2_n4 / dist4 * normal2)

    # ── Face forces (triangle angle preservation) ──
    for (v0, v1, v2, a0_nom, a1_nom, a2_nom) in face_constraints:
        p0, p1, p2 = pos[v0], pos[v1], pos[v2]
        ab = p1 - p0
        ac = p2 - p0
        bc = p2 - p1

        len_ab = np.linalg.norm(ab)
        len_ac = np.linalg.norm(ac)
        len_bc = np.linalg.norm(bc)
        if len_ab < 1e-7 or len_ac < 1e-7 or len_bc < 1e-7:
            continue

        ab_n = ab / len_ab
        ac_n = ac / len_ac
        bc_n = bc / len_bc

        cos_a0 = np.clip(np.dot(ab_n, ac_n), -1.0, 1.0)
        cos_a1 = np.clip(-np.dot(ab_n, bc_n), -1.0, 1.0)
        cos_a2 = np.clip(np.dot(ac_n, bc_n), -1.0, 1.0)

        diff0 = (a0_nom - math.acos(cos_a0)) * DEFAULT_K_FACE
        diff1 = (a1_nom - math.acos(cos_a1)) * DEFAULT_K_FACE
        diff2 = (a2_nom - math.acos(cos_a2)) * DEFAULT_K_FACE

        face_normal = np.cross(ab, ac)
        fn_len = np.linalg.norm(face_normal)
        if fn_len < 1e-12:
            continue
        face_normal /= fn_len

        nxac = np.cross(face_normal, ac_n) / len_ac
        nxab = np.cross(face_normal, ab_n) / len_ab
        nxbc = np.cross(face_normal, bc_n) / len_bc

        forces[v0] -= diff0 * (nxac - nxab) + diff1 * nxab - diff2 * nxac
        forces[v1] -= diff0 * nxab - diff1 * (nxab + nxbc) + diff2 * nxbc
        forces[v2] += diff0 * nxac - diff1 * nxbc + diff2 * (nxbc - nxac)

    return forces


def compute_strain(
    vertices: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
) -> np.ndarray:
    """
    Per-vertex Cauchy strain = average percent deviation of incident edge lengths.
    Ghassaei's formula: strain_v = mean(|L - L0| / L0) for all edges at vertex v.
    """
    n_verts = len(vertices)
    strain = np.zeros(n_verts)
    counts = np.zeros(n_verts)

    for e_idx in range(len(edges)):
        v1, v2 = edges[e_idx]
        L = np.linalg.norm(vertices[v1] - vertices[v2])
        L0 = rest_lengths[e_idx]
        if L0 < 1e-12:
            continue
        edge_strain = abs(L - L0) / L0

        strain[v1] += edge_strain
        strain[v2] += edge_strain
        counts[v1] += 1
        counts[v2] += 1

    mask = counts > 0
    strain[mask] /= counts[mask]
    return strain


# ── Vertex mass computation ──────────────────────────

def _compute_vertex_masses(
    paper: PaperState,
    triangles: list[list[int]],
) -> np.ndarray:
    """
    Compute per-vertex mass from triangle areas (dimensionless).

    mass_v = sum(area_t / 3) for each incident triangle.
    No material density/thickness — keeps physics scale-invariant
    (matching OrigamiSimulator's approach).
    """
    n_verts = paper.num_vertices
    masses = np.zeros(n_verts, dtype=np.float64)
    rest = paper.rest_positions

    for tri in triangles:
        if len(tri) < 3:
            continue
        v0, v1, v2 = tri[0], tri[1], tri[2]
        e1 = rest[v1] - rest[v0]
        e2 = rest[v2] - rest[v0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        # Distribute 1/3 of triangle area-mass to each vertex
        masses[v0] += area / 3.0
        masses[v1] += area / 3.0
        masses[v2] += area / 3.0

    # Ensure no zero masses (for isolated vertices)
    min_mass = np.min(masses[masses > 0]) if np.any(masses > 0) else 1e-6
    masses[masses < 1e-15] = min_mass

    return masses


# ── Adaptive timestep ────────────────────────────────

def _compute_adaptive_dt(
    beams: list[tuple],
    vertex_masses: np.ndarray,
    default_dt: float,
) -> float:
    """
    Compute adaptive timestep from max natural frequency.
    dt = min(0.9 / (2*pi*maxFreq), default_dt)
    where maxFreq = max(sqrt(k / min_mass)) over all beams.

    Matches OrigamiSimulator beam.getNaturalFrequency().
    """
    if not beams:
        return default_dt

    max_freq_sq = 0.0
    for (a, b, L0, k, d) in beams:
        min_mass = min(vertex_masses[a], vertex_masses[b])
        if min_mass < 1e-15:
            continue
        freq_sq = k / min_mass
        if freq_sq > max_freq_sq:
            max_freq_sq = freq_sq

    if max_freq_sq <= 0:
        return default_dt

    max_freq = math.sqrt(max_freq_sq)
    adaptive_dt = 0.9 / (2.0 * math.pi * max_freq)

    return min(adaptive_dt, default_dt)


# ── Constraint builders ──────────────────────────────

def _build_beams(
    paper: PaperState,
    vertex_masses: np.ndarray,
    damping: float,
) -> list[tuple]:
    """
    Build beam (bar) constraints from edges.

    Each beam is (a, b, L0, k, d) where:
      k = K_AXIAL / L0   (dimensionless, matches OrigamiSimulator beam.getK())
      d = damping * 2 * sqrt(k * minMass)  (OrigamiSimulator beam.getD())
    """
    beams = []
    for e_idx in range(paper.num_edges):
        v1, v2 = int(paper.edges_vertices[e_idx][0]), int(paper.edges_vertices[e_idx][1])
        L0 = paper.rest_lengths[e_idx]
        if L0 < 1e-12:
            continue

        # OrigamiSimulator: k = axialStiffness / currentLength
        k = DEFAULT_K_AXIAL / L0

        # Per-beam damping coefficient
        min_mass = min(vertex_masses[v1], vertex_masses[v2])
        d = damping * 2.0 * math.sqrt(k * min_mass)

        beams.append((v1, v2, L0, k, d))
    return beams


def _build_creases(paper: PaperState) -> list[tuple]:
    """
    Build crease constraints (fold hinges + facet hinges).
    Each crease needs 4 nodes: n1-n2 (hinge edge), n3 and n4 (wing tips).

    Precomputes projection coefficients for proper force distribution
    matching OrigamiSimulator's Crease.getCoef().
    """
    creases = []
    triangles = paper.triangulated_faces
    rest = paper.rest_positions

    # Build edge-to-triangle adjacency
    edge_to_tris = {}
    for t_idx, tri in enumerate(triangles):
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_to_tris.setdefault(e, []).append((t_idx, tri))

    k_fold = DEFAULT_K_FOLD
    k_facet = DEFAULT_K_FACET

    for e_idx in range(paper.num_edges):
        v1, v2 = int(paper.edges_vertices[e_idx][0]), int(paper.edges_vertices[e_idx][1])
        edge_key = tuple(sorted([v1, v2]))

        adj_tris = edge_to_tris.get(edge_key, [])
        if len(adj_tris) != 2:
            continue  # boundary or non-manifold

        # Find wing vertices
        tri_a = adj_tris[0][1]
        tri_b = adj_tris[1][1]

        wing_a = [v for v in tri_a if v != v1 and v != v2]
        wing_b = [v for v in tri_b if v != v1 and v != v2]

        if not wing_a or not wing_b:
            continue

        n3 = wing_a[0]  # wing node on face 1
        n4 = wing_b[0]  # wing node on face 2

        # Compute projection coefficients from rest positions
        # OrigamiSimulator crease.js getCoef():
        #   vector1 = edgeVector from edgeNode, normalized
        #   vector2 = wingNode - edgeNode
        #   projLength = dot(vector1, vector2)
        #   coef = 1 - projLength / creaseLength
        #
        # We compute coef for each (wingNode, edgeNode) pair.
        # coef1_n3 = coef(n3, v1), coef1_n4 = coef(n4, v1)  -> for hinge node v1
        # coef2_n3 = coef(n3, v2), coef2_n4 = coef(n4, v2)  -> for hinge node v2
        edge_vec = rest[v2] - rest[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue
        edge_unit = edge_vec / edge_len

        # Coefficients for node n3 (wing node 1)
        # From edge node v1:
        vec_n3_from_v1 = rest[n3] - rest[v1]
        proj_n3_v1 = np.dot(edge_unit, vec_n3_from_v1)
        coef_n3_at_v1 = 1.0 - proj_n3_v1 / edge_len

        # From edge node v2 (vector goes v1->v2, so from v2: vector = v1-v2 = -edge_vec)
        vec_n3_from_v2 = rest[n3] - rest[v2]
        proj_n3_v2 = np.dot(-edge_unit, vec_n3_from_v2)
        coef_n3_at_v2 = 1.0 - proj_n3_v2 / edge_len

        # Coefficients for node n4 (wing node 2)
        vec_n4_from_v1 = rest[n4] - rest[v1]
        proj_n4_v1 = np.dot(edge_unit, vec_n4_from_v1)
        coef_n4_at_v1 = 1.0 - proj_n4_v1 / edge_len

        vec_n4_from_v2 = rest[n4] - rest[v2]
        proj_n4_v2 = np.dot(-edge_unit, vec_n4_from_v2)
        coef_n4_at_v2 = 1.0 - proj_n4_v2 / edge_len

        asgn = paper.edges_assignment[e_idx]

        if asgn in ("M", "V"):
            target = np.radians(paper.edges_foldAngle[e_idx])
            creases.append((
                v1, v2, n3, n4, target, k_fold, "fold",
                0.0, 0.0,  # h1, h2 placeholders (computed dynamically)
                coef_n3_at_v1, coef_n4_at_v1,  # coefficients for hinge node v1
                coef_n3_at_v2, coef_n4_at_v2,  # coefficients for hinge node v2
            ))
        elif asgn in ("F", "U"):
            creases.append((
                v1, v2, n3, n4, np.pi, k_facet, "facet",
                0.0, 0.0,
                coef_n3_at_v1, coef_n4_at_v1,
                coef_n3_at_v2, coef_n4_at_v2,
            ))

    return creases


def _build_face_constraints(
    paper: PaperState,
    triangles: list[list[int]],
) -> list[tuple]:
    """
    Build face (triangle angle preservation) constraints.

    For each triangulated face, records vertex indices and nominal interior
    angles computed from rest positions.

    Returns list of (v0, v1, v2, angle0, angle1, angle2).
    """
    rest = paper.rest_positions
    constraints = []

    for tri in triangles:
        if len(tri) < 3:
            continue
        v0, v1, v2 = tri[0], tri[1], tri[2]
        p0, p1, p2 = rest[v0], rest[v1], rest[v2]

        ab = p1 - p0
        ac = p2 - p0
        bc = p2 - p1

        len_ab = np.linalg.norm(ab)
        len_ac = np.linalg.norm(ac)
        len_bc = np.linalg.norm(bc)

        if len_ab < 1e-12 or len_ac < 1e-12 or len_bc < 1e-12:
            continue

        ab_n = ab / len_ab
        ac_n = ac / len_ac
        bc_n = bc / len_bc

        # Nominal angles at each vertex
        # angle at v0 = angle between ab and ac
        a0 = math.acos(float(np.clip(np.dot(ab_n, ac_n), -1.0, 1.0)))
        # angle at v1 = angle between ba and bc = acos(-dot(ab, bc))
        a1 = math.acos(float(np.clip(-np.dot(ab_n, bc_n), -1.0, 1.0)))
        # angle at v2 = angle between ca and cb = acos(dot(ac, bc))
        a2 = math.acos(float(np.clip(np.dot(ac_n, bc_n), -1.0, 1.0)))

        constraints.append((v0, v1, v2, a0, a1, a2))

    return constraints


# ── Dihedral angle computation ───────────────────────

def _compute_dihedral_angle(p1, p2, p3, p4) -> float:
    """
    Compute dihedral angle between planes (p1,p2,p3) and (p1,p2,p4).

         p3
        / | \\
       /  |  \\
    p1----+----p2   (hinge edge)
       \\  |  /
        \\ | /
         p4
    """
    e = p2 - p1
    e_len = np.linalg.norm(e)
    if e_len < 1e-12:
        return np.pi

    n1 = np.cross(p3 - p1, e)
    n2 = np.cross(e, p4 - p1)

    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)

    if n1_len < 1e-12 or n2_len < 1e-12:
        return np.pi

    n1 = n1 / n1_len
    n2 = n2 / n2_len

    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
    sin_theta = np.dot(np.cross(n1, n2), e / e_len)

    return np.arctan2(sin_theta, cos_theta)


def _compute_energy_breakdown(
    pos: np.ndarray,
    beams: list[tuple],
    creases: list[tuple],
    face_constraints: list[tuple],
    fold_percent: float,
    paper: PaperState,
) -> dict:
    """Compute energy for each constraint type."""
    e_bar = 0.0
    for (a, b, L0, k, d) in beams:
        L = np.linalg.norm(pos[b] - pos[a])
        e_bar += 0.5 * k * (L - L0) ** 2

    e_facet = 0.0
    e_fold = 0.0
    for crease in creases:
        (n1, n2, n3, n4, target, k, ctype,
         _h1, _h2, _c1n3, _c1n4, _c2n3, _c2n4) = crease
        theta = _compute_dihedral_angle(pos[n1], pos[n2], pos[n3], pos[n4])
        edge_len = np.linalg.norm(pos[n2] - pos[n1])
        actual_target = target * fold_percent if ctype == "fold" else target
        delta = (actual_target - theta + math.pi) % (2 * math.pi) - math.pi
        energy = 0.5 * k * edge_len * delta ** 2

        if ctype == "fold":
            e_fold += energy
        else:
            e_facet += energy

    e_face = 0.0
    k_face = DEFAULT_K_FACE
    for (v0, v1, v2, a0_nom, a1_nom, a2_nom) in face_constraints:
        p0, p1, p2 = pos[v0], pos[v1], pos[v2]

        ab = p1 - p0
        ac = p2 - p0
        bc = p2 - p1

        len_ab = np.linalg.norm(ab)
        len_ac = np.linalg.norm(ac)
        len_bc = np.linalg.norm(bc)

        if len_ab < 1e-12 or len_ac < 1e-12 or len_bc < 1e-12:
            continue

        ab_n = ab / len_ab
        ac_n = ac / len_ac
        bc_n = bc / len_bc

        a0_cur = math.acos(float(np.clip(np.dot(ab_n, ac_n), -1.0, 1.0)))
        a1_cur = math.acos(float(np.clip(-np.dot(ab_n, bc_n), -1.0, 1.0)))
        a2_cur = math.acos(float(np.clip(np.dot(ac_n, bc_n), -1.0, 1.0)))

        e_face += 0.5 * k_face * ((a0_nom - a0_cur) ** 2 + (a1_nom - a1_cur) ** 2 + (a2_nom - a2_cur) ** 2)

    return {
        "total": float(e_bar + e_facet + e_fold + e_face),
        "bar": float(e_bar),
        "facet": float(e_facet),
        "fold": float(e_fold),
        "face": float(e_face),
    }
