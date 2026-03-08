"""Bar-and-hinge physics solver — NumPy port of Ghassaei's GPU solver."""
from __future__ import annotations

import numpy as np

from .paper import PaperState


# ── Default solver parameters ─────────────────────────
# These are *relative* stiffness scales. The actual stiffness
# is derived from material properties but capped for stability.

DEFAULT_K_AXIAL_SCALE = 70.0    # bar stiffness scale
DEFAULT_K_FACET_SCALE = 0.2     # facet hinge stiffness scale
DEFAULT_K_FOLD_SCALE = 0.7      # fold crease stiffness scale
DEFAULT_DT = 0.005
DEFAULT_DAMPING = 0.15
DEFAULT_N_STEPS = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-10
MAX_FORCE_MAG = 100.0           # clamp individual force magnitude


def simulate(
    paper: PaperState,
    fold_percent: float = 1.0,
    n_steps: int = DEFAULT_N_STEPS,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
) -> PaperState:
    """
    Run bar-and-hinge physics simulation.

    Updates vertex positions to satisfy fold constraints.
    Computes strain per vertex and energy breakdown.

    Three constraint types:
      1. BAR (axial spring) — every edge resists stretching
      2. FACET HINGE — triangulation diagonals keep faces flat
      3. FOLD HINGE — crease edges drive toward target fold angle
    """
    pos = paper.vertices_coords.copy()
    last_pos = pos.copy()

    # Build constraint lists
    beams = _build_beams(paper)
    creases = _build_creases(paper)

    if not beams and not creases:
        paper.strain_per_vertex = compute_strain(
            pos, paper.edges_vertices, paper.rest_lengths
        )
        return paper

    for step in range(n_steps):
        forces = np.zeros_like(pos)

        # ── Beam forces (axial springs) ───────────────
        for (a, b, L0, k) in beams:
            delta = pos[b] - pos[a]
            L = np.linalg.norm(delta)
            if L < 1e-12:
                continue
            strain = (L - L0) / L0
            F_mag = k * strain * L0
            # Clamp force magnitude
            F_mag = np.clip(F_mag, -MAX_FORCE_MAG, MAX_FORCE_MAG)
            F_dir = delta / L
            forces[a] += F_mag * F_dir
            forces[b] -= F_mag * F_dir

        # ── Crease forces (dihedral angle springs) ────
        for (n1, n2, n3, n4, target_angle, k, ctype) in creases:
            actual_target = target_angle * fold_percent if ctype == "fold" else target_angle

            theta = _compute_dihedral_angle(pos[n1], pos[n2], pos[n3], pos[n4])
            delta_theta = theta - actual_target

            edge_len = np.linalg.norm(pos[n2] - pos[n1])
            if edge_len < 1e-12:
                continue

            torque = k * edge_len * delta_theta
            # Clamp torque
            torque = np.clip(torque, -MAX_FORCE_MAG, MAX_FORCE_MAG)

            f1, f2, f3, f4 = _torque_to_forces(
                pos[n1], pos[n2], pos[n3], pos[n4], torque
            )
            forces[n1] += f1
            forces[n2] += f2
            forces[n3] += f3
            forces[n4] += f4

        # Clamp total force per vertex
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        mask = force_mags > MAX_FORCE_MAG
        if np.any(mask):
            scale = np.where(mask, MAX_FORCE_MAG / (force_mags + 1e-12), 1.0)
            forces *= scale

        # ── Verlet integration ────────────────────────
        velocity = (1.0 - damping) * (pos - last_pos)
        new_pos = pos + velocity + forces * dt * dt
        last_pos = pos.copy()
        pos = new_pos

        # Check for NaN/Inf and abort if detected
        if not np.all(np.isfinite(pos)):
            pos = last_pos
            break

        # ── Convergence check ─────────────────────────
        kinetic_energy = np.sum((pos - last_pos) ** 2)
        if kinetic_energy < DEFAULT_CONVERGENCE_THRESHOLD:
            break

    # Update paper state
    paper.vertices_coords = pos
    paper.strain_per_vertex = compute_strain(pos, paper.edges_vertices, paper.rest_lengths)
    paper.energy = _compute_energy_breakdown(pos, beams, creases, fold_percent)

    return paper


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


# ── Constraint builders ───────────────────────────────

def _build_beams(paper: PaperState) -> list[tuple]:
    """Build beam (bar) constraints from edges."""
    # Use normalized stiffness — we cap k to prevent instability
    # Characteristic length = average edge length
    avg_len = float(np.mean(paper.rest_lengths[paper.rest_lengths > 1e-12])) if np.any(paper.rest_lengths > 1e-12) else 1.0
    k_base = min(paper.material.k_axial * DEFAULT_K_AXIAL_SCALE, 1e4) / avg_len

    beams = []
    for e_idx in range(paper.num_edges):
        v1, v2 = paper.edges_vertices[e_idx]
        L0 = paper.rest_lengths[e_idx]
        if L0 < 1e-12:
            continue
        # Stiffness proportional to 1/L0 but capped
        k = min(k_base / L0, 1e4)
        beams.append((v1, v2, L0, k))
    return beams


def _build_creases(paper: PaperState) -> list[tuple]:
    """
    Build crease constraints (fold hinges + facet hinges).
    Each crease needs 4 nodes: n1-n2 (hinge edge), n3 and n4 (wing tips).
    """
    creases = []
    triangles = paper.triangulated_faces

    # Build edge-to-triangle adjacency
    edge_to_tris = {}
    for t_idx, tri in enumerate(triangles):
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edge_to_tris.setdefault(e, []).append((t_idx, tri))

    k_fold = min(paper.material.k_facet * DEFAULT_K_FOLD_SCALE, 1e3)
    k_facet = min(paper.material.k_facet * DEFAULT_K_FACET_SCALE, 1e3)

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

        n3 = wing_a[0]
        n4 = wing_b[0]

        asgn = paper.edges_assignment[e_idx]

        if asgn in ("M", "V"):
            target = np.radians(paper.edges_foldAngle[e_idx])
            creases.append((v1, v2, n3, n4, target, k_fold, "fold"))
        elif asgn in ("F", "U"):
            creases.append((v1, v2, n3, n4, np.pi, k_facet, "facet"))

    return creases


# ── Dihedral angle computation ────────────────────────

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


def _torque_to_forces(p1, p2, p3, p4, torque) -> tuple:
    """Convert torque around hinge edge (p1-p2) to forces on all 4 nodes."""
    e = p2 - p1
    e_len = np.linalg.norm(e)
    if e_len < 1e-12:
        zero = np.zeros(3)
        return zero, zero, zero, zero

    e_unit = e / e_len

    # Project wing vertices onto hinge line to get lever arms
    arm3 = p3 - p1 - np.dot(p3 - p1, e_unit) * e_unit
    arm4 = p4 - p1 - np.dot(p4 - p1, e_unit) * e_unit

    dist3 = np.linalg.norm(arm3)
    dist4 = np.linalg.norm(arm4)

    if dist3 < 1e-12 or dist4 < 1e-12:
        zero = np.zeros(3)
        return zero, zero, zero, zero

    # Force on wing nodes — perpendicular to arm, in fold direction
    f3 = (torque / dist3) * np.cross(e_unit, arm3 / dist3)
    f4 = -(torque / dist4) * np.cross(e_unit, arm4 / dist4)

    # Forces on hinge nodes balance the wing forces
    f_total = f3 + f4
    f1 = -f_total * 0.5
    f2 = -f_total * 0.5

    return f1, f2, f3, f4


def _compute_energy_breakdown(
    pos: np.ndarray,
    beams: list[tuple],
    creases: list[tuple],
    fold_percent: float,
) -> dict:
    """Compute energy for each constraint type."""
    e_bar = 0.0
    for (a, b, L0, k) in beams:
        L = np.linalg.norm(pos[b] - pos[a])
        e_bar += 0.5 * k * (L - L0) ** 2

    e_facet = 0.0
    e_fold = 0.0
    for (n1, n2, n3, n4, target, k, ctype) in creases:
        theta = _compute_dihedral_angle(pos[n1], pos[n2], pos[n3], pos[n4])
        edge_len = np.linalg.norm(pos[n2] - pos[n1])
        actual_target = target * fold_percent if ctype == "fold" else target
        energy = 0.5 * k * edge_len * (theta - actual_target) ** 2

        if ctype == "fold":
            e_fold += energy
        else:
            e_facet += energy

    return {
        "total": float(e_bar + e_facet + e_fold),
        "bar": float(e_bar),
        "facet": float(e_facet),
        "fold": float(e_fold),
    }
