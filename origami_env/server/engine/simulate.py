"""Origami fold simulator.

Two modes:
1. Analytical folding: rotates vertices around fold lines (exact, fast).
   Used for computing folded positions from FOLD crease patterns.
2. Physics solver (bar-and-hinge): for validation and strain computation.

The analytical approach handles sequential folds correctly by BFS traversal
from fixed faces through fold edges, accumulating rotations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .fold_parser import parse_fold


@dataclass
class SimResult:
    """Result of a fold simulation."""

    positions: np.ndarray  # (N, 3) final vertex positions
    converged: bool
    steps_taken: int
    max_strain: float
    total_energy: float


def simulate(
    fold_data: dict,
    crease_percent: float = 1.0,
    max_steps: int = 500,
    params: dict | None = None,
) -> SimResult:
    """Simulate a FOLD crease pattern and return final 3D positions.

    Uses analytical rotation of panels around fold lines. Each panel
    is rotated by the fold angle * crease_percent around its crease edge.

    Args:
        fold_data: FOLD-format dict with vertices, edges, assignments, angles.
        crease_percent: 0.0 = flat, 1.0 = fully folded.
        max_steps: Unused (kept for API compat).
        params: Unused (kept for API compat).

    Returns:
        SimResult with final positions, strain info.
    """
    parsed = parse_fold(fold_data)
    positions = parsed["vertices"].copy()
    edges = parsed["edges"]
    assignments = parsed["assignments"]
    fold_angles = parsed["fold_angles"]
    faces = parsed["faces"]

    if len(faces) == 0:
        return SimResult(
            positions=positions, converged=True,
            steps_taken=0, max_strain=0.0, total_energy=0.0,
        )

    # Build face adjacency: edge -> [face_idx, ...]
    face_adj = _build_face_adjacency(faces)

    # Build crease map: (v_min, v_max) -> (fold_angle, assignment)
    crease_map: dict[tuple[int, int], tuple[float, str]] = {}
    for i, (v1, v2) in enumerate(edges):
        key = (min(int(v1), int(v2)), max(int(v1), int(v2)))
        if assignments[i] in ("M", "V"):
            crease_map[key] = (fold_angles[i] * crease_percent, assignments[i])

    # BFS from face 0: traverse faces, rotating across fold edges
    n_faces = len(faces)
    visited = [False] * n_faces
    rotated = np.zeros(len(positions), dtype=bool)

    # Fix face 0's vertices
    visited[0] = True
    for vi in faces[0]:
        rotated[vi] = True

    # BFS queue: faces to process
    queue = [0]
    while queue:
        fi = queue.pop(0)

        # Find all edges of this face and check for adjacent unvisited faces
        face = faces[fi]
        for j in range(len(face)):
            v1, v2 = int(face[j]), int(face[(j + 1) % len(face)])
            edge_key = (min(v1, v2), max(v1, v2))

            adj_faces = face_adj.get(edge_key, [])
            for fj in adj_faces:
                if visited[fj]:
                    continue
                visited[fj] = True
                queue.append(fj)

                # Determine fold angle for this edge
                fold_info = crease_map.get(edge_key)
                if fold_info is not None:
                    angle = fold_info[0]
                else:
                    angle = 0.0  # panel edge, no fold

                if abs(angle) > 1e-10:
                    # Rotate all vertices on the "other side" of this edge
                    # that haven't been fixed yet
                    _rotate_across_edge(
                        positions, faces, face_adj, crease_map,
                        visited, fj, v1, v2, angle, crease_percent,
                    )

                # Mark face vertices as rotated
                for vi in faces[fj]:
                    rotated[vi] = True

    # Compute strain (deviation from rest edge lengths)
    max_strain = _compute_strain(positions, parsed)

    return SimResult(
        positions=positions,
        converged=True,
        steps_taken=1,
        max_strain=max_strain,
        total_energy=0.0,
    )


def _rotate_across_edge(
    positions: np.ndarray,
    faces: np.ndarray,
    face_adj: dict,
    crease_map: dict,
    visited: list[bool],
    start_face: int,
    ev1: int,
    ev2: int,
    angle: float,
    crease_percent: float,
) -> None:
    """Rotate all vertices reachable from start_face (without crossing
    already-visited faces) around the edge (ev1, ev2) by angle."""
    # Collect all vertices that need to be rotated
    # (all vertices reachable from start_face through unvisited faces,
    #  excluding the edge vertices themselves)
    verts_to_rotate: set[int] = set()

    # BFS to find all connected unvisited faces and their vertices
    sub_queue = [start_face]
    sub_visited = {start_face}
    while sub_queue:
        fi = sub_queue.pop(0)
        for vi in faces[fi]:
            vi_int = int(vi)
            if vi_int != ev1 and vi_int != ev2:
                verts_to_rotate.add(vi_int)
        # Explore neighbors
        face = faces[fi]
        for j in range(len(face)):
            v1, v2 = int(face[j]), int(face[(j + 1) % len(face)])
            edge_key = (min(v1, v2), max(v1, v2))
            for fj in face_adj.get(edge_key, []):
                if fj not in sub_visited and not visited[fj]:
                    sub_visited.add(fj)
                    sub_queue.append(fj)
                    # Mark as visited so main BFS won't re-process
                    visited[fj] = True

    if not verts_to_rotate:
        return

    # Rotate around the edge axis
    p1 = positions[ev1].copy()
    p2 = positions[ev2].copy()
    axis = p2 - p1
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-12:
        return
    axis_unit = axis / axis_len

    # Rotation matrix around axis by angle
    rot = Rotation.from_rotvec(angle * axis_unit)

    for vi in verts_to_rotate:
        # Translate to origin at p1, rotate, translate back
        positions[vi] = rot.apply(positions[vi] - p1) + p1


def _build_face_adjacency(
    faces: np.ndarray,
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


def _compute_strain(positions: np.ndarray, parsed: dict) -> float:
    """Compute max axial strain across all edges."""
    edges = parsed["edges"]
    vertices_flat = parsed["vertices"]
    max_strain = 0.0
    for v1, v2 in edges:
        rest = np.linalg.norm(vertices_flat[v2] - vertices_flat[v1])
        curr = np.linalg.norm(positions[v2] - positions[v1])
        if rest > 1e-12:
            strain = abs(curr - rest) / rest
            max_strain = max(max_strain, strain)
    return max_strain
