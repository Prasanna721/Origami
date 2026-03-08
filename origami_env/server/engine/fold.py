"""Fold operations — apply geometric folds to paper state."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .paper import PaperState, _compute_edge_lengths


class FoldError(Exception):
    """Raised when a fold operation is invalid."""
    pass


def apply_fold(paper: PaperState, fold: dict) -> PaperState:
    """
    Apply one fold operation to the paper.

    fold = {
        "type": "valley" | "mountain" | "pleat" | "crimp",
        "line": {"start": [x, y], "end": [x, y]},
        "angle": 0-180 (degrees),
        "layer_select": "all" | "top" | "bottom" (optional),
    }

    Returns a new PaperState with the fold applied.
    """
    fold_type = fold.get("type", "valley")

    # Handle compound folds
    if fold_type == "pleat":
        return _apply_pleat(paper, fold)
    if fold_type == "crimp":
        return _apply_crimp(paper, fold)

    line = fold.get("line", {})
    start = np.array(line.get("start", [0, 0.5]), dtype=np.float64)
    end = np.array(line.get("end", [1, 0.5]), dtype=np.float64)
    angle_deg = fold.get("angle", 180.0)

    # Validate
    if np.linalg.norm(end - start) < 1e-10:
        raise FoldError("Fold line has zero length")
    if angle_deg <= 0 or angle_deg > 180:
        raise FoldError(f"Fold angle must be in (0, 180], got {angle_deg}")

    # Convert to 3D line points
    line_start_3d = np.array([start[0], start[1], 0.0])
    line_end_3d = np.array([end[0], end[1], 0.0])

    # Check fold line intersects the paper (at least touches the bounding region)
    if not _line_intersects_paper(start, end, paper):
        raise FoldError("Fold line does not intersect the paper")

    # Work on a copy
    new_paper = paper.copy()

    # Step 1: Split faces at fold line (insert new vertices/edges)
    new_paper = _split_faces_at_line(new_paper, start, end)

    # Step 2: Classify vertices — which side of fold line
    line_dir = end - start
    moving_mask = np.zeros(new_paper.num_vertices, dtype=bool)
    for i, v in enumerate(new_paper.vertices_coords):
        # Signed distance from fold line (2D, using cross product)
        to_v = v[:2] - start
        cross = line_dir[0] * to_v[1] - line_dir[1] * to_v[0]
        if cross > 1e-10:
            moving_mask[i] = True
        # Vertices ON the line (|cross| < epsilon) stay fixed (hinge)

    if not np.any(moving_mask):
        raise FoldError("No vertices on the moving side of fold line")

    # Step 3: Apply rotation via quaternion
    fold_angle_rad = math.radians(angle_deg)
    if fold_type == "mountain":
        fold_angle_rad = -fold_angle_rad

    axis = np.array([line_dir[0], line_dir[1], 0.0])
    axis = axis / np.linalg.norm(axis)

    quat = _quaternion_from_axis_angle(axis, fold_angle_rad)

    for i in range(new_paper.num_vertices):
        if moving_mask[i]:
            # Translate to fold line origin, rotate, translate back
            v = new_paper.vertices_coords[i] - line_start_3d
            v = _quaternion_rotate(quat, v)
            new_paper.vertices_coords[i] = v + line_start_3d

    # Step 4: Update edge assignments for new crease edges
    # Find edges that lie along the fold line
    for e_idx in range(new_paper.num_edges):
        v1_idx, v2_idx = new_paper.edges_vertices[e_idx]
        v1 = new_paper.vertices_coords[v1_idx]
        v2 = new_paper.vertices_coords[v2_idx]

        # Check if this edge is on the fold line
        if _edge_on_fold_line(v1[:2], v2[:2], start, end, paper.rest_positions):
            if new_paper.edges_assignment[e_idx] in ("F", "U", "B"):
                if fold_type == "valley":
                    new_paper.edges_assignment[e_idx] = "V"
                    new_paper.edges_foldAngle[e_idx] = angle_deg
                else:
                    new_paper.edges_assignment[e_idx] = "M"
                    new_paper.edges_foldAngle[e_idx] = -angle_deg

    # Step 5: Update layer tracking
    # Count how many vertices moved — gives rough layer estimate
    n_moving = int(np.sum(moving_mask))
    n_total = new_paper.num_vertices
    if n_moving > 0 and n_total > 0:
        # Each fold roughly doubles the layers in the folded region
        new_paper.num_layers = min(new_paper.num_layers + 1, 2 ** new_paper.fold_count + 1)

    # Update face_orders: faces on moving side stack on top
    moving_faces = []
    fixed_faces = []
    for f_idx, face in enumerate(new_paper.faces_vertices):
        face_moving = any(moving_mask[v] for v in face if v < len(moving_mask))
        if face_moving:
            moving_faces.append(f_idx)
        else:
            fixed_faces.append(f_idx)

    # Build ordering pairs: moving faces are on top of fixed faces they overlap with
    new_orders = list(new_paper.face_orders)
    for mf in moving_faces:
        for ff in fixed_faces:
            # +1 means mf is above ff
            new_orders.append((mf, ff, 1))
    new_paper.face_orders = new_orders

    # Step 6: Update fold count
    new_paper.fold_count += 1

    # Step 7: Recompute rest lengths for new edges (if any were added)
    if new_paper.num_edges > len(new_paper.rest_lengths):
        new_paper.rest_lengths = _compute_edge_lengths(
            new_paper.rest_positions, new_paper.edges_vertices
        )

    return new_paper


def _apply_pleat(paper: PaperState, fold: dict) -> PaperState:
    """Pleat = valley fold + mountain fold on parallel lines."""
    line = fold.get("line", {})
    start = np.array(line.get("start", [0, 0.33]), dtype=np.float64)
    end = np.array(line.get("end", [1, 0.33]), dtype=np.float64)
    angle = fold.get("angle", 180.0)

    # Second line offset (default: 1/3 of sheet)
    line2 = fold.get("line2", {})
    if line2:
        start2 = np.array(line2.get("start", [0, 0.66]), dtype=np.float64)
        end2 = np.array(line2.get("end", [1, 0.66]), dtype=np.float64)
    else:
        offset = np.array([0.0, paper.height / 3.0])
        start2 = start + offset
        end2 = end + offset

    paper = apply_fold(paper, {"type": "valley", "line": {"start": start.tolist(), "end": end.tolist()}, "angle": angle})
    paper = apply_fold(paper, {"type": "mountain", "line": {"start": start2.tolist(), "end": end2.tolist()}, "angle": angle})
    return paper


def _apply_crimp(paper: PaperState, fold: dict) -> PaperState:
    """Crimp = mountain fold + valley fold on parallel lines."""
    line = fold.get("line", {})
    start = np.array(line.get("start", [0, 0.33]), dtype=np.float64)
    end = np.array(line.get("end", [1, 0.33]), dtype=np.float64)
    angle = fold.get("angle", 180.0)

    line2 = fold.get("line2", {})
    if line2:
        start2 = np.array(line2.get("start", [0, 0.66]), dtype=np.float64)
        end2 = np.array(line2.get("end", [1, 0.66]), dtype=np.float64)
    else:
        offset = np.array([0.0, paper.height / 3.0])
        start2 = start + offset
        end2 = end + offset

    paper = apply_fold(paper, {"type": "mountain", "line": {"start": start.tolist(), "end": end.tolist()}, "angle": angle})
    paper = apply_fold(paper, {"type": "valley", "line": {"start": start2.tolist(), "end": end2.tolist()}, "angle": angle})
    return paper


# ── Face splitting ────────────────────────────────────

def _split_faces_at_line(paper: PaperState, line_start: np.ndarray, line_end: np.ndarray) -> PaperState:
    """
    Split faces that the fold line passes through.
    Inserts new vertices at intersection points, splits edges and faces.
    """
    line_dir = line_end - line_start

    new_vertices = list(paper.vertices_coords)
    new_rest_positions = list(paper.rest_positions)
    new_edges = list(paper.edges_vertices)
    new_assignments = list(paper.edges_assignment)
    new_fold_angles = list(paper.edges_foldAngle)
    new_faces = []

    # Build edge lookup: for each edge, check if fold line crosses it
    edge_intersections = {}  # edge_idx -> new_vertex_idx

    for e_idx in range(len(new_edges)):
        v1_idx, v2_idx = new_edges[e_idx]
        v1 = new_vertices[v1_idx][:2] if isinstance(new_vertices[v1_idx], np.ndarray) else np.array(new_vertices[v1_idx][:2])
        v2 = new_vertices[v2_idx][:2] if isinstance(new_vertices[v2_idx], np.ndarray) else np.array(new_vertices[v2_idx][:2])

        intersection = _line_segment_intersection(
            line_start, line_end, v1, v2
        )

        if intersection is not None:
            # Check it's not at an existing vertex
            dist_to_v1 = np.linalg.norm(intersection - v1)
            dist_to_v2 = np.linalg.norm(intersection - v2)
            if dist_to_v1 < 1e-8 or dist_to_v2 < 1e-8:
                continue

            # Create new vertex
            new_v_idx = len(new_vertices)
            new_v_3d = np.array([intersection[0], intersection[1], 0.0])
            new_vertices.append(new_v_3d)
            new_rest_positions.append(new_v_3d.copy())
            edge_intersections[e_idx] = new_v_idx

    # Split intersected edges
    edges_to_remove = set()
    for e_idx, new_v_idx in edge_intersections.items():
        v1_idx, v2_idx = new_edges[e_idx]
        asgn = new_assignments[e_idx]
        angle = new_fold_angles[e_idx]

        # Replace original edge with two sub-edges
        edges_to_remove.add(e_idx)
        new_edges.append(np.array([v1_idx, new_v_idx]))
        new_assignments.append(asgn)
        new_fold_angles.append(angle)

        new_edges.append(np.array([new_v_idx, v2_idx]))
        new_assignments.append(asgn)
        new_fold_angles.append(angle)

    # Rebuild face list — split faces that contain intersected edges
    for face in paper.faces_vertices:
        face_intersected_vertices = []

        # Check which edges of this face were split
        for i in range(len(face)):
            v_a = face[i]
            v_b = face[(i + 1) % len(face)]

            # Find the edge index for this face edge
            for e_idx in edge_intersections:
                ev1, ev2 = new_edges[e_idx] if e_idx < paper.num_edges else (new_edges[e_idx][0], new_edges[e_idx][1])
                if e_idx < paper.num_edges:
                    ev1, ev2 = paper.edges_vertices[e_idx]
                    if (ev1 == v_a and ev2 == v_b) or (ev1 == v_b and ev2 == v_a):
                        face_intersected_vertices.append((i, edge_intersections[e_idx]))

        if len(face_intersected_vertices) >= 2:
            # Face is split by fold line — create two sub-faces
            sub_faces = _split_face_with_vertices(face, face_intersected_vertices)
            new_faces.extend(sub_faces)

            # Add edge between the two new vertices (the fold crease edge)
            nv1 = face_intersected_vertices[0][1]
            nv2 = face_intersected_vertices[1][1]
            new_edges.append(np.array([nv1, nv2]))
            new_assignments.append("F")  # will be updated to M/V later
            new_fold_angles.append(0.0)
        else:
            new_faces.append(list(face))

    # Remove old split edges and rebuild arrays
    final_edges = []
    final_assignments = []
    final_fold_angles = []
    for i in range(len(new_edges)):
        if i not in edges_to_remove:
            e = new_edges[i]
            if isinstance(e, np.ndarray):
                final_edges.append(e.tolist())
            else:
                final_edges.append(list(e))
            final_assignments.append(new_assignments[i])
            final_fold_angles.append(new_fold_angles[i])

    paper.vertices_coords = np.array(new_vertices, dtype=np.float64)
    paper.rest_positions = np.array(new_rest_positions, dtype=np.float64)
    paper.edges_vertices = np.array(final_edges, dtype=np.int32) if final_edges else paper.edges_vertices
    paper.edges_assignment = final_assignments if final_assignments else paper.edges_assignment
    paper.edges_foldAngle = np.array(final_fold_angles, dtype=np.float64) if final_fold_angles else paper.edges_foldAngle
    paper.faces_vertices = new_faces if new_faces else paper.faces_vertices
    paper.strain_per_vertex = np.zeros(len(paper.vertices_coords))
    paper.rest_lengths = _compute_edge_lengths(paper.rest_positions, paper.edges_vertices)

    return paper


def _split_face_with_vertices(face: list[int], intersections: list[tuple]) -> list[list[int]]:
    """Split a polygon face into two sub-faces at the intersection points."""
    if len(intersections) < 2:
        return [face]

    # Sort intersections by position along the face
    intersections = sorted(intersections, key=lambda x: x[0])
    (i1, nv1), (i2, nv2) = intersections[0], intersections[1]

    # Build two sub-faces
    face1 = []
    face2 = []

    n = len(face)
    # Face 1: from intersection 1 to intersection 2 (forward)
    face1.append(nv1)
    idx = (i1 + 1) % n
    while idx != (i2 + 1) % n:
        face1.append(face[idx])
        idx = (idx + 1) % n
    face1.append(nv2)

    # Face 2: from intersection 2 to intersection 1 (forward)
    face2.append(nv2)
    idx = (i2 + 1) % n
    while idx != (i1 + 1) % n:
        face2.append(face[idx])
        idx = (idx + 1) % n
    face2.append(nv1)

    result = []
    if len(face1) >= 3:
        result.append(face1)
    if len(face2) >= 3:
        result.append(face2)

    return result if result else [face]


# ── Geometry helpers ──────────────────────────────────

def _line_intersects_paper(start: np.ndarray, end: np.ndarray, paper: PaperState) -> bool:
    """Check if a fold line intersects the paper's bounding region."""
    # Simple check: does the line pass through the paper's bounding box?
    bb_min = paper.vertices_coords[:, :2].min(axis=0)
    bb_max = paper.vertices_coords[:, :2].max(axis=0)

    # Extend line to check intersection with bounding box edges
    corners = [
        (bb_min, np.array([bb_max[0], bb_min[1]])),
        (np.array([bb_max[0], bb_min[1]]), bb_max),
        (bb_max, np.array([bb_min[0], bb_max[1]])),
        (np.array([bb_min[0], bb_max[1]]), bb_min),
    ]

    for c1, c2 in corners:
        if _line_segment_intersection(start, end, c1, c2) is not None:
            return True

    # Also check if line is entirely inside
    for pt in [start, end]:
        if bb_min[0] <= pt[0] <= bb_max[0] and bb_min[1] <= pt[1] <= bb_max[1]:
            return True

    return False


def _line_segment_intersection(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> np.ndarray | None:
    """
    Find intersection point of line (p1-p2) with segment (p3-p4).
    Line is infinite; segment is finite.
    Returns intersection point or None.
    """
    d1 = p2 - p1
    d2 = p4 - p3

    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None  # parallel

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom

    # u must be in [0, 1] for segment intersection (t can be anything for infinite line)
    if 0.0 - 1e-10 <= u <= 1.0 + 1e-10:
        point = p3 + u * d2
        return point

    return None


def _edge_on_fold_line(
    v1: np.ndarray, v2: np.ndarray,
    line_start: np.ndarray, line_end: np.ndarray,
    rest_positions: np.ndarray,
) -> bool:
    """Check if an edge lies approximately along the fold line."""
    line_dir = line_end - line_start
    line_len = np.linalg.norm(line_dir)
    if line_len < 1e-12:
        return False
    line_unit = line_dir / line_len

    # Check both vertices are close to the fold line
    for v in [v1, v2]:
        to_v = v[:2] - line_start[:2] if len(v) > 2 else v - line_start
        # Distance from point to line
        proj = np.dot(to_v, line_unit)
        closest = line_start + proj * line_unit
        dist = np.linalg.norm(v[:2] - closest[:2]) if len(v) > 2 else np.linalg.norm(v - closest)
        if dist > 1e-6:
            return False

    return True


def _quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion [w, x, y, z] from axis and angle."""
    half = angle / 2.0
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = [w, x, y, z]."""
    w, x, y, z = q
    # v' = q * v * q^-1 (using the rotation formula)
    t = 2.0 * np.cross(q[1:], v)
    return v + w * t + np.cross(q[1:], t)
