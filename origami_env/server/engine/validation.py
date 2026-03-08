"""Validation — Kawasaki, Maekawa, self-intersection checks."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .paper import PaperState


def validate_state(paper: PaperState) -> dict:
    """Run all validation checks. Returns violation report."""
    kawasaki_v, kawasaki_err = _check_kawasaki(paper)
    maekawa_v = _check_maekawa(paper)
    self_int = _check_self_intersections(paper)
    max_strain = float(np.max(paper.strain_per_vertex)) if len(paper.strain_per_vertex) > 0 else 0.0
    strain_exceeded = max_strain > paper.material.max_strain

    return {
        "kawasaki_violations": kawasaki_v,
        "kawasaki_total_error": kawasaki_err,
        "maekawa_violations": maekawa_v,
        "self_intersections": self_int,
        "strain_exceeded": strain_exceeded,
        "max_strain_ratio": max_strain / paper.material.max_strain if paper.material.max_strain > 0 else 0.0,
        "is_valid": (
            kawasaki_v == 0 and
            maekawa_v == 0 and
            self_int == 0 and
            not strain_exceeded
        ),
    }


def _check_kawasaki(paper: PaperState) -> tuple[int, float]:
    """
    Kawasaki-Justin theorem: at each interior vertex,
    the alternating sum of sector angles equals 0
    (equivalently, even angles sum = odd angles sum = 180 degrees).

    Returns (violation_count, total_error_degrees).
    """
    # Build vertex adjacency: which edges are incident on each vertex
    vertex_edges = defaultdict(list)
    for e_idx in range(paper.num_edges):
        v1, v2 = paper.edges_vertices[e_idx]
        vertex_edges[int(v1)].append(e_idx)
        vertex_edges[int(v2)].append(e_idx)

    # Identify interior vertices (not on boundary edges only)
    boundary_vertices = set()
    for e_idx in range(paper.num_edges):
        if paper.edges_assignment[e_idx] == "B":
            v1, v2 = paper.edges_vertices[e_idx]
            boundary_vertices.add(int(v1))
            boundary_vertices.add(int(v2))

    interior_vertices = set()
    for v_idx in range(paper.num_vertices):
        edges = vertex_edges[v_idx]
        has_crease = any(
            paper.edges_assignment[e] in ("M", "V")
            for e in edges
        )
        if has_crease and v_idx not in boundary_vertices:
            interior_vertices.add(v_idx)

    violations = 0
    total_error = 0.0

    for v_idx in interior_vertices:
        edges = vertex_edges[v_idx]
        crease_edges = [
            e for e in edges
            if paper.edges_assignment[e] in ("M", "V")
        ]

        if len(crease_edges) < 2:
            continue

        # Get angles of crease edges around the vertex
        angles = []
        for e_idx in crease_edges:
            v1, v2 = paper.edges_vertices[e_idx]
            other = int(v2) if int(v1) == v_idx else int(v1)
            # Use rest positions (flat) for angle computation
            p_center = paper.rest_positions[v_idx][:2]
            p_other = paper.rest_positions[other][:2]
            angle = math.atan2(p_other[1] - p_center[1], p_other[0] - p_center[0])
            angles.append(angle)

        angles.sort()

        if len(angles) < 2:
            continue

        # Compute sector angles (gaps between consecutive crease lines)
        sectors = []
        for i in range(len(angles)):
            sector = angles[(i + 1) % len(angles)] - angles[i]
            if sector < 0:
                sector += 2 * math.pi
            sectors.append(sector)

        # Kawasaki condition: alternating sum = 0
        alt_sum = sum(s * (-1) ** i for i, s in enumerate(sectors))
        error = abs(alt_sum)
        error_deg = math.degrees(error)

        if error_deg > 1.0:  # 1 degree tolerance
            violations += 1
            total_error += error_deg

    return violations, total_error


def _check_maekawa(paper: PaperState) -> int:
    """
    Maekawa-Justin theorem: at each interior vertex,
    |number_of_mountain - number_of_valley| = 2.

    Returns violation count.
    """
    vertex_edges = defaultdict(list)
    for e_idx in range(paper.num_edges):
        v1, v2 = paper.edges_vertices[e_idx]
        vertex_edges[int(v1)].append(e_idx)
        vertex_edges[int(v2)].append(e_idx)

    violations = 0
    checked = set()

    for v_idx in range(paper.num_vertices):
        edges = vertex_edges[v_idx]

        # Count M and V edges at this vertex
        m_count = sum(1 for e in edges if paper.edges_assignment[e] == "M")
        v_count = sum(1 for e in edges if paper.edges_assignment[e] == "V")

        # Only check vertices that have crease lines
        total_creases = m_count + v_count
        if total_creases < 2:
            continue

        # Skip boundary vertices
        has_boundary = any(paper.edges_assignment[e] == "B" for e in edges)
        if has_boundary:
            continue

        if v_idx in checked:
            continue
        checked.add(v_idx)

        if abs(m_count - v_count) != 2:
            violations += 1

    return violations


def _check_self_intersections(paper: PaperState) -> int:
    """
    Detect face-through-face penetrations (NOT stacked layers).

    In folded origami, faces legitimately overlap in projection — this is
    the whole point of folding. A "self-intersection" means two faces
    PASS THROUGH each other, which is physically impossible.

    We detect this by checking if two non-adjacent triangles' planes
    cross each other at significantly different heights (z-separation),
    indicating one punches through the other rather than lying flat on top.
    """
    triangles = paper.triangulated_faces
    n_tris = len(triangles)

    if n_tris < 2:
        return 0

    # Characteristic scale: use the paper's thickness * num_layers as tolerance
    # Stacked layers within this range are expected, not intersections
    thickness = paper.material.thickness_m
    stack_tol = thickness * max(paper.num_layers, 2) * 10  # generous tolerance
    stack_tol = max(stack_tol, 0.05)  # minimum 5cm tolerance

    # Pre-compute triangle data
    tri_sets = [set(tri) for tri in triangles]
    tri_coords = [paper.vertices_coords[tri] for tri in triangles]
    tri_normals = []
    for tc in tri_coords:
        n = np.cross(tc[1] - tc[0], tc[2] - tc[0])
        nlen = np.linalg.norm(n)
        tri_normals.append(n / nlen if nlen > 1e-12 else np.array([0, 0, 1.0]))

    count = 0
    for i in range(n_tris):
        for j in range(i + 1, n_tris):
            # Skip if triangles share any vertex
            if tri_sets[i] & tri_sets[j]:
                continue

            ta, tb = tri_coords[i], tri_coords[j]

            # Quick bounding box check
            a_min, a_max = ta.min(axis=0), ta.max(axis=0)
            b_min, b_max = tb.min(axis=0), tb.max(axis=0)
            if np.any(a_max < b_min - stack_tol) or np.any(b_max < a_min - stack_tol):
                continue

            # Check Z-separation: if triangles are at similar Z heights,
            # they're just stacked layers (expected in folded origami)
            z_a = ta[:, 2]
            z_b = tb[:, 2]
            z_gap = min(z_a.max(), z_b.max()) - max(z_a.min(), z_b.min())
            if abs(z_gap) < stack_tol:
                # Z ranges overlap within tolerance — stacked, not intersecting
                continue

            # Check for genuine face-through-face penetration:
            # Both triangle's centroids must be on OPPOSITE sides of
            # the other triangle's plane, with significant penetration depth
            c_a = ta.mean(axis=0)
            c_b = tb.mean(axis=0)

            # Distance from centroid A to plane of B
            d_a_to_b = abs(np.dot(tri_normals[j], c_a - tb[0]))
            # Distance from centroid B to plane of A
            d_b_to_a = abs(np.dot(tri_normals[i], c_b - ta[0]))

            # Both centroids very close to other's plane = genuine pass-through
            # Must be closer than stack tolerance (which accounts for legitimate stacking)
            penetration_depth = min(d_a_to_b, d_b_to_a)
            if penetration_depth > stack_tol:
                # Centroids are far from each other's planes — not penetrating
                continue

            # Final check: normals must be significantly non-parallel
            # Parallel faces are stacked layers, not intersections
            dot = abs(np.dot(tri_normals[i], tri_normals[j]))
            if dot > 0.3:
                continue

            count += 1

    return count
