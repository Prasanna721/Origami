"""Paper state representation — FOLD-format compatible."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .materials import Material, MATERIALS


@dataclass
class PaperState:
    """Core origami state. All geometry stored in FOLD-compatible arrays."""

    # ── Geometry (FOLD format) ────────────────────────
    vertices_coords: np.ndarray          # (N, 3) 3D positions
    edges_vertices: np.ndarray           # (E, 2) edge connectivity (int)
    faces_vertices: list[list[int]]      # ragged: polygon vertex indices (CCW)
    edges_assignment: list[str]          # (E,) "M"|"V"|"B"|"F"|"U"
    edges_foldAngle: np.ndarray          # (E,) target fold angle in degrees

    # ── Physics reference ─────────────────────────────
    rest_lengths: np.ndarray             # (E,) original edge lengths
    rest_positions: np.ndarray           # (N, 3) flat sheet positions (strain reference)
    strain_per_vertex: np.ndarray        # (N,) per-vertex Cauchy strain
    energy: dict = field(default_factory=lambda: {
        "total": 0.0, "bar": 0.0, "facet": 0.0, "fold": 0.0, "face": 0.0,
    })

    # ── Layers ────────────────────────────────────────
    face_orders: list[tuple] = field(default_factory=list)
    num_layers: int = 1

    # ── Material ──────────────────────────────────────
    material: Material = field(default_factory=lambda: MATERIALS["paper"])

    # ── Sheet dimensions ──────────────────────────────
    width: float = 1.0
    height: float = 1.0
    fold_count: int = 0

    # ── Cache ─────────────────────────────────────────
    _cached_triangulated_faces: Optional[list] = field(default=None, repr=False)

    # ── Properties ────────────────────────────────────

    @property
    def num_vertices(self) -> int:
        return len(self.vertices_coords)

    @property
    def num_edges(self) -> int:
        return len(self.edges_vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces_vertices)

    @property
    def num_mountain(self) -> int:
        return self.edges_assignment.count("M")

    @property
    def num_valley(self) -> int:
        return self.edges_assignment.count("V")

    @property
    def bounding_box(self) -> np.ndarray:
        """(3,) min bounding box dimensions of the folded state."""
        if len(self.vertices_coords) == 0:
            return np.zeros(3)
        mins = self.vertices_coords.min(axis=0)
        maxs = self.vertices_coords.max(axis=0)
        return maxs - mins

    @property
    def triangulated_faces(self) -> list[list[int]]:
        """Ear-clipping triangulation of all polygon faces (cached)."""
        if self._cached_triangulated_faces is None:
            triangles = []
            for face in self.faces_vertices:
                triangles.extend(_triangulate_face(face, self.vertices_coords))
            self._cached_triangulated_faces = triangles
        return self._cached_triangulated_faces

    def __setattr__(self, name, value):
        """Invalidate triangulation cache when faces_vertices changes."""
        super().__setattr__(name, value)
        if name == "faces_vertices":
            super().__setattr__("_cached_triangulated_faces", None)

    # ── Serialization ─────────────────────────────────

    def to_fold_json(self) -> dict:
        """Export as FOLD-format JSON dict."""
        result = {
            "file_spec": 1.1,
            "file_creator": "OrigamiRL",
            "frame_classes": ["creasePattern"],
            "frame_attributes": ["3D"],
            "vertices_coords": self.vertices_coords.tolist(),
            "edges_vertices": self.edges_vertices.tolist(),
            "edges_assignment": list(self.edges_assignment),
            "edges_foldAngle": self.edges_foldAngle.tolist(),
            "faces_vertices": [list(f) for f in self.faces_vertices],
        }
        return result

    @classmethod
    def from_fold_json(
        cls,
        data: dict,
        material: Optional[Material] = None,
        source: str = "auto",
    ) -> PaperState:
        """Import from FOLD JSON dict.

        Parameters
        ----------
        data : dict
            FOLD-format JSON dict.
        material : Material, optional
            Override material.
        source : str
            ``"auto"`` (default) auto-detects OrigamiSimulator FOLD files,
            ``"origami_simulator"`` forces the coord-swap path,
            any other value skips detection.
        """
        coords = np.array(data["vertices_coords"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((len(coords), 1))])

        # ── OrigamiSimulator coordinate swap detection ────────
        is_origami_sim = source == "origami_simulator"
        if source == "auto" and not is_origami_sim:
            creator = data.get("file_creator", "")
            if "origami simulator" in creator.lower():
                is_origami_sim = True
            elif len(coords) > 0:
                # Heuristic: all Y near zero but Z is non-zero
                y_near_zero = np.allclose(coords[:, 1], 0.0, atol=1e-6)
                z_nonzero = not np.allclose(coords[:, 2], 0.0, atol=1e-6)
                if y_near_zero and z_nonzero:
                    is_origami_sim = True

        if is_origami_sim:
            # OrigamiSimulator uses [x, y, z] where y=up; we want [x, z, y]
            coords = coords[:, [0, 2, 1]]

        edges = np.array(data["edges_vertices"], dtype=np.int32)
        assignments = data.get("edges_assignment", ["U"] * len(edges))

        # Handle both "edges_foldAngle" (singular) and "edges_foldAngles" (plural, OrigamiSimulator compat)
        fold_angles = np.zeros(len(edges), dtype=np.float64)
        raw_angles = data.get("edges_foldAngle") or data.get("edges_foldAngles")
        if raw_angles is not None:
            for i, a in enumerate(raw_angles):
                fold_angles[i] = float(a) if a is not None else 0.0

        faces = data.get("faces_vertices", [])
        if not faces:
            faces = _faces_from_edges(coords, edges)

        rest_lengths = _compute_edge_lengths(coords, edges)

        mat = material or MATERIALS["paper"]

        return cls(
            vertices_coords=coords,
            edges_vertices=edges,
            faces_vertices=faces,
            edges_assignment=assignments,
            edges_foldAngle=fold_angles,
            rest_lengths=rest_lengths,
            rest_positions=coords.copy(),
            strain_per_vertex=np.zeros(len(coords)),
            material=mat,
            width=float(coords[:, 0].max() - coords[:, 0].min()),
            height=float(coords[:, 1].max() - coords[:, 1].min()),
        )

    def to_observation_dict(self) -> dict:
        """Simplified dict for the Observation (sent to LLM / frontend)."""
        bb = self.bounding_box
        return {
            "vertices_coords": self.vertices_coords.tolist(),
            "edges_vertices": self.edges_vertices.tolist(),
            "faces_vertices": [list(f) for f in self.faces_vertices],
            "edges_assignment": list(self.edges_assignment),
            "edges_foldAngle": self.edges_foldAngle.tolist(),
            "rest_positions": self.rest_positions.tolist(),
            "triangulated_faces": self.triangulated_faces,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "num_faces": self.num_faces,
            "bounding_box": bb.tolist(),
            "num_layers": self.num_layers,
            "width": self.width,
            "height": self.height,
            "material": self.material.to_dict(),
            "strain_per_vertex": self.strain_per_vertex.tolist(),
            "energy": dict(self.energy),
            "fold_count": self.fold_count,
        }

    def copy(self) -> PaperState:
        """Deep copy."""
        return PaperState(
            vertices_coords=self.vertices_coords.copy(),
            edges_vertices=self.edges_vertices.copy(),
            faces_vertices=[list(f) for f in self.faces_vertices],
            edges_assignment=list(self.edges_assignment),
            edges_foldAngle=self.edges_foldAngle.copy(),
            rest_lengths=self.rest_lengths.copy(),
            rest_positions=self.rest_positions.copy(),
            strain_per_vertex=self.strain_per_vertex.copy(),
            energy=dict(self.energy),
            face_orders=list(self.face_orders),
            num_layers=self.num_layers,
            material=self.material,
            width=self.width,
            height=self.height,
            fold_count=self.fold_count,
            _cached_triangulated_faces=None,
        )


def create_flat_sheet(
    width: float = 1.0,
    height: float = 1.0,
    material: Material | str = "paper",
    subdivisions: int = 0,
) -> PaperState:
    """
    Create a flat rectangular sheet.

    subdivisions=0: 4 vertices, 5 edges (4 boundary + 1 diagonal), 2 triangle faces
    subdivisions=N: (N+1)^2 vertices grid for higher-resolution physics
    """
    if isinstance(material, str):
        material = MATERIALS[material]

    if subdivisions == 0:
        # Simple quad split into 2 triangles
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ], dtype=np.float64)

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # boundary
            [0, 2],  # diagonal (flat/triangulation)
        ], dtype=np.int32)

        assignments = ["B", "B", "B", "B", "F"]
        fold_angles = np.zeros(5, dtype=np.float64)
        faces = [[0, 1, 2], [0, 2, 3]]
    else:
        # Grid subdivision
        n = subdivisions + 1
        vertices = []
        for j in range(n + 1):
            for i in range(n + 1):
                vertices.append([
                    width * i / n,
                    height * j / n,
                    0.0,
                ])
        vertices = np.array(vertices, dtype=np.float64)

        edges_list = []
        assignments = []
        faces = []

        def idx(i, j):
            return j * (n + 1) + i

        # Horizontal edges
        for j in range(n + 1):
            for i in range(n):
                edges_list.append([idx(i, j), idx(i + 1, j)])
                assignments.append("B" if j == 0 or j == n else "F")

        # Vertical edges
        for j in range(n):
            for i in range(n + 1):
                edges_list.append([idx(i, j), idx(i, j + 1)])
                assignments.append("B" if i == 0 or i == n else "F")

        # Diagonal edges + faces
        for j in range(n):
            for i in range(n):
                edges_list.append([idx(i, j), idx(i + 1, j + 1)])
                assignments.append("F")
                faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
                faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])

        edges = np.array(edges_list, dtype=np.int32)
        fold_angles = np.zeros(len(edges), dtype=np.float64)

    rest_lengths = _compute_edge_lengths(vertices, edges)

    return PaperState(
        vertices_coords=vertices,
        edges_vertices=edges,
        faces_vertices=faces,
        edges_assignment=assignments,
        edges_foldAngle=fold_angles,
        rest_lengths=rest_lengths,
        rest_positions=vertices.copy(),
        strain_per_vertex=np.zeros(len(vertices)),
        material=material,
        width=width,
        height=height,
    )


# ── Helpers ───────────────────────────────────────────

def _compute_edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute euclidean length of each edge."""
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]
    return np.linalg.norm(v2 - v1, axis=1)


def _triangulate_face(face: list[int], vertices: np.ndarray) -> list[list[int]]:
    """Simple ear-clipping triangulation of a polygon face."""
    if len(face) <= 3:
        return [list(face)]

    triangles = []
    remaining = list(face)

    max_iters = len(remaining) * 2
    iters = 0
    while len(remaining) > 3 and iters < max_iters:
        iters += 1
        found_ear = False
        for i in range(len(remaining)):
            prev_idx = remaining[(i - 1) % len(remaining)]
            curr_idx = remaining[i]
            next_idx = remaining[(i + 1) % len(remaining)]

            # Check convexity (CCW winding)
            v_prev = vertices[prev_idx][:2]
            v_curr = vertices[curr_idx][:2]
            v_next = vertices[next_idx][:2]

            cross = (v_curr[0] - v_prev[0]) * (v_next[1] - v_prev[1]) - \
                    (v_curr[1] - v_prev[1]) * (v_next[0] - v_prev[0])

            if cross <= 0:
                continue

            # Check no other vertex inside this triangle
            is_ear = True
            for j, v_idx in enumerate(remaining):
                if v_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if _point_in_triangle_2d(
                    vertices[v_idx][:2], v_prev, v_curr, v_next
                ):
                    is_ear = False
                    break

            if is_ear:
                triangles.append([prev_idx, curr_idx, next_idx])
                remaining.pop(i)
                found_ear = True
                break

        if not found_ear:
            # Fallback: fan triangulation from first vertex
            for i in range(1, len(remaining) - 1):
                triangles.append([remaining[0], remaining[i], remaining[i + 1]])
            break

    if len(remaining) == 3:
        triangles.append(remaining)

    return triangles


def _point_in_triangle_2d(p, a, b, c) -> bool:
    """Check if point p is inside triangle (a, b, c) in 2D."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def _faces_from_edges(vertices: np.ndarray, edges: np.ndarray) -> list[list[int]]:
    """
    Reconstruct faces from edge connectivity for planar graphs.

    Uses half-edge traversal: at each vertex, follow edges in CCW order
    to trace minimal face cycles.
    """
    from collections import defaultdict
    import math

    n_verts = len(vertices)
    if n_verts < 3:
        return []

    # Build adjacency: for each vertex, list of neighbors
    adj = defaultdict(set)
    for v1, v2 in edges:
        adj[int(v1)].add(int(v2))
        adj[int(v2)].add(int(v1))

    # For each vertex, sort neighbors by angle (CCW)
    sorted_adj = {}
    for v, neighbors in adj.items():
        if not neighbors:
            continue
        center = vertices[v][:2]
        angle_pairs = []
        for nb in neighbors:
            nb_pos = vertices[nb][:2]
            angle = math.atan2(nb_pos[1] - center[1], nb_pos[0] - center[0])
            angle_pairs.append((angle, nb))
        angle_pairs.sort()
        sorted_adj[v] = [nb for _, nb in angle_pairs]

    # Half-edge traversal: for each directed edge (u, v),
    # the "next" edge is the one after u in v's sorted neighbor list
    def next_halfedge(u, v):
        neighbors = sorted_adj.get(v, [])
        if not neighbors:
            return v, u
        try:
            idx = neighbors.index(u)
        except ValueError:
            return v, neighbors[0]
        # Next edge in CCW order from v is the neighbor BEFORE u (CW = previous)
        prev_idx = (idx - 1) % len(neighbors)
        return v, neighbors[prev_idx]

    # Trace all faces
    visited_halfedges = set()
    faces = []

    for v1, v2 in edges:
        for u, v in [(int(v1), int(v2)), (int(v2), int(v1))]:
            if (u, v) in visited_halfedges:
                continue

            # Trace face
            face = []
            cu, cv = u, v
            max_steps = n_verts + 2
            steps = 0
            while steps < max_steps:
                if (cu, cv) in visited_halfedges:
                    break
                visited_halfedges.add((cu, cv))
                face.append(cu)
                cu, cv = next_halfedge(cu, cv)
                steps += 1
                if cu == u and cv == v:
                    break

            if len(face) >= 3 and len(face) <= n_verts:
                # Check it's a valid bounded face (not the outer face)
                # Compute signed area — positive = CCW = interior face
                area = 0.0
                for i in range(len(face)):
                    p1 = vertices[face[i]][:2]
                    p2 = vertices[face[(i + 1) % len(face)]][:2]
                    area += p1[0] * p2[1] - p2[0] * p1[1]

                if area > 1e-10:  # CCW = interior face
                    faces.append(face)

    # Fallback: if no faces found, use Delaunay triangulation
    if not faces:
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(vertices[:, :2])
            faces = [list(simplex) for simplex in tri.simplices]
        except Exception:
            # Last resort: fan triangulation from vertex 0
            for i in range(1, n_verts - 1):
                faces.append([0, i, i + 1])

    return faces
