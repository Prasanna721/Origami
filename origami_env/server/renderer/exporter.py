"""FOLD JSON export, OBJ export for 3D printing / external renderers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..engine.paper import PaperState


def export_fold_json(paper: PaperState, fold_history: list | None = None) -> dict:
    """
    Full FOLD JSON with metadata.
    Compatible with OrigamiSimulator and other FOLD-aware tools.
    """
    fold_data = paper.to_fold_json()

    # Add FOLD metadata
    fold_data["file_spec"] = 1.1
    fold_data["file_creator"] = "origami_env"
    fold_data["file_classes"] = ["singleModel"]
    fold_data["frame_classes"] = ["foldedForm"]

    # Custom metadata
    fold_data["origami_env:material"] = paper.material.to_dict()
    fold_data["origami_env:fold_count"] = paper.fold_count
    fold_data["origami_env:strain_per_vertex"] = paper.strain_per_vertex.tolist()
    fold_data["origami_env:energy"] = paper.energy

    if fold_history:
        fold_data["origami_env:fold_history"] = fold_history

    return fold_data


def export_obj(paper: PaperState) -> str:
    """
    Wavefront OBJ format string.
    Faces are triangulated for compatibility.
    """
    lines = []
    lines.append("# Origami Environment Export")
    lines.append(f"# Fold count: {paper.fold_count}")
    lines.append(f"# Material: {paper.material.name}")
    lines.append("")

    # Vertices
    for v in paper.vertices_coords:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    lines.append("")

    # Vertex normals (compute per-face normals, average at vertices)
    normals = _compute_vertex_normals(paper)
    for n in normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    lines.append("")

    # Faces (1-indexed in OBJ format)
    triangles = paper.triangulated_faces
    for tri in triangles:
        v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
        lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")

    return "\n".join(lines) + "\n"


def save_fold_json(paper: PaperState, output_path: str, fold_history: list | None = None) -> str:
    """Export FOLD JSON and save to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = export_fold_json(paper, fold_history)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path


def save_obj(paper: PaperState, output_path: str) -> str:
    """Export OBJ and save to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    obj_str = export_obj(paper)
    with open(output_path, "w") as f:
        f.write(obj_str)
    return output_path


def _compute_vertex_normals(paper: PaperState) -> np.ndarray:
    """Compute per-vertex normals by averaging incident face normals."""
    verts = paper.vertices_coords
    normals = np.zeros_like(verts)

    triangles = paper.triangulated_faces
    for tri in triangles:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(face_normal)
        if norm > 1e-12:
            face_normal /= norm
        for idx in tri:
            normals[idx] += face_normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return normals
