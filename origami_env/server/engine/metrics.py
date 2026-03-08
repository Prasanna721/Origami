"""Metrics computation — compactness, strain, efficiency, deployability, shape."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .paper import PaperState


def compute_all_metrics(paper: PaperState, task: dict, validation: dict) -> dict:
    """Compute every metric from paper state, task, and validation report."""
    bb = paper.bounding_box
    original_area = paper.width * paper.height
    folded_area = _compute_folded_area(bb)
    original_vol = paper.width * paper.height * paper.material.thickness_m
    folded_vol = bb[0] * bb[1] * bb[2] if np.all(bb > 0) else 0.0
    material_vol = original_area * paper.material.thickness_m * paper.num_layers

    deployment_ratio = folded_area / original_area if original_area > 0 else 1.0
    compactness = 1.0 - deployment_ratio

    target_box = task.get("target_box")
    fits = _fits_in_box(bb, target_box) if target_box else None

    max_strain = float(np.max(paper.strain_per_vertex)) if len(paper.strain_per_vertex) > 0 else 0.0
    mean_strain = float(np.mean(paper.strain_per_vertex)) if len(paper.strain_per_vertex) > 0 else 0.0

    return {
        # ── Validity ──────────────────────────────────
        "is_valid": validation.get("is_valid", True),
        "kawasaki_violations": validation.get("kawasaki_violations", 0),
        "kawasaki_total_error": validation.get("kawasaki_total_error", 0.0),
        "maekawa_violations": validation.get("maekawa_violations", 0),
        "self_intersections": validation.get("self_intersections", 0),
        "strain_exceeded": validation.get("strain_exceeded", False),

        # ── Compactness ───────────────────────────────
        "deployment_ratio": deployment_ratio,
        "compactness": compactness,
        "volume_compaction": folded_vol / original_vol if original_vol > 0 else 0.0,
        "packing_efficiency": material_vol / folded_vol if folded_vol > 1e-15 else 0.0,
        "fits_target_box": fits,
        "bounding_box": bb.tolist(),

        # ── Structural ────────────────────────────────
        "max_strain": max_strain,
        "mean_strain": mean_strain,
        "total_energy": paper.energy.get("total", 0.0),
        "energy_bar": paper.energy.get("bar", 0.0),
        "energy_facet": paper.energy.get("facet", 0.0),
        "energy_fold": paper.energy.get("fold", 0.0),
        "material_max_strain": paper.material.max_strain,

        # ── Efficiency ────────────────────────────────
        "fold_count": paper.fold_count,
        "folding_efficiency": compactness / max(paper.fold_count, 1),
        "crease_complexity": _assignment_entropy(paper.edges_assignment),

        # ── Deployability ─────────────────────────────
        "is_deployable": _check_deployability(paper) if task.get("must_deploy") else None,
        "deployment_force_estimate": _estimate_deployment_force(paper),

        # ── Shape similarity ──────────────────────────
        "chamfer_distance": (
            _compute_chamfer_distance(paper, task["target_shape"])
            if "target_shape" in task else None
        ),
        "hausdorff_distance": (
            _compute_hausdorff_distance(paper, task["target_shape"])
            if "target_shape" in task else None
        ),
    }


# ── Compactness helpers ───────────────────────────────

def _compute_folded_area(bounding_box: np.ndarray) -> float:
    """Approximate folded area from bounding box (top-down projection)."""
    return float(bounding_box[0] * bounding_box[1])


def _fits_in_box(bb: np.ndarray, target_box: list | None) -> bool | None:
    """Check if folded bounding box fits inside target box."""
    if target_box is None:
        return None
    target = np.array(target_box)
    # Sort both so orientation doesn't matter
    bb_sorted = np.sort(bb)
    target_sorted = np.sort(target)
    return bool(np.all(bb_sorted <= target_sorted + 1e-6))


# ── Efficiency helpers ────────────────────────────────

def _assignment_entropy(assignments: list[str]) -> float:
    """Shannon entropy of edge assignment distribution (higher = more complex)."""
    if not assignments:
        return 0.0
    counts = Counter(assignments)
    total = len(assignments)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


# ── Deployability ─────────────────────────────────────

def _check_deployability(paper: PaperState) -> bool:
    """
    Simple deployability check: a fold pattern is deployable if
    it can be unfolded without self-intersection.

    Simplified check: if the current state has no self-intersections
    and the fold energy is finite, we consider it deployable.
    """
    if paper.energy.get("total", float("inf")) > 1e6:
        return False
    # A fully flat-foldable pattern (low strain) is generally deployable
    max_strain = float(np.max(paper.strain_per_vertex)) if len(paper.strain_per_vertex) > 0 else 0.0
    return max_strain < paper.material.max_strain * 2.0


def _estimate_deployment_force(paper: PaperState) -> float:
    """
    Estimate force needed to deploy (unfold) the structure.
    Proportional to the total fold energy gradient.
    """
    return paper.energy.get("fold", 0.0) * paper.material.k_axial * 0.01


# ── Shape similarity ─────────────────────────────────

def _compute_chamfer_distance(paper: PaperState, target_shape: dict) -> float:
    """
    Average nearest-point distance between folded vertices and target shape vertices.
    """
    if "vertices" not in target_shape:
        return 0.0

    target_verts = np.array(target_shape["vertices"], dtype=np.float64)
    source_verts = paper.vertices_coords

    # Source to target
    total = 0.0
    for sv in source_verts:
        dists = np.linalg.norm(target_verts - sv, axis=1)
        total += np.min(dists)
    # Target to source
    for tv in target_verts:
        dists = np.linalg.norm(source_verts - tv, axis=1)
        total += np.min(dists)

    n = len(source_verts) + len(target_verts)
    return float(total / n) if n > 0 else 0.0


def _compute_hausdorff_distance(paper: PaperState, target_shape: dict) -> float:
    """Maximum nearest-point distance (worst-case shape error)."""
    if "vertices" not in target_shape:
        return 0.0

    target_verts = np.array(target_shape["vertices"], dtype=np.float64)
    source_verts = paper.vertices_coords

    max_dist = 0.0
    for sv in source_verts:
        dists = np.linalg.norm(target_verts - sv, axis=1)
        max_dist = max(max_dist, np.min(dists))
    for tv in target_verts:
        dists = np.linalg.norm(source_verts - tv, axis=1)
        max_dist = max(max_dist, np.min(dists))

    return float(max_dist)
