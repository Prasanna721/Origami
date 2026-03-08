"""3D folded state rendering via matplotlib."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from ..engine.paper import PaperState


def render_folded_state(
    paper: PaperState,
    output_path: str | None = None,
    view_angle: tuple = (30, 45),
    figsize: tuple = (8, 6),
    dpi: int = 150,
    show_strain: bool = True,
) -> Image.Image:
    """
    3D wireframe + face shading with strain colors.
    Blue (0 strain) -> Yellow -> Red (max strain).
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    verts = paper.vertices_coords
    triangles = paper.triangulated_faces

    if triangles and show_strain:
        # Compute per-face strain (average of vertex strains)
        face_strains = []
        for tri in triangles:
            s = np.mean([paper.strain_per_vertex[v] for v in tri])
            face_strains.append(s)

        max_strain = max(face_strains) if face_strains else 1e-6
        if max_strain < 1e-10:
            max_strain = 1e-6

        # Create colored face collection
        norm = plt.Normalize(0, max_strain)
        cmap = cm.get_cmap("coolwarm")

        poly_verts = []
        colors = []
        for i, tri in enumerate(triangles):
            polygon = [verts[v] for v in tri]
            poly_verts.append(polygon)
            colors.append(cmap(norm(face_strains[i])))

        collection = Poly3DCollection(
            poly_verts, alpha=0.7,
            facecolors=colors,
            edgecolors="#333333",
            linewidths=0.3,
        )
        ax.add_collection3d(collection)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label="Strain")

        # Mark material limit on colorbar
        if paper.material.max_strain < max_strain:
            cbar.ax.axhline(
                y=paper.material.max_strain,
                color="red", linewidth=2, linestyle="--",
            )
    else:
        # Simple wireframe
        for tri in triangles:
            polygon = [verts[v] for v in tri]
            polygon.append(polygon[0])
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            zs = [p[2] for p in polygon]
            ax.plot(xs, ys, zs, color="#555", linewidth=0.5)

    # Draw crease edges with assignment colors
    edge_colors = {"M": "#e74c3c", "V": "#3498db", "B": "#2c3e50"}
    for e_idx in range(paper.num_edges):
        asgn = paper.edges_assignment[e_idx]
        if asgn in edge_colors:
            v1, v2 = paper.edges_vertices[e_idx]
            p1, p2 = verts[v1], verts[v2]
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=edge_colors[asgn],
                linewidth=1.5 if asgn == "B" else 1.0,
            )

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Equal aspect ratio
    _set_axes_equal(ax)

    ax.set_title(f"Folded State (folds: {paper.fold_count})", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()

    img = _fig_to_pil(fig, dpi)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    plt.close(fig)
    return img


def render_strain_heatmap(
    paper: PaperState,
    output_path: str | None = None,
    figsize: tuple = (6, 5),
    dpi: int = 150,
) -> Image.Image:
    """Top-down strain heatmap using triangulated faces."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    verts = paper.vertices_coords
    triangles = paper.triangulated_faces

    if not triangles:
        ax.text(0.5, 0.5, "No faces", ha="center", va="center")
    else:
        from matplotlib.tri import Triangulation

        tri_indices = np.array(triangles)
        if len(tri_indices) > 0 and tri_indices.shape[1] == 3:
            triang = Triangulation(verts[:, 0], verts[:, 1], tri_indices)
            tcf = ax.tripcolor(
                triang, paper.strain_per_vertex,
                cmap="coolwarm", shading="gouraud",
            )
            fig.colorbar(tcf, ax=ax, label="Strain")

            # Mark material limit
            ax.set_title(
                f"Strain Heatmap (max: {paper.strain_per_vertex.max():.4f}, "
                f"limit: {paper.material.max_strain})",
                fontsize=9,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()

    img = _fig_to_pil(fig, dpi)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    plt.close(fig)
    return img


def render_side_by_side(
    paper: PaperState,
    output_path: str | None = None,
    figsize: tuple = (14, 5),
    dpi: int = 150,
) -> Image.Image:
    """Combined: 2D crease pattern + 3D folded state + metrics text."""
    from .render_2d import render_crease_pattern

    fig = plt.figure(figsize=figsize)

    # Left: 2D crease pattern
    ax1 = fig.add_subplot(131)
    _draw_crease_pattern_on_ax(paper, ax1)

    # Center: 3D folded
    ax2 = fig.add_subplot(132, projection="3d")
    _draw_folded_on_ax(paper, ax2)

    # Right: metrics text
    ax3 = fig.add_subplot(133)
    ax3.axis("off")
    metrics_text = (
        f"Fold Count: {paper.fold_count}\n"
        f"Vertices: {paper.num_vertices}\n"
        f"Edges: {paper.num_edges}\n"
        f"Mountain: {paper.num_mountain}\n"
        f"Valley: {paper.num_valley}\n"
        f"Bounding Box: {paper.bounding_box}\n"
        f"Max Strain: {paper.strain_per_vertex.max():.6f}\n"
        f"Mean Strain: {paper.strain_per_vertex.mean():.6f}\n"
        f"Material: {paper.material.name}\n"
        f"Max Allowed: {paper.material.max_strain}\n"
        f"Total Energy: {paper.energy.get('total', 0):.4f}"
    )
    ax3.text(
        0.1, 0.5, metrics_text,
        fontsize=9, fontfamily="monospace",
        verticalalignment="center",
        transform=ax3.transAxes,
    )
    ax3.set_title("Metrics", fontsize=10)

    plt.tight_layout()

    img = _fig_to_pil(fig, dpi)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    plt.close(fig)
    return img


# ── Internal drawing helpers ──────────────────────────

def _draw_crease_pattern_on_ax(paper: PaperState, ax):
    """Draw crease pattern on an existing matplotlib axes."""
    from .render_2d import EDGE_STYLES

    verts = paper.rest_positions[:, :2]
    for e_idx in range(paper.num_edges):
        v1, v2 = paper.edges_vertices[e_idx]
        asgn = paper.edges_assignment[e_idx]
        style = EDGE_STYLES.get(asgn, EDGE_STYLES["U"])
        p1, p2 = verts[v1], verts[v2]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax.set_aspect("equal")
    ax.set_title("Crease Pattern", fontsize=10)


def _draw_folded_on_ax(paper: PaperState, ax):
    """Draw folded state on an existing 3D axes."""
    verts = paper.vertices_coords
    triangles = paper.triangulated_faces

    for tri in triangles:
        polygon = [verts[v].tolist() for v in tri]
        collection = Poly3DCollection([polygon], alpha=0.4, edgecolors="#555", linewidths=0.3)
        collection.set_facecolor("#88bbee")
        ax.add_collection3d(collection)

    _set_axes_equal(ax)
    ax.set_title(f"Folded (folds: {paper.fold_count})", fontsize=10)


def _set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    max_span = spans.max()
    centers = limits.mean(axis=1)
    for setter, center in zip(
        [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], centers
    ):
        setter([center - max_span / 2, center + max_span / 2])


def _fig_to_pil(fig, dpi: int = 150) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()
