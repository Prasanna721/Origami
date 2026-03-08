"""2D crease pattern rendering via matplotlib."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from PIL import Image

from ..engine.paper import PaperState

# Standard origami edge styles
EDGE_STYLES = {
    "M": {"color": "#e74c3c", "linestyle": (0, (8, 3, 2, 3)), "linewidth": 2.0, "label": "Mountain"},
    "V": {"color": "#3498db", "linestyle": (0, (6, 3)),        "linewidth": 2.0, "label": "Valley"},
    "B": {"color": "#2c3e50", "linestyle": "-",                "linewidth": 2.5, "label": "Boundary"},
    "F": {"color": "#bdc3c7", "linestyle": "-",                "linewidth": 0.5, "label": "Flat"},
    "U": {"color": "#95a5a6", "linestyle": ":",                "linewidth": 1.0, "label": "Unassigned"},
}


def render_crease_pattern(
    paper: PaperState,
    output_path: str | None = None,
    figsize: tuple = (6, 6),
    dpi: int = 150,
    show_vertices: bool = True,
    show_legend: bool = True,
) -> Image.Image:
    """
    Render 2D crease pattern with standard origami colors.
    M = red dashed, V = blue dash-dot, B = black solid, F = gray, U = dotted.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use rest positions (flat) for 2D crease pattern
    verts = paper.rest_positions[:, :2]

    # Draw edges
    drawn_styles = set()
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
            solid_capstyle="round",
        )
        drawn_styles.add(asgn)

    # Draw vertices
    if show_vertices:
        ax.scatter(
            verts[:, 0], verts[:, 1],
            c="#555555", s=8, zorder=5,
        )

    # Legend
    if show_legend:
        handles = []
        for asgn in sorted(drawn_styles):
            if asgn in EDGE_STYLES:
                style = EDGE_STYLES[asgn]
                handles.append(mlines.Line2D(
                    [], [],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=style["label"],
                ))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title(f"Crease Pattern ({paper.num_mountain}M / {paper.num_valley}V)", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Add grid
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save or return
    img = _fig_to_pil(fig, dpi)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    plt.close(fig)
    return img


def render_crease_pattern_svg(paper: PaperState) -> str:
    """Return inline SVG string for the crease pattern."""
    verts = paper.rest_positions[:, :2]

    # Compute viewBox
    x_min, y_min = verts.min(axis=0)
    x_max, y_max = verts.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.05
    vb = f"{x_min - margin} {y_min - margin} {x_max - x_min + 2 * margin} {y_max - y_min + 2 * margin}"

    svg_colors = {"M": "red", "V": "blue", "B": "black", "F": "#ccc", "U": "gray"}
    svg_dashes = {"M": "8,3,2,3", "V": "6,3", "B": "none", "F": "none", "U": "2,4"}

    lines = []
    for e_idx in range(paper.num_edges):
        v1, v2 = paper.edges_vertices[e_idx]
        asgn = paper.edges_assignment[e_idx]
        p1, p2 = verts[v1], verts[v2]
        color = svg_colors.get(asgn, "gray")
        dash = svg_dashes.get(asgn, "none")
        stroke_dash = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        lw = "2.5" if asgn == "B" else "2" if asgn in ("M", "V") else "0.5"
        lines.append(
            f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" '
            f'stroke="{color}" stroke-width="{lw}"{stroke_dash} stroke-linecap="round"/>'
        )

    return (
        f'<svg viewBox="{vb}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:100%">\n'
        + "\n".join(lines)
        + "\n</svg>"
    )


def _fig_to_pil(fig, dpi: int = 150) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).copy()
