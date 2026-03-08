"""Per-step PNG capture and episode summary grid."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..engine.paper import PaperState
from .render_2d import render_crease_pattern
from .render_3d import render_folded_state, render_strain_heatmap


def capture_step(
    paper: PaperState,
    step_num: int,
    episode_dir: str,
    dpi: int = 120,
) -> dict:
    """
    Save renders for one step. Returns dict of relative file paths.

    Creates:
      - {episode_dir}/crease_step_{N}.png
      - {episode_dir}/folded_step_{N}.png
      - {episode_dir}/strain_step_{N}.png
    """
    Path(episode_dir).mkdir(parents=True, exist_ok=True)
    paths = {}

    crease_path = f"{episode_dir}/crease_step_{step_num}.png"
    render_crease_pattern(paper, output_path=crease_path, dpi=dpi)
    paths["crease_2d"] = crease_path

    folded_path = f"{episode_dir}/folded_step_{step_num}.png"
    render_folded_state(paper, output_path=folded_path, dpi=dpi)
    paths["folded_3d"] = folded_path

    strain_path = f"{episode_dir}/strain_step_{step_num}.png"
    render_strain_heatmap(paper, output_path=strain_path, dpi=dpi)
    paths["strain_heatmap"] = strain_path

    return paths


def capture_episode_summary(
    paper: PaperState,
    fold_history: list,
    task: dict,
    metrics: dict,
    episode_dir: str,
    dpi: int = 100,
) -> str:
    """
    Grid summary of entire episode. Returns path to summary image.

    Layout:
    Row 1: crease pattern snapshots per step
    Row 2: 3D folded state snapshots per step
    Row 3: final metrics text
    """
    Path(episode_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{episode_dir}/summary.png"

    n_steps = len(fold_history) + 1  # include initial state
    n_cols = min(n_steps, 6)  # cap at 6 columns

    fig_width = 3 * n_cols
    fig_height = 9  # 3 rows * 3 height each
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Row 1: Load crease pattern images
    for i in range(n_cols):
        step_idx = i if n_steps <= 6 else int(i * (n_steps - 1) / (n_cols - 1))
        ax = fig.add_subplot(3, n_cols, i + 1)
        img_path = f"{episode_dir}/crease_step_{step_idx}.png"
        if Path(img_path).exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.set_title(f"Step {step_idx}", fontsize=8)
        ax.axis("off")

    # Row 2: Load folded state images
    for i in range(n_cols):
        step_idx = i if n_steps <= 6 else int(i * (n_steps - 1) / (n_cols - 1))
        ax = fig.add_subplot(3, n_cols, n_cols + i + 1)
        img_path = f"{episode_dir}/folded_step_{step_idx}.png"
        if Path(img_path).exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.axis("off")

    # Row 3: Metrics text panel (span all columns)
    ax_text = fig.add_subplot(3, 1, 3)
    ax_text.axis("off")

    task_name = task.get("name", "unknown")
    material_name = task.get("material", {})
    if isinstance(material_name, dict):
        material_name = material_name.get("name", "unknown")

    metrics_text = (
        f"Task: {task_name}  |  Material: {material_name}  |  "
        f"Folds: {metrics.get('fold_count', 0)}  |  "
        f"Valid: {metrics.get('is_valid', False)}\n"
        f"Compactness: {metrics.get('compactness', 0):.3f}  |  "
        f"Deploy Ratio: {metrics.get('deployment_ratio', 0):.3f}  |  "
        f"Fits Box: {metrics.get('fits_target_box', None)}\n"
        f"Max Strain: {metrics.get('max_strain', 0):.6f}  |  "
        f"Mean Strain: {metrics.get('mean_strain', 0):.6f}  |  "
        f"Total Energy: {metrics.get('total_energy', 0):.4f}\n"
        f"Kawasaki Violations: {metrics.get('kawasaki_violations', 0)}  |  "
        f"Maekawa Violations: {metrics.get('maekawa_violations', 0)}  |  "
        f"Self-Intersections: {metrics.get('self_intersections', 0)}"
    )
    ax_text.text(
        0.05, 0.5, metrics_text,
        fontsize=10, fontfamily="monospace",
        verticalalignment="center",
        transform=ax_text.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )

    plt.suptitle(f"Episode Summary — {task_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
