"""GIF assembly from fold animation frames."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engine.paper import PaperState, create_flat_sheet
from ..engine.fold import apply_fold
from ..engine.physics import simulate
from ..engine.materials import MATERIALS, Material
from .render_3d import render_folded_state


def record_fold_animation(
    paper_initial: PaperState,
    fold_history: list,
    output_path: str,
    fps: int = 10,
    frames_per_fold: int = 8,
) -> str:
    """
    Generate animated GIF of the folding sequence.

    For each fold in history:
      - Interpolate fold_percent from 0.0 to 1.0
      - Run physics at each fold_percent
      - Render 3D frame via matplotlib

    Returns path to GIF.
    """
    import imageio.v2 as imageio

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []

    # Initial frame
    img = render_folded_state(paper_initial, figsize=(6, 4), dpi=80)
    frames.append(np.array(img))

    # Apply each fold progressively
    paper = paper_initial.copy()
    for fold_info in fold_history:
        fold_dict = {
            "type": fold_info.get("type", "valley"),
            "line": fold_info.get("line", {"start": [0, 0.5], "end": [1, 0.5]}),
            "angle": fold_info.get("angle", 180),
            "layer_select": fold_info.get("layer_select", "all"),
        }

        # Apply the fold geometry
        paper = apply_fold(paper, fold_dict)

        # Interpolate fold_percent for animation
        for f in range(frames_per_fold):
            percent = (f + 1) / frames_per_fold
            try:
                animated = simulate(
                    paper.copy(),
                    fold_percent=percent,
                    n_steps=100,
                )
                img = render_folded_state(animated, figsize=(6, 4), dpi=80)
            except Exception:
                img = render_folded_state(paper, figsize=(6, 4), dpi=80)
            frames.append(np.array(img))

    # Hold final frame
    for _ in range(fps):
        frames.append(frames[-1])

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    return output_path


def record_strain_evolution(
    paper_initial: PaperState,
    fold_history: list,
    output_path: str,
    fps: int = 5,
) -> str:
    """
    GIF showing how strain heatmap develops through the fold sequence.
    """
    import imageio.v2 as imageio
    from .render_3d import render_strain_heatmap

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frames = []

    paper = paper_initial.copy()

    # Initial frame
    img = render_strain_heatmap(paper, figsize=(5, 4), dpi=80)
    frames.append(np.array(img))

    for fold_info in fold_history:
        fold_dict = {
            "type": fold_info.get("type", "valley"),
            "line": fold_info.get("line", {"start": [0, 0.5], "end": [1, 0.5]}),
            "angle": fold_info.get("angle", 180),
            "layer_select": fold_info.get("layer_select", "all"),
        }

        paper = apply_fold(paper, fold_dict)
        try:
            paper = simulate(paper, fold_percent=1.0, n_steps=200)
        except Exception:
            pass

        img = render_strain_heatmap(paper, figsize=(5, 4), dpi=80)
        frames.append(np.array(img))

    # Hold final frame
    for _ in range(fps * 2):
        frames.append(frames[-1])

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    return output_path
