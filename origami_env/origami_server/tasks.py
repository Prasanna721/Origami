"""Task definitions for origami RL training.

Each task defines a target shape as a reference FOLD crease pattern.
The LLM must discover a crease pattern that folds into the same shape.

Starting simple (triangle) and progressing to harder folds.
"""

TASKS: dict[str, dict] = {
    "triangle": {
        "name": "triangle",
        "description": "Fold the paper in half diagonally to make a triangle",
        "difficulty": 1,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            "edges_assignment": ["B", "B", "B", "B", "V"],
            "edges_foldAngle": [0, 0, 0, 0, 180],
            "faces_vertices": [[0, 1, 2], [0, 2, 3]],
        },
    },
    "half_fold": {
        "name": "half_fold",
        "description": "Fold the paper in half horizontally",
        "difficulty": 1,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [1, 0], [1, 1], [0, 1], [0, 0.5], [1, 0.5],
            ],
            "edges_vertices": [
                [0, 1], [1, 5], [5, 2], [2, 3], [3, 4], [4, 0],
                [4, 5],
            ],
            "edges_assignment": ["B", "B", "B", "B", "B", "B", "V"],
            "edges_foldAngle": [0, 0, 0, 0, 0, 0, 180],
            "faces_vertices": [[0, 1, 5, 4], [4, 5, 2, 3]],
        },
    },
    "quarter_fold": {
        "name": "quarter_fold",
        "description": "Fold the paper into quarters (two perpendicular folds)",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [0.5, 0], [1, 0],
                [0, 0.5], [0.5, 0.5], [1, 0.5],
                [0, 1], [0.5, 1], [1, 1],
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0],
                # Fold lines
                [1, 4], [4, 7],  # vertical fold
                [3, 4], [4, 5],  # horizontal fold
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "V", "V", "V",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                180, 180, 180, 180,
            ],
            "faces_vertices": [
                [0, 1, 4, 3],  # bottom-left
                [1, 2, 5, 4],  # bottom-right
                [3, 4, 7, 6],  # top-left
                [4, 5, 8, 7],  # top-right
            ],
        },
    },
    "letter_fold": {
        "name": "letter_fold",
        "description": "Tri-fold the paper like a letter (two parallel folds)",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [1, 0],
                [0, 1/3], [1, 1/3],
                [0, 2/3], [1, 2/3],
                [0, 1], [1, 1],
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 3], [3, 5], [5, 7], [7, 6], [6, 4], [4, 2], [2, 0],
                # Fold lines
                [2, 3],  # first fold (valley)
                [4, 5],  # second fold (mountain)
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "M",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                180, -180,
            ],
            "faces_vertices": [
                [0, 1, 3, 2],  # bottom strip
                [2, 3, 5, 4],  # middle strip
                [4, 5, 7, 6],  # top strip
            ],
        },
    },
    # ── New tasks: real-world origami applications ──────────────────────
    "solar_panel": {
        "name": "solar_panel",
        "description": "Deployable solar panel array — three alternating mountain-valley folds compact the panel for launch, then unfold to full area in orbit",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            # 4 horizontal strips, alternating M/V/M folds → accordion/bellows
            "vertices_coords": [
                [0, 0], [1, 0],           # row 0 (bottom)
                [0, 0.25], [1, 0.25],     # row 1
                [0, 0.5], [1, 0.5],       # row 2 (center)
                [0, 0.75], [1, 0.75],     # row 3
                [0, 1], [1, 1],           # row 4 (top)
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 3], [3, 5], [5, 7], [7, 9], [9, 8],
                [8, 6], [6, 4], [4, 2], [2, 0],
                # Fold creases (3 horizontal folds)
                [2, 3],   # fold 1 — mountain
                [4, 5],   # fold 2 — valley
                [6, 7],   # fold 3 — mountain
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
                "M", "V", "M",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                -180, 180, -180,
            ],
            "faces_vertices": [
                [0, 1, 3, 2],  # strip 1
                [2, 3, 5, 4],  # strip 2
                [4, 5, 7, 6],  # strip 3
                [6, 7, 9, 8],  # strip 4
            ],
        },
    },
    "shelter": {
        "name": "shelter",
        "description": "Emergency A-frame shelter — four folds create a tent shape with a ridge and two floor flaps",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            # 5 horizontal strips: floor flap / wall / ridge / wall / floor flap
            # Folds: V at 0.2, M at 0.4, V at 0.6 (ridge), M at 0.8
            # This creates an A-frame tent cross-section
            "vertices_coords": [
                [0, 0], [1, 0],           # row 0 (bottom)
                [0, 0.2], [1, 0.2],       # row 1
                [0, 0.4], [1, 0.4],       # row 2
                [0, 0.6], [1, 0.6],       # row 3 (ridge)
                [0, 0.8], [1, 0.8],       # row 4
                [0, 1], [1, 1],           # row 5 (top)
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 3], [3, 5], [5, 7], [7, 9], [9, 11],
                [11, 10], [10, 8], [8, 6], [6, 4], [4, 2], [2, 0],
                # Fold creases (4 folds)
                [2, 3],   # fold 1 — valley (floor flap hinge)
                [4, 5],   # fold 2 — mountain (wall rises)
                [6, 7],   # fold 3 — valley (ridge peak — inverted V)
                [8, 9],   # fold 4 — mountain (wall rises)
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "M", "V", "M",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                90, -90, 90, -90,
            ],
            "faces_vertices": [
                [0, 1, 3, 2],    # floor flap (front)
                [2, 3, 5, 4],    # wall (left)
                [4, 5, 7, 6],    # roof panel (left)
                [6, 7, 9, 8],    # roof panel (right)
                [8, 9, 11, 10],  # wall (right) / floor flap
            ],
        },
    },
    "starshade": {
        "name": "starshade",
        "description": "NASA starshade flasher fold — four petal-pairs wrap in a pinwheel from corner valley folds",
        "difficulty": 3,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            # Flasher fold: 8 triangular sectors from center.
            # Folds ONLY at corner diagonals → pairs of sectors form flat petals.
            # Each petal wraps 90° around the previous one (pinwheel stow).
            # This IS the origami pattern NASA uses for the starshade inner disk.
            "vertices_coords": [
                [0, 0],       # 0: bottom-left corner
                [0.5, 0],     # 1: bottom midpoint
                [1, 0],       # 2: bottom-right corner
                [1, 0.5],     # 3: right midpoint
                [1, 1],       # 4: top-right corner
                [0.5, 1],     # 5: top midpoint
                [0, 1],       # 6: top-left corner
                [0, 0.5],     # 7: left midpoint
                [0.5, 0.5],   # 8: center
            ],
            "edges_vertices": [
                # Boundary (8 perimeter segments)
                [0, 1], [1, 2], [2, 3], [3, 4],
                [4, 5], [5, 6], [6, 7], [7, 0],
                # Corner diagonals = VALLEY FOLDS (petal hinges)
                [8, 0], [8, 2], [8, 4], [8, 6],
                # Edge midpoint radials = NO FOLD (petals stay flat)
                [8, 1], [8, 3], [8, 5], [8, 7],
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "V", "V", "V",
                "B", "B", "B", "B",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                90, 90, 90, 90,
                0, 0, 0, 0,
            ],
            "faces_vertices": [
                [8, 0, 1],  # petal 1a (bottom-left)
                [8, 1, 2],  # petal 1b (bottom-right)
                [8, 2, 3],  # petal 2a (right-bottom)
                [8, 3, 4],  # petal 2b (right-top)
                [8, 4, 5],  # petal 3a (top-right)
                [8, 5, 6],  # petal 3b (top-left)
                [8, 6, 7],  # petal 4a (left-top)
                [8, 7, 0],  # petal 4b (left-bottom)
            ],
        },
    },
}


def get_task(name: str | None = None) -> dict:
    """Get a task by name. Defaults to 'triangle'."""
    if name is None:
        name = "triangle"
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]


def list_tasks() -> list[str]:
    """List all available task names."""
    return list(TASKS.keys())
