"""Task pool, curriculum levels, difficulty sampling."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# ── Task Definitions ─────────────────────────────────────────
# Level 1: Learn the format
# Level 2: Parallel/grid folds
# Level 3: Real-world applications with material constraints
# Level 4: Complex geometry targets

TASK_POOL: List[Dict[str, Any]] = [
    # ── Level 1 ──────────────────────────────────────────────
    {
        "name": "half_fold",
        "description": "Fold a paper sheet in half. Simplest possible task.",
        "difficulty": 1,
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.50,
        "max_folds": 3,
        "target_box": [1.0, 0.5, 0.01],
        "must_deploy": False,
    },
    {
        "name": "quarter_fold",
        "description": "Fold paper into quarters using two perpendicular folds.",
        "difficulty": 1,
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.25,
        "max_folds": 5,
        "target_box": [0.5, 0.5, 0.02],
        "must_deploy": False,
    },

    # ── Level 2 ──────────────────────────────────────────────
    {
        "name": "letter_fold",
        "description": "Tri-fold a letter-sized sheet for envelope fitting.",
        "difficulty": 2,
        "width": 0.216,  # ~8.5 inches
        "height": 0.279,  # ~11 inches
        "material": "paper",
        "target_ratio": 0.33,
        "max_folds": 5,
        "target_box": [0.216, 0.093, 0.005],
        "must_deploy": False,
    },
    {
        "name": "map_fold",
        "description": "Fold a map sheet into a compact grid. Must be deployable (unfoldable).",
        "difficulty": 2,
        "width": 1.0,
        "height": 1.0,
        "material": "paper",
        "target_ratio": 0.125,
        "max_folds": 8,
        "target_box": [0.25, 0.25, 0.02],
        "must_deploy": True,
    },

    # ── Level 3 ──────────────────────────────────────────────
    {
        "name": "solar_panel",
        "description": (
            "Pack a 1m x 1m Mylar solar panel into a compact package for satellite deployment. "
            "Must be deployable. Aim for Miura-ori or similar efficient pattern."
        ),
        "difficulty": 3,
        "width": 1.0,
        "height": 1.0,
        "material": "mylar",
        "target_ratio": 0.05,
        "max_folds": 20,
        "target_box": [0.15, 0.15, 0.05],
        "must_deploy": True,
    },
    {
        "name": "shelter_wall",
        "description": (
            "Fold a 2m x 1m aluminum sheet into a compact panel for deployable shelter. "
            "Rigid material — minimize strain at creases."
        ),
        "difficulty": 3,
        "width": 2.0,
        "height": 1.0,
        "material": "aluminum",
        "target_ratio": 0.10,
        "max_folds": 15,
        "target_box": [0.5, 0.25, 0.1],
        "must_deploy": True,
    },

    # ── Level 4 ──────────────────────────────────────────────
    {
        "name": "stent",
        "description": (
            "Fold a 0.1m x 0.05m nitinol sheet into a compact stent-like structure. "
            "Superelastic material — high strain tolerance. "
            "Must fit in a small delivery catheter."
        ),
        "difficulty": 4,
        "width": 0.1,
        "height": 0.05,
        "material": "nitinol",
        "target_ratio": 0.09,
        "max_folds": 25,
        "target_box": [0.02, 0.02, 0.01],
        "must_deploy": True,
    },
]


def get_task_pool() -> List[Dict[str, Any]]:
    """Return the full task pool."""
    return TASK_POOL


def get_tasks_by_difficulty(level: int) -> List[Dict[str, Any]]:
    """Get tasks at a specific difficulty level."""
    return [t for t in TASK_POOL if t["difficulty"] == level]


def get_task_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a task by name."""
    for t in TASK_POOL:
        if t["name"] == name:
            return t.copy()
    return None


def sample_task(
    seed: Optional[int] = None,
    difficulty: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sample a random task from the pool.

    Args:
        seed: Random seed for reproducibility.
        difficulty: If given, only sample from that difficulty level.
    """
    rng = random.Random(seed)

    if difficulty is not None:
        pool = get_tasks_by_difficulty(difficulty)
    else:
        pool = TASK_POOL

    if not pool:
        pool = TASK_POOL

    return rng.choice(pool).copy()
