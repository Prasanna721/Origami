"""Reward functions for GRPO training.

These run on the Colab client side, NOT on the server.
Three functions: code_valid, no_cheating, fold_quality.

Same structure as the 2048 pattern:
  extract strategy function → sandbox execute → score from metrics.
"""
from __future__ import annotations

import re
import traceback
from typing import Any, Callable, Dict, List

from .client import OrigamiEnvClient


# ── Strategy execution ──────────────────────────────────────────

def execute_strategy(
    strategy_fn: Callable,
    client: OrigamiEnvClient,
    task_name: str | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Execute a fold_strategy against the environment.

    strategy_fn takes paper_state dict and returns one fold dict or None (stop).
    Loops until done or strategy returns None.
    """
    obs = client.reset(task_name=task_name, seed=seed)

    while not obs.get("done", False):
        paper_state = obs.get("paper_state", {})

        try:
            fold = strategy_fn(paper_state)
        except Exception:
            fold = None

        if fold is None:
            obs = client.stop()
        else:
            action = {
                "fold_type": fold.get("type", "valley"),
                "fold_line": fold.get("line", {"start": [0, 0.5], "end": [1, 0.5]}),
                "fold_angle": fold.get("angle", 180),
                "layer_select": fold.get("layer_select", "all"),
            }
            obs = client.step(action)

    return obs


def extract_strategy_function(code: str) -> Callable | None:
    """
    Extract fold_strategy function from LLM-generated code.

    Expected signature:
        def fold_strategy(paper_state: dict) -> dict | None:
            ...
    """
    try:
        namespace = {}
        exec(code, namespace)
        fn = namespace.get("fold_strategy")
        if callable(fn):
            return fn
        return None
    except Exception:
        return None


# ── Reward functions ────────────────────────────────────────────

def reward_code_valid(completions: List[str], **kwargs) -> List[float]:
    """
    Reward 1: Does the code parse and define fold_strategy?

    Returns:
        1.0 if code defines a valid fold_strategy function
        0.1 if code parses but no fold_strategy
        0.0 if syntax error
    """
    rewards = []
    for code in completions:
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            rewards.append(0.0)
            continue

        fn = extract_strategy_function(code)
        if fn is not None:
            rewards.append(1.0)
        else:
            rewards.append(0.1)

    return rewards


def reward_no_cheating(completions: List[str], **kwargs) -> List[float]:
    """
    Reward 2: Penalize cheating patterns.

    Checks for:
        - Direct manipulation of internal state
        - Importing disallowed modules (os, sys, subprocess)
        - Hardcoded fold sequences without reading paper_state
    """
    BANNED_PATTERNS = [
        r"import\s+os\b",
        r"import\s+sys\b",
        r"import\s+subprocess\b",
        r"__import__",
        r"eval\s*\(",
        r"exec\s*\(",
        r"open\s*\(",
        r"\.env\b",
    ]

    rewards = []
    for code in completions:
        score = 1.0

        for pattern in BANNED_PATTERNS:
            if re.search(pattern, code):
                score -= 0.3

        # Check if function reads paper_state
        if "paper_state" not in code:
            score -= 0.2

        rewards.append(max(0.0, score))

    return rewards


def reward_fold_quality(
    completions: List[str],
    client: OrigamiEnvClient | None = None,
    task_name: str | None = None,
    seed: int | None = None,
    **kwargs,
) -> List[float]:
    """
    Reward 3: Execute strategy and score from metrics.

    This is the main quality signal. Runs the strategy in the environment
    and extracts the server-computed reward.

    Requires a running environment server (client must be provided).
    """
    if client is None:
        return [0.0] * len(completions)

    rewards = []
    for code in completions:
        fn = extract_strategy_function(code)
        if fn is None:
            rewards.append(0.0)
            continue

        try:
            final_obs = execute_strategy(fn, client, task_name=task_name, seed=seed)
            server_reward = final_obs.get("reward", 0.0) or 0.0
            # Normalize to 0-1 range (server reward is roughly -10 to +35)
            normalized = max(0.0, min(1.0, (server_reward + 10) / 45))
            rewards.append(normalized)
        except Exception:
            rewards.append(0.0)

    return rewards
