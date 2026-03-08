"""
GRPO Training Script — Origami RL
==================================

Follows the 2048 Unsloth/OpenEnv pattern:
1. Launch OpenEnv server as local subprocess
2. Load LLM with LoRA
3. Define reward functions (code_valid, no_cheating, fold_quality)
4. Train with GRPOTrainer

Run on Colab with GPU.
"""

import os
import sys
import subprocess
import functools
import itertools
from typing import Callable

import numpy as np

# ── OpenEnv imports ─────────────────────────────────────────

from origami_env.server.models import OrigamiAction, OrigamiObservation
from origami_env.client.client import OrigamiEnvClient


# ── Server launch config ───────────────────────────────────

global port
global openenv_process
port = 9000
openenv_process = None
server = "origami_env.server.app:app"


# ── Prompt ──────────────────────────────────────────────────

PROMPT = """
Write a Python function that folds origami to maximize compactness.
You are given a paper_state dict with geometry and physics data.
Return a fold dict or None to stop folding:

```python
def fold_strategy(paper_state):
    # paper_state keys: vertices_coords, edges_vertices, edges_assignment,
    #   bounding_box, num_layers, width, height, material, strain_per_vertex, fold_count
    #
    # Return a fold dict:
    return {
        "type": "valley",  # or "mountain"
        "line": {"start": [x1, y1], "end": [x2, y2]},  # 0-1 normalized
        "angle": 180,  # degrees
    }
    # Return None when done folding
```
All helper functions should be inside def fold_strategy. Only output the short function `fold_strategy`.
""".strip()


# ── Helper: extract function from LLM response ─────────────

def extract_function(text):
    """Extract Python function from triple-backtick response."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def fold_strategy(paper_state"):
            return fx
    return None


# ── Strategy executor ───────────────────────────────────────

def _execute_strategy(strategy_fn, openenv_client):
    """
    Execute a fold_strategy against the origami environment.
    strategy_fn takes paper_state dict, returns fold dict or None.
    Returns final observation.
    """
    result = openenv_client.reset(task_name="half_fold")
    obs = result.observation

    max_steps = 20
    steps = 0
    while not obs.done and steps < max_steps:
        paper_state = obs.paper_state
        fold = strategy_fn(paper_state)

        if fold is None:
            action = OrigamiAction(
                fold_type="stop",
                fold_line={"start": [0, 0], "end": [0, 0]},
                fold_angle=0,
            )
        else:
            action = OrigamiAction(
                fold_type=fold.get("type", "valley"),
                fold_line=fold.get("line", {"start": [0, 0.5], "end": [1, 0.5]}),
                fold_angle=fold.get("angle", 180),
                layer_select=fold.get("layer_select", "all"),
            )

        result = openenv_client.step(action)
        obs = result.observation
        steps += 1

    return obs


# ── Reward Functions ────────────────────────────────────────

def code_valid(completions, **kwargs):
    """Does the generated code parse as valid Python? +1 / -0.5 / -2"""
    from unsloth import check_python_modules, create_locked_down_function

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(-2.0)
        else:
            try:
                create_locked_down_function(function)
                scores.append(1.0)
            except Exception:
                scores.append(-0.5)
    return scores


def no_cheating(completions, **kwargs):
    """Only stdlib imports? +1 / -20 / -1"""
    from unsloth import check_python_modules

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)
        else:
            scores.append(-1.0)
    return scores


def fold_quality(completions, **kwargs):
    """Execute strategy, score from metrics. +20 / +5 / +2 / -1 / -3 / 0"""
    from unsloth import check_python_modules, create_locked_down_function, execute_with_time_limit

    global port, openenv_process

    @execute_with_time_limit(5)
    def run_strategy(strategy, client):
        return _execute_strategy(strategy, client)

    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)

        if function is None:
            scores.append(0)
            continue

        ok, info = check_python_modules(function)
        if "error" in info:
            scores.append(0)
            continue

        try:
            strategy = create_locked_down_function(function)
        except Exception:
            scores.append(0)
            continue

        try:
            # Connect to running OpenEnv server
            port, openenv_process = launch_openenv(port, openenv_process)
            obs = run_strategy(strategy, openenv_process)

            m = obs.metrics if hasattr(obs, 'metrics') else {}
            compactness = m.get("compactness", 0)
            is_valid = m.get("is_valid", False)

            print(f"  Compactness={compactness:.3f} Valid={is_valid} Folds={m.get('fold_count', 0)}")

            if compactness > 0.8 and is_valid:
                scores.append(20.0)
            elif compactness > 0.5:
                scores.append(5.0)
            else:
                scores.append(2.0)
        except TimeoutError:
            print("  Timeout")
            scores.append(-1.0)
        except Exception as e:
            print(f"  Exception: {e}")
            scores.append(-3.0)

    return scores


# ── Main training loop ──────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Origami RL — GRPO Training")
    print("=" * 60)
    print()
    print("This script follows the 2048 Unsloth/OpenEnv pattern.")
    print("Run on Colab with GPU. Requires: unsloth, trl, openenv-core")
    print()
    print("Steps:")
    print("1. Install deps: pip install unsloth trl openenv-core")
    print("2. Launch: python -m origami_env.training.train_grpo")
    print()

    # The actual training code would go here, following the 2048 pattern:
    #
    # from unsloth import FastLanguageModel, is_port_open, launch_openenv
    # model, tokenizer = FastLanguageModel.from_pretrained(...)
    # model = FastLanguageModel.get_peft_model(model, ...)
    #
    # launch_openenv = functools.partial(launch_openenv, server=server, ...)
    # port, openenv_process = launch_openenv(port, openenv_process)
    #
    # dataset = Dataset.from_list([{"prompt": [...], ...}] * 1000)
    # trainer = GRPOTrainer(
    #     model=model, processing_class=tokenizer,
    #     reward_funcs=[code_valid, no_cheating, fold_quality],
    #     args=GRPOConfig(...), train_dataset=dataset,
    # )
    # trainer.train()

    print("Training script ready. See comments for full Colab integration.")
