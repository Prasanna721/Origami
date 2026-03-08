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
from origami_env.training.runner import TrainingRunner


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


# ── Demo: run sample strategies with broadcast ──────────────

def demo_strategies():
    """Six strategies inspired by classic origami patterns (Origami Simulator style).

    Each produces visually distinct folding in the 3D viewer:
    1. Half fold — single horizontal valley
    2. Quarter fold — two perpendicular valleys (book fold)
    3. Diagonal — mountain fold corner-to-corner
    4. Letter fold — tri-fold parallel valleys
    5. Miura-ori inspired — zigzag grid (alternating M/V)
    6. Waterbomb base — diagonal cross + perpendicular cross
    """

    def strategy_half(paper_state):
        """Simple horizontal valley fold — clean half."""
        if paper_state.get("fold_count", 0) >= 1:
            return None
        return {"type": "valley", "line": {"start": [0, 0.5], "end": [1, 0.5]}, "angle": 180}

    def strategy_quarter(paper_state):
        """Two perpendicular folds — quarter compression."""
        fc = paper_state.get("fold_count", 0)
        if fc == 0:
            return {"type": "valley", "line": {"start": [0, 0.5], "end": [1, 0.5]}, "angle": 180}
        elif fc == 1:
            return {"type": "valley", "line": {"start": [0.5, 0], "end": [0.5, 1]}, "angle": 180}
        return None

    def strategy_diagonal(paper_state):
        """Diagonal mountain fold — triangle shape."""
        if paper_state.get("fold_count", 0) >= 1:
            return None
        return {"type": "mountain", "line": {"start": [0, 0], "end": [1, 1]}, "angle": 180}

    def strategy_letter(paper_state):
        """Tri-fold — like folding a letter into an envelope."""
        fc = paper_state.get("fold_count", 0)
        if fc == 0:
            return {"type": "valley", "line": {"start": [0, 0.33], "end": [1, 0.33]}, "angle": 180}
        elif fc == 1:
            return {"type": "valley", "line": {"start": [0, 0.66], "end": [1, 0.66]}, "angle": 180}
        return None

    def strategy_miura(paper_state):
        """Miura-ori inspired — zigzag alternating M/V grid for max compression."""
        fc = paper_state.get("fold_count", 0)
        folds = [
            {"type": "valley", "line": {"start": [0, 0.33], "end": [1, 0.33]}, "angle": 180},
            {"type": "mountain", "line": {"start": [0, 0.66], "end": [1, 0.66]}, "angle": 180},
            {"type": "valley", "line": {"start": [0.33, 0], "end": [0.33, 1]}, "angle": 180},
            {"type": "mountain", "line": {"start": [0.66, 0], "end": [0.66, 1]}, "angle": 180},
        ]
        if fc < len(folds):
            return folds[fc]
        return None

    def strategy_waterbomb(paper_state):
        """Waterbomb base — diagonal cross then perpendicular cross."""
        fc = paper_state.get("fold_count", 0)
        folds = [
            {"type": "valley", "line": {"start": [0, 0], "end": [1, 1]}, "angle": 180},
            {"type": "valley", "line": {"start": [1, 0], "end": [0, 1]}, "angle": 180},
            {"type": "mountain", "line": {"start": [0, 0.5], "end": [1, 0.5]}, "angle": 180},
            {"type": "mountain", "line": {"start": [0.5, 0], "end": [0.5, 1]}, "angle": 180},
        ]
        if fc < len(folds):
            return folds[fc]
        return None

    return [strategy_half, strategy_quarter, strategy_diagonal,
            strategy_letter, strategy_miura, strategy_waterbomb]


# ── Main training loop ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Origami RL — GRPO Training")
    parser.add_argument("--demo", action="store_true", help="Run demo batch with sample strategies (tests grid viewer)")
    parser.add_argument("--demo-batches", type=int, default=3, help="Number of demo batches to run")
    parser.add_argument("--port", type=int, default=8000, help="Server port for broadcast")
    args = parser.parse_args()

    if args.demo:
        import time
        import threading
        import uvicorn

        print("=" * 60, flush=True)
        print("Origami RL — Demo Mode (Training Grid Viewer)", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)
        print(f"Starting server on http://127.0.0.1:{args.port}", flush=True)
        print(f"Open http://127.0.0.1:{args.port}/viewer/training.html in your browser", flush=True)
        print(flush=True)

        # Start the server in a background thread
        from origami_env.server.app import app, broadcast

        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": "127.0.0.1", "port": args.port, "log_level": "warning"},
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)

        # Create runner with broadcast
        runner = TrainingRunner(broadcast=broadcast, task_name="half_fold")
        broadcast.start_training()

        strategies = demo_strategies()

        for batch_idx in range(args.demo_batches):
            print(f"\n--- Batch {batch_idx + 1}/{args.demo_batches} ---", flush=True)
            scores = runner.run_batch(strategies, batch_id=batch_idx + 1)
            print(f"Scores: {scores}", flush=True)
            print(f"Avg: {sum(scores)/len(scores):.2f}, Best: {max(scores):.2f}", flush=True)
            time.sleep(2)  # Pause between batches so viewer can show results

        broadcast.end_training(total_batches=args.demo_batches, best_score=max(max(s) for s in [scores]))
        print("\nDemo complete. Server still running for viewer. Ctrl+C to exit.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
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
        print("3. Demo mode: python -m origami_env.training.train_grpo --demo")
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
        # from origami_env.server.app import broadcast
        # runner = TrainingRunner(broadcast=broadcast, task_name="half_fold")
        # broadcast.start_training()
        #
        # # In fold_quality reward function, use runner.run_episode() instead of
        # # direct _execute_strategy() to get broadcast support.
        #
        # dataset = Dataset.from_list([{"prompt": [...], ...}] * 1000)
        # trainer = GRPOTrainer(
        #     model=model, processing_class=tokenizer,
        #     reward_funcs=[code_valid, no_cheating, fold_quality],
        #     args=GRPOConfig(...), train_dataset=dataset,
        # )
        # trainer.train()
        # broadcast.end_training()

        print("Training script ready. See comments for full Colab integration.")
