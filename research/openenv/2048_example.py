"""
Reference Implementation: OpenEnv + GRPO Reinforcement Learning for 2048 Game
==============================================================================

Extracted from the Unsloth / OpenEnv notebook:
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/
  OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb

This file contains ALL code cells from the notebook, organized into sections.
It serves as a reference for how to build an OpenEnv-based RL environment
and connect it to GRPO training via TRL.

KEY ARCHITECTURE:
  1. OpenEnv provides a server-based game environment (2048 via OpenSpiel)
  2. The LLM generates a Python *strategy function* (code-as-action)
  3. The strategy function is executed against the environment
  4. Three reward functions score the output:
     - function_works: Does the generated code parse and compile?
     - no_cheating: Does it only use stdlib imports?
     - strategy_succeeds: Does the strategy actually play the game well?
  5. GRPO (from TRL) uses these rewards to train the model

PROMPT/RESPONSE FORMAT:
  - Prompt asks the LLM to write a Python function `strategy(board)` that
    takes a list-of-lists board state and returns "0","1","2","3" (up/right/down/left)
  - Response is wrapped in ```python ... ``` backticks
  - The function is extracted, sandboxed, and executed against the live game

REWARD STRUCTURE:
  - function_works:     +1.0 if valid Python, -2.0 if no function / syntax error, -0.5 if exec fails
  - no_cheating:        +1.0 if only stdlib imports, -20.0 if non-stdlib imports, -1.0 if no function
  - strategy_succeeds:  +20.0 if reaches 2048, +2.0 if function runs but doesn't win,
                        -1.0 on timeout, -3.0 on exception, 0 if function broken
"""


# =============================================================================
# CELL 1: Installation (pip installs - shown for reference, not executable here)
# =============================================================================
"""
%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
    except: get_numpy = "numpy"
    !uv pip install -qqq \\
        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers==4.56.2" trackio \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\
        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth trackio
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo
"""


# =============================================================================
# CELL 2: Install OpenEnv from source
# =============================================================================
"""
%%capture
!pip install -qqq fastapi uvicorn requests open_spiel
!pip install fastapi uvicorn requests
!pip install open_spiel --prefer-binary
!git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1
%cd OpenEnv
!git checkout 83dda10
"""

import subprocess, sys, os
from pathlib import Path
# sys.path.insert(0, '.')  # Add OpenEnv root for envs module
# sys.path.insert(0, './src')
# working_directory = str(Path.cwd().parent.absolute() / "OpenEnv")


# =============================================================================
# CELL 3: Load the model with Unsloth
# =============================================================================
import os
from unsloth import FastLanguageModel
import torch

max_seq_length = 768  # Can increase for longer RL output
lora_rank = 4         # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    load_in_4bit = True,
    max_seq_length = max_seq_length,
    offload_embedding = True,  # Offload embeddings to save more VRAM
)


# =============================================================================
# CELL 4: Apply LoRA adapters
# =============================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,  # *2 speeds up training
    use_gradient_checkpointing = "unsloth",  # Reduces memory usage
    random_state = 3407,
)


# =============================================================================
# CELL 5: OpenEnv imports (environment-specific)
# =============================================================================
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation


# =============================================================================
# CELL 6: OpenEnv process launch configuration
# =============================================================================
global port
global openenv_process
port = 9000
openenv_process = None
server = "envs.openspiel_env.server.app:app"
environment = {
    **os.environ,
    "PYTHONPATH": f"{working_directory}/src",
    "OPENSPIEL_GAME": "2048",
    "OPENSPIEL_AGENT_PLAYER": "0",
    "OPENSPIEL_OPPONENT_POLICY": "random",
}

# Augment Unsloth's OpenEnv creation function
import functools
from unsloth import is_port_open, launch_openenv
launch_openenv = functools.partial(
    launch_openenv,
    working_directory = working_directory,
    server = server,
    environment = environment,
    openenv_class = OpenSpielEnv,
)


# =============================================================================
# CELL 7: Reset the environment and observe initial state
# =============================================================================
port, openenv_process = launch_openenv(port, openenv_process)
result = openenv_process.reset()
current_state = result.observation
# current_state is an OpenSpielObservation with:
#   .done            -> bool
#   .reward          -> float or None
#   .info_state      -> list of floats (flat board)
#   .legal_actions   -> list of ints (e.g. [0,1,2,3])
#   .game_phase      -> str
#   .current_player_id -> int


# =============================================================================
# CELL 8: Convert flat info_state to 2D board
# =============================================================================
import numpy as np

def convert_to_board(current_state):
    n = len(current_state.info_state)
    size = int(np.sqrt(n))
    board = np.array_split(np.array(current_state.info_state, dtype=int), size)
    board = [x.tolist() for x in board]
    return board, size


# =============================================================================
# CELL 9: Pretty-print the 2048 board (collapsible in notebook)
# =============================================================================
def render_board(obs, colors: bool = True, border: bool = True, dot_for_zero: bool = True) -> str:
    """
    Pretty-print the board with colors that scale from 0 up to self.target.
    Uses ANSI 256-color codes (works in most terminals). Set colors=False to disable.
    """
    import math
    b, size = convert_to_board(obs)
    mx = max((max(row) for row in b), default=0)
    cell_w = max(3, len(str(mx)))

    RESET = "\x1b[0m"

    # A smooth-ish gradient from cool -> warm
    GRAD = [33, 39, 45, 51, 50, 49, 48, 47, 46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196]
    ZERO_FG = 239  # dim gray

    def color_code(v: int) -> str:
        if not colors:
            return ""
        if v == 0:
            return f"\x1b[38;5;{ZERO_FG}m"
        t = max(2, 2048)
        try:
            r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))
        except ValueError:
            r = 0.0
        idx = int(round(r * (len(GRAD) - 1)))
        return f"\x1b[38;5;{GRAD[idx]}m"

    def fmt(v: int) -> str:
        s = "." if (v == 0 and dot_for_zero) else str(v)
        s = s.rjust(cell_w)
        return color_code(v) + s + (RESET if colors else "")

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join("\u2500" * cell_w for _ in range(size)) + right

    rows = []
    if border:
        rows.append(hline("\u250c", "\u252c", "\u2510"))
    for r in range(size):
        content = "\u2502".join(fmt(v) for v in b[r])
        rows.append(("\u2502" + content + "\u2502") if border else content)
        if border:
            rows.append(hline("\u2514" if r == size - 1 else "\u251c",
                            "\u2534" if r == size - 1 else "\u253c",
                            "\u2518" if r == size - 1 else "\u2524"))
    return "\n".join(rows)


# =============================================================================
# CELL 10: Demonstrate stepping through the environment
# =============================================================================
# Action mapping: 0 = up, 1 = right, 2 = down, 3 = left
action = OpenSpielAction(action_id=0, game_name="2048")
result = openenv_process.step(action)
current_state = result.observation
print(render_board(current_state))


# =============================================================================
# CELL 11: RL Environment - Strategy execution with time limit
# =============================================================================
from typing import Callable
from unsloth import execute_with_time_limit
import itertools

def _execute_strategy(strategy, current_state: OpenSpielObservation):
    """
    Execute a strategy function against the 2048 environment.
    The strategy receives a board (list of lists) and returns an action int.
    Runs until the game is done or the strategy fails.
    Returns (steps, whether_2048_was_reached).
    """
    assert callable(strategy)

    steps = 0
    total_reward = 0
    while not current_state.done:
        board, size = convert_to_board(current_state)
        action = strategy(board)
        try:
            action = int(action)
        except:
            return steps, False
        steps += 1
        if type(action) is not int or action not in current_state.legal_actions:
            return steps, max(itertools.chain.from_iterable(board)) == 2048

        global port, openenv_process
        port, openenv_process = launch_openenv(port, openenv_process)
        action = OpenSpielAction(action_id=action, game_name="2048")
        result = openenv_process.step(action)
        current_state = result.observation
        if result.reward is not None:
            total_reward += result.reward
    return steps, max(itertools.chain.from_iterable(board)) == 2048


# Time-limited wrapper (2 seconds default, later changed to 5)
@execute_with_time_limit(2)
def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
    return _execute_strategy(strategy, current_state)


# =============================================================================
# CELL 12: Test with a trivial strategy
# =============================================================================
def always_move_left(board):
    return 3

# Reset OpenEnv to an initial state!
port, openenv_process = launch_openenv(port, openenv_process)
result = openenv_process.reset()
current_state = result.observation
try:
    steps, if_done = execute_strategy(always_move_left, current_state)
except TimeoutError as e:
    print(f"Timed out with error = {str(e)}")
print(f"steps={steps}, if_done={if_done}")


# =============================================================================
# CELL 13: Extend time limit to 5 seconds for actual RL training
# =============================================================================
@execute_with_time_limit(5)
def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
    return _execute_strategy(strategy, current_state)


# =============================================================================
# CELL 14: Code safety - check_python_modules (anti-reward-hacking)
# =============================================================================
from unsloth import check_python_modules

# Example: allowed (only stdlib)
sample_ok = """
def strategy(board):
    import math
    from typing import Callable
    return "0"
"""
ok, info = check_python_modules(sample_ok)
print("Only Python imports?", ok)  # True
print(info)

# Example: disallowed (numpy is non-stdlib)
sample_bad = """
def strategy(board):
    from numpy import matmul
    return "0"
"""
ok, info = check_python_modules(sample_bad)
print("Only Python imports?", ok)  # False
print(info)


# =============================================================================
# CELL 15: Sandboxed function execution (no global variable leakage)
# =============================================================================
from unsloth import create_locked_down_function

# This will fail - np is not defined inside the sandbox
function_bad = """
def import_numpy():
    np.matmul
    print("Success")
"""
f = create_locked_down_function(function_bad)
try:
    f()
except Exception as e:
    print(str(e))  # "name 'np' is not defined"

# This will work - no external references
function_good = """
def add(a, b):
    def adder(a):
        return a + b
    return adder(b) + b
"""
f = create_locked_down_function(function_good)
try:
    print(f(10, 20))  # 60
except Exception as e:
    print(str(e))


# =============================================================================
# CELL 16: THE PROMPT - How the LLM interacts with the environment
# =============================================================================
prompt = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "0", "1", "2", "3" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "0" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()
print(prompt)


# =============================================================================
# CELL 17: Test inference before RL training
# =============================================================================
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 512,
    streamer = TextStreamer(tokenizer, skip_prompt=False),
)


# =============================================================================
# CELL 18: REWARD FUNCTION 1 - extract_function (helper)
# =============================================================================
def extract_function(text):
    """
    Extract a Python function wrapped in triple backticks from the LLM response.
    Returns the function source string, or None if not found.
    """
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"):
            return fx
    return None


# =============================================================================
# CELL 19: REWARD FUNCTION 2 - function_works
# =============================================================================
def function_works(completions, **kwargs):
    """
    Reward: Does the generated code parse as valid Python and compile?
    +1.0  if valid function that can be created
    -0.5  if it has the right structure but exec fails
    -2.0  if no function extracted or syntax error
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_strategy = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores


# =============================================================================
# CELL 20: REWARD FUNCTION 3 - no_cheating
# =============================================================================
def no_cheating(completions, **kwargs):
    """
    Reward: Does the function only use Python stdlib imports?
    +1.0   if only stdlib imports
    -20.0  if non-stdlib imports (heavy penalty!)
    -1.0   if function extraction failed
    """
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)  # Penalize heavily!
        else:
            scores.append(-1.0)  # Failed creating function
    return scores


# =============================================================================
# CELL 21: REWARD FUNCTION 4 - strategy_succeeds
# =============================================================================
import numpy as np

global PRINTER
PRINTER = 0

def strategy_succeeds(completions, **kwargs):
    """
    Reward: Does the strategy actually play 2048 successfully?
    +20.0  if the strategy reaches 2048 (massive reward!)
    +2.0   if the function runs and plays but doesn't reach 2048
    -1.0   if timeout (strategy takes too long)
    -3.0   if exception during execution
     0     if function is broken/can't be created
    """
    global PRINTER
    scores = []
    for completion in completions:
        printed = False
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if PRINTER % 5 == 0:
            printed = True
            print(function)
        PRINTER += 1
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        try:
            # Reset OpenEnv to an initial state!
            global port, openenv_process
            port, openenv_process = launch_openenv(port, openenv_process)
            result = openenv_process.reset()
            current_state = result.observation
            steps, if_done = execute_strategy(new_strategy, current_state)
            print(f"Steps = {steps} If Done = {if_done}")
            if printed is False:
                print(function)
            print(render_board(current_state))
            if if_done:
                scores.append(20.0)  # Success - massively reward!
            else:
                scores.append(2.0)   # Failed but function works!
        except TimeoutError as e:
            print("Timeout")
            scores.append(-1.0)  # Failed with timeout
        except Exception as e:
            print(f"Exception = {str(e)}")
            scores.append(-3.0)  # Failed
    return scores


# =============================================================================
# CELL 22: Create the dataset (replicated prompt)
# =============================================================================
from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": prompt.strip()}],
        "answer": 0,
        "reasoning_effort": "low",
    }
] * 1000)

maximum_length = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    add_generation_prompt=True,
))
print(f"Prompt token length: {maximum_length}")


# =============================================================================
# CELL 23: GRPO Training Configuration
# =============================================================================
max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 2e-4,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,   # Increase to 4 for smoother training
    num_generations = 2,               # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1,            # Set to 1 for a full training run
    max_steps = 600,
    save_steps = 100,
    report_to = "trackio",             # Can use Weights & Biases, TrackIO
    output_dir = "outputs",
)


# =============================================================================
# CELL 24: Create GRPO Trainer and Train
# =============================================================================
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,       # Reward 1: Is it valid Python?
        no_cheating,          # Reward 2: Only stdlib imports?
        strategy_succeeds,    # Reward 3: Does it actually play 2048?
    ],
    args = training_args,
    train_dataset = dataset,
)

# Start training! (~5 hours for 600 steps on T4)
# Expect 0 reward for ~first 100 steps, then gradual improvement
trainer.train()


# =============================================================================
# CELL 25: Inference after RL training
# =============================================================================
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt=False),
)


# =============================================================================
# CELL 26: Save the model (optional)
# =============================================================================
# Merge and push to hub in mxfp4 4bit format
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="mxfp4")
if False:
    model.push_to_hub_merged("repo_id/repo_name", tokenizer, token="hf...", save_method="mxfp4")

# Merge and push to hub in 16bit
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="merged_16bit")
if False:
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method="merged_16bit", token="")
