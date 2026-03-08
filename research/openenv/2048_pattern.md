# 2048 Example — Code-as-Policy Pattern

**Full code**: [2048_example.py](./2048_example.py)

## Key Insight: LLM Writes Code, Not Moves

The LLM does NOT play move-by-move. Instead:
1. LLM receives a prompt asking it to write a `strategy(board)` function
2. LLM generates Python code wrapped in triple backticks
3. Code is extracted, sandboxed (no global access), and executed against the live game
4. Reward is based on whether the strategy works

This is **"code-as-policy"** — the LLM generates an algorithm, not individual actions.

## The Prompt

```
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "0", "1", "2", "3" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "0" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
```

## Three Reward Functions

| Function | Score | Condition |
|---|---|---|
| `function_works` | +1.0 | Valid Python that compiles |
| | -0.5 | Right structure but exec fails |
| | -2.0 | No function / syntax error |
| `no_cheating` | +1.0 | Only stdlib imports |
| | -20.0 | Non-stdlib imports |
| `strategy_succeeds` | +20.0 | Reaches tile 2048 |
| | +2.0 | Runs but doesn't win |
| | -1.0 | Timeout (>5 sec) |
| | -3.0 | Exception |

## Training Setup

- Model: `unsloth/gpt-oss-20b` with LoRA (r=4)
- Trainer: `trl.GRPOTrainer` with `trl.GRPOConfig`
- Dataset: 1000 copies of the same prompt (diversity from temperature=1.0)
- `num_generations=2`, `max_steps=600`, `lr=2e-4`, `optim=adamw_8bit`
- ~5 hours on T4, rewards start appearing after ~100 steps

## OpenEnv-Specific Patterns

```python
# Launch environment server
from unsloth import launch_openenv
launch_openenv = functools.partial(
    launch_openenv,
    working_directory=working_directory,
    server="envs.openspiel_env.server.app:app",
    environment={**os.environ, "OPENSPIEL_GAME": "2048", ...},
    openenv_class=OpenSpielEnv,
)

# Reset and step
port, openenv_process = launch_openenv(port, openenv_process)
result = openenv_process.reset()
result = openenv_process.step(OpenSpielAction(action_id=0, game_name="2048"))
```

## Safety Utilities (from Unsloth)

- `check_python_modules(code)` — returns (ok, info); ok=True if only stdlib imports
- `create_locked_down_function(code)` — sandboxed exec, no global variable leakage
- `execute_with_time_limit(seconds)` — decorator for timeout enforcement
