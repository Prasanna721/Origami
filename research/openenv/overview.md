# OpenEnv Framework (Meta) — What We Need to Know

**Repo**: https://github.com/meta-pytorch/OpenEnv
**Version**: 0.2.1 stable (required by hackathon), 0.2.2.dev0 in dev
**License**: BSD 3-Clause | **Python**: 3.10+

## Architecture

Client-server over WebSocket. Environment runs inside Docker as a FastAPI server.

```
LLM/Agent  <-->  EnvClient (WebSocket)  <-->  Environment (FastAPI server in Docker)
```

## Core Types (from `openenv.core.env_server.types`)

```python
class Action(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    done: bool = False
    reward: bool | int | float | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
```

## Creating a Custom Environment (Server Side)

Subclass `Environment[ActT, ObsT, StateT]`:

```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, seed=None, episode_id=None, **kwargs) -> ObsT:
        """Initialize episode, return initial observation."""

    def step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT:
        """Execute action, return observation (with .done and .reward set)."""

    @property
    def state(self) -> StateT:
        """Return current episode state."""
```

## Creating a Custom Client

Subclass `EnvClient[ActT, ObsT, StateT]`:

```python
from openenv.core.env_client import EnvClient

class MyEnvClient(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: ActT) -> Dict:
        """Convert action to JSON dict for wire transmission."""

    def _parse_result(self, payload: Dict) -> StepResult[ObsT]:
        """Parse server response into StepResult."""

    def _parse_state(self, payload: Dict) -> StateT:
        """Parse state response into your State type."""
```

## Project Structure (`openenv init my_env`)

```
my_env/
  __init__.py
  models.py          # Action, Observation, State subclasses
  client.py           # EnvClient subclass
  openenv.yaml        # Manifest
  pyproject.toml      # deps: "openenv-core[core]>=0.2.1"
  server/
    __init__.py
    my_env_environment.py   # Environment subclass
    app.py                  # create_app() call
    Dockerfile
    requirements.txt
```

## openenv.yaml Manifest

```yaml
spec_version: 1
name: my_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## Server App Entry Point

```python
from openenv.core.env_server.http_server import create_app
app = create_app(MyEnvironment, MyAction, MyObservation, env_name="my_env")
```

## Deploy to HF Spaces

```bash
openenv push [--repo-id username/my_env] [--private]
```

Dockerfile uses `ghcr.io/meta-pytorch/openenv-base:latest`, runs `uvicorn server.app:app --host 0.0.0.0 --port 8000`.

## Concrete Example: Chess Environment

```python
class ChessAction(Action):
    move: str  # UCI format "e2e4"

class ChessObservation(Observation):
    fen: str = ""
    legal_moves: List[str] = []
    is_check: bool = False
    result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"

class ChessState(State):
    fen: str = "rnbqkbnr/..."
    current_player: str = "white"
    move_history: List[str] = []
```

Reward: +1.0 win, -1.0 loss, 0.0 draw, -0.1 invalid move.

## Integration with GRPO Training

From `examples/grpo_blackjack/`:
1. `play_game()` — orchestrates env.reset() / env.step() loop
2. `format_prompt()` — converts game state to LLM prompt
3. `parse_action()` — extracts actions from LLM text
4. `simple_grpo_loss()` — GRPO with KL penalty
5. Compatible with: **TRL**, **Unsloth**, **SkyRL**, **ART**, **Oumi**
