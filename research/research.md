# Origami RL Environment — Research Index

## What We're Building
An OpenEnv environment where an LLM learns to design optimal origami fold patterns — solar panel packing, deployable structures, medical stents. LLM generates a `fold_strategy()` function (code-as-policy), executed against a bar-and-hinge physics simulation.

---

## Architecture (START HERE)

| File | What's In It |
|------|-------------|
| **[plan/architecture.md](plan/architecture.md)** | **Full architecture: action space, state, physics, rewards, rendering, project structure, implementation order** |
| **[plan/openenv_arch.md](plan/openenv_arch.md)** | **Complete OpenEnv environment: repo structure, Pydantic models, engine (paper/fold/physics/validation/metrics/materials), renderer (2D/3D/screenshots/GIF recording/export), environment class, React frontend, app+Docker, client, task system, API reference, deployment, testing** |

### Decisions (Locked)

| Decision | Choice |
|----------|--------|
| LLM interaction | Code-as-policy (LLM writes `fold_strategy()` function) |
| Action space | Named fold ops (valley/mountain + fold line + angle) |
| State format | FOLD-compatible JSON |
| Physics engine | Bar-and-hinge model (NumPy port of Ghassaei) |
| Validation | Kawasaki + Maekawa + triangle-triangle intersection |
| Primary task | Solar panel packing (Miura-ori discovery) |
| Training render | matplotlib headless |
| Demo render | React + @react-three/fiber |
| Training | GRPO via TRL + Unsloth on Colab |
| Deployment | Docker Space on HF Spaces |

---

## OpenEnv (The Framework)

| File | What's In It |
|------|-------------|
| [openenv/overview.md](openenv/overview.md) | OpenEnv architecture, API, types, project structure, deployment |
| [openenv/2048_pattern.md](openenv/2048_pattern.md) | Code-as-policy pattern, reward functions, GRPO training |
| [openenv/2048_example.py](openenv/2048_example.py) | Full extracted code from Unsloth 2048 Colab (636 lines) |

---

## Origami Domain Knowledge

### Quick Reference
| File | What's In It |
|------|-------------|
| [origami/fold_types_deep.md](origami/fold_types_deep.md) | **All fold operations**, Huzita-Justin axioms, crane step-by-step (31 steps), compression patterns (Miura-ori, Kresling, flasher), complexity scaling |
| [origami/math_physics_deep.md](origami/math_physics_deep.md) | **Kawasaki/Maekawa theorems** with code, bar-and-hinge model, energy formulas, strain computation, rigid foldability, computational complexity table |
| [origami/rendering_research.md](origami/rendering_research.md) | **Rendering options**: Ghassaei simulator, OrigamiOdyssey (R3F), Three.js in React, Gradio integration, recording |
| [origami/applications_deep.md](origami/applications_deep.md) | **Real-world apps**: NASA solar panels, JWST, stents, self-folding robots, metamaterials |

### Earlier Research (Summaries)
| File | What's In It |
|------|-------------|
| [origami/simulation_engines.md](origami/simulation_engines.md) | Ghassaei, rigid-origami Gym env, SWOMPS, Tachi |
| [origami/fold_format.md](origami/fold_format.md) | FOLD file format — JSON standard for crease patterns |
| [origami/physics.md](origami/physics.md) | Physics summary (Kawasaki, Maekawa, simulation approaches) |
| [origami/materials.md](origami/materials.md) | Material properties (paper, mylar, aluminum), stress viz |
| [origami/metrics.md](origami/metrics.md) | All metrics: validity, compactness, stress, shape similarity |
| [origami/existing_work.md](origami/existing_work.md) | Prior work: IJCAI 2023, Nature 2022, UCLA robotics |
| [origami/python_tools.md](origami/python_tools.md) | Libraries: rigid-origami, PyOri, numpy, trimesh |

---

## Deliverables Checklist

- [ ] Engine: paper.py, fold_engine.py, physics.py, validation.py, metrics.py
- [ ] OpenEnv server: models.py, origami_environment.py, app.py, Dockerfile
- [ ] Reward functions: code_valid, physically_valid, fold_quality
- [ ] Training notebook: Colab with GRPO + Unsloth/TRL
- [ ] Rendering: matplotlib (training) + React/R3F (demo)
- [ ] Deploy to HF Spaces
- [ ] 1-minute demo video on YouTube
- [ ] Public GitHub repo
