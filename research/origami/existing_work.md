# Existing RL & ML Work on Origami

## "Automating Rigid Origami Design" — IJCAI 2023 (Most Relevant)

- **Paper**: https://www.ijcai.org/proceedings/2023/0645.pdf
- **Code**: https://github.com/belalugaX/rigid-origami

**RL Formulation:**
- Environment = board game on a grid
- State = current vertex graph forming crease pattern
- Actions = place vertices on grid (with action masking for symmetry)
- Reward = sparse, based on objective (shape approximation, packaging)
- Validation = triangle-triangle intersection + kinematic check

**Algorithms tested:** Random, BFS, DFS, MCTS, Evolutionary, PPO
**Finding:** Action space grows exponentially. Symmetry masking is critical.

## ML for Origami Inverse Design — Nature 2022

- Decision tree / random forest for inverse design
- Given desired mechanical properties → predict crease pattern parameters
- Relevant for: how to define reward functions around structural properties

## UCLA Robotic Origami (2023)

- Deep learning for robotic paper folding
- RL for physical execution of folds (robot arm)
- Different scope (physical manipulation, not pattern design)

## Key Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Bern & Hayes | 1996 | Global flat-foldability is NP-complete |
| Lang, "Computational origami" | 1996 | TreeMaker algorithm |
| Tachi & Demaine, "Origamizer" | 2017 | Universal folding algorithm |
| Ghassaei et al. | 2018 | GPU truss model simulation |
| Lang, "Additive algorithm" | 2021 | Scalable local origami design |
| Nature, "Algorithmic origami" | 2022 | Optimization frameworks |
| IJCAI, "Automating Rigid Origami" | 2023 | RL for origami design |

## Origami AI Roadmap (Community)

Source: https://origami.kosmulski.org/blog/2023-01-12-origami-ai-roadmap

Key insight — a minimal AI origami system needs:
1. Intent → stick model converter (**the hard/missing part**)
2. Stick model → crease pattern (TreeMaker exists)
3. CP simulation/visualization (Ghassaei exists)

Training data scarcity is a major challenge.
