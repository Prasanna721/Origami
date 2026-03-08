# Python Libraries for Origami RL

## Origami-Specific

### rigid-origami (TOP PICK)
- **GitHub**: https://github.com/belalugaX/rigid-origami
- Gym environment for origami design as board game
- Has: PPO, MCTS, evolutionary, BFS, DFS agents
- Has: triangle-triangle intersection + kinematic validation
- Has: action masking, symmetry enforcement
- Output: PNG patterns, OBJ models, GIF animations
- Install: conda env + `pip install gym-rori`

### PyOri
- `pip3 install pyori`
- Graph-based crease pattern generation
- Huzita-Justin axiom implementation
- Flat-foldability checks
- FOLD + SVG export
- Dependencies: numpy, matplotlib, scipy

### FoldZ
- https://github.com/generic-github-user/FoldZ
- Early development, ambitious scope, not production-ready

## Supporting Libraries

| Library | What We Use It For |
|---------|-------------------|
| **numpy** | Core geometry, linear algebra |
| **scipy** | Constraint solving, optimization, spatial queries |
| **trimesh** | Mesh ops, collision detection, STL/OBJ I/O |
| **matplotlib** | 2D crease pattern visualization |
| **plotly** | 3D interactive visualization (Gradio-compatible) |
| **networkx** | Crease pattern graph analysis |
| **shapely** | 2D polygon operations, area calculations |
| **gymnasium** | RL environment API standard |

## Key Sources

- [Ghassaei Simulator](https://github.com/amandaghassaei/OrigamiSimulator) — MIT, JS/WebGL
- [FOLD format](https://github.com/edemaine/fold) — JSON standard
- [TreeMaker](https://langorigami.com/article/treemaker/) — crease pattern from tree diagrams
- [SWOMPS](https://github.com/zzhuyii/OrigamiSimulator) — MATLAB multi-physics reference
- [Tachi's tools](https://origami.c.u-tokyo.ac.jp/~tachi/software/) — foundational algorithms
