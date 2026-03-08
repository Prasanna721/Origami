# Origami Simulation Engines

## Ghassaei's Origami Simulator (Top Pick)

- **URL**: https://origamisimulator.org/ | [GitHub](https://github.com/amandaghassaei/OrigamiSimulator)
- **License**: MIT | **Language**: JavaScript/WebGL

**Physics model (truss model):**
- Folds all creases simultaneously via iterative constraint solving
- Three constraint types:
  - **Distance** — prevent stretching/compressing
  - **Face** — prevent shearing
  - **Angular** — fold or flatten the sheet
- Each constraint has a **stiffness parameter** (higher = more rigid)
- All computation runs in **GPU fragment shaders** for real-time

**Strain visualization:**
- Blue (no strain) → Red (max strain)
- Strain = avg percent deviation of distance constraints (Cauchy/engineering strain)

**I/O**: SVG, FOLD → FOLD, STL, OBJ

**For us**: Port the truss model physics to NumPy. The constraint system maps directly to our reward signal.

---

## rigid-origami (Existing RL Gym Environment)

- **GitHub**: https://github.com/belalugaX/rigid-origami
- **Paper**: IJCAI 2023 — "Automating Rigid Origami Design"

Already has Gym API, PPO, MCTS, evolutionary agents. Formulates origami as a board game on a grid. Validation via triangle-triangle intersection + kinematic checks.

**For us**: Study the validation logic. The grid-based vertex placement can be our starting point.

---

## SWOMPS (Multi-Physics, MATLAB)

- **GitHub**: https://github.com/zzhuyii/OrigamiSimulator
- Large deformation + heat transfer + contact detection
- Five different solvers
- Companion dataset generator: https://github.com/zzhuyii/GenerateOrigamiDataSet

**For us**: Reference for stress/contact physics only. Not directly usable (MATLAB).

---

## Tachi's Software (Foundational, Proprietary)

- Rigid Origami Simulator, Freeform Origami, Origamizer
- **Origamizer**: universal — can fold ANY 3D polyhedral surface
- Windows-only, proprietary

**For us**: Algorithms are key references. Origamizer algorithm (Tachi & Demaine, SoCG 2017) proves universality.

---

## Others (Low Priority)

| Tool | Language | Notes |
|------|----------|-------|
| Origami Editor 3D | Java | GUI editor |
| Oriedita | Java | Crease pattern editor, FOLD I/O |
| VPython Origami | Python | Minimal, unmaintained |
