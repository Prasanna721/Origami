# Physics of Paper Folding

## Fundamental Theorems

### Kawasaki's Theorem (Angles)
- At a single vertex, flat-foldable iff alternating angle sums = 180 on each side
- **Necessary but not sufficient** for global flat-foldability
- Use as: reward penalty for violation

### Maekawa's Theorem (Mountain/Valley Count)
- At any flat-fold vertex: `|M - V| = 2`
- Total creases at a vertex must be **even**
- Use as: hard constraint in action validation

### Global Flat-Foldability
- **NP-complete** (Bern & Hayes, 1996)
- Hardness comes from determining valid **layer ordering**
- Local conditions (Kawasaki + Maekawa) necessary but not sufficient
- Use as: approximate via constraint satisfaction for reward

### Rigid Foldability
- Can it fold from flat to folded state with rigid panels (no face bending)?
- Critical for engineering: metal, plastic panels must be rigid
- Checked via: triangle-triangle intersection + rigid body constraints
- Use as: pass/fail validation check

## Layer Ordering
- When paper folds flat, layers stack — which face is above which?
- Must satisfy: no face penetration
- FOLD format: `faceOrders` triples `[f, g, s]`
- This is the NP-hard part — approximate for RL

## Simulation Approaches (Ranked for RL Use)

1. **Bar-and-hinge** (Ghassaei) — edges as bars, rotational springs at hinges. Fast. Best for RL.
2. **Rigid body + compliant creases** — rigid panels, torsional spring creases. Good middle ground.
3. **FEM** — full stress/strain tensor. Accurate but too slow for RL training loop.
