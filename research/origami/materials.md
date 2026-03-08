# Material Properties & Stress Simulation

## Material Types

| Material | Young's Modulus | Thickness | Use Case |
|----------|----------------|-----------|----------|
| Standard paper | ~2-4 GPa | 0.1 mm | Baseline origami |
| Kami (origami paper) | ~2 GPa | 0.07 mm | Thin, easy to fold |
| **Mylar** | ~3-5 GPa | 0.01-0.1 mm | **Space solar panels** |
| **Aluminum foil** | ~70 GPa | 0.02 mm | **Deployable structures** |
| Foil-backed paper | Composite | 0.1-0.2 mm | Metal + paper laminate |
| Tyvek (synthetic) | ~0.2 GPa | 0.15 mm | Tear-resistant shelters |

## How Material Affects the Environment

Material properties change:
- **Fold radius constraint** — thicker/stiffer = larger minimum bend radius
- **Max layers** — thickness limits how many layers can stack
- **Stress at creases** — Young's modulus × curvature = stress concentration
- **Strain tolerance** — each material has a failure threshold

## Stress Visualization

**Ghassaei's approach (what we port):**
- Per-vertex strain = avg percent deviation of edge lengths from rest lengths
- Color map: blue (0 strain) → red (max strain)
- This is **Cauchy strain / engineering strain**

**For RL reward:**
- `total_strain = mean(per_vertex_strain)` — lower = better physical validity
- `max_strain` — must stay below material failure threshold
- `crease_stress = f(fold_angle, thickness, Young's_modulus)` — per-crease penalty

## Thickness Considerations
- Real paper has nonzero thickness → gaps at vertices where layers meet
- SWOMPS reference: panel=1mm, crease=0.3mm, panel E=2000MPa, crease E=20MPa
- For engineering apps: offset panels, tapered hinges accommodate thickness
- Start zero-thickness, add thickness as advanced feature

## How to Use in Prompt

```
Material: Mylar (space-grade)
- Thickness: 0.05mm
- Young's modulus: 4 GPa
- Max strain before failure: 3%
- Max fold layers: 12

Sheet: 1m × 1m
Constraint: Pack into 10cm × 10cm × 5cm
Must deploy to > 0.8m² area
```
