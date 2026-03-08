# Metrics for the Origami RL Environment

## Fold Validity

| Metric | Formula | Use |
|--------|---------|-----|
| Kawasaki violation | Σ |alternating_angle_sum - 180°| per vertex | Reward penalty |
| Maekawa violation | Σ |abs(M-V) - 2| per vertex | Hard constraint |
| Self-intersection count | Triangle-triangle intersection testing | Reward penalty |
| Rigid foldability | Kinematic simulation pass/fail | Gate reward |

## Compactness / Efficiency

| Metric | Formula | Use |
|--------|---------|-----|
| **Deployment ratio** | `area_folded / area_unfolded` | Primary reward signal |
| Volume compaction | `bbox_folded / bbox_unfolded` | Secondary reward |
| Fold count | Count of M + V edges | Efficiency penalty |
| Folding efficiency | `compaction_ratio / fold_count` | Combined metric |
| Packing efficiency | `material_volume / bbox_volume` | How well it fills space |

## Structural / Stress

| Metric | Formula | Use |
|--------|---------|-----|
| Cauchy strain | Per-vertex avg deviation of edge lengths | Reward penalty |
| Max strain | `max(strain_per_vertex)` | Must stay below material limit |
| Structural energy | Sum of constraint violation energies | Lower = more stable |

## Shape Similarity (for target-matching tasks)

| Metric | Formula | Use |
|--------|---------|-----|
| Chamfer distance | Avg nearest-point distance between shapes | Primary shape reward |
| Hausdorff distance | Max distance between shapes | Worst-case shape error |
| IoU (3D) | Voxelized intersection over union | Alternative shape reward |

## Foldability Quality

| Metric | Formula | Use |
|--------|---------|-----|
| Flat-foldability score | Sum of angular deviations from target | For flat-fold tasks |
| Deployability | Reverse fold simulation collision check | Engineering reward |
| Crease pattern complexity | Entropy of M/V assignments | Simplicity bonus |

## Composite Reward Function

```python
reward = (
    w1 * deployment_ratio          # How compact is the fold?
    - w2 * total_strain            # Physical validity
    - w3 * self_intersections      # No paper penetration
    - w4 * kawasaki_violation      # Angle consistency
    - w5 * fold_count_penalty      # Fewer folds = better
    + w6 * deployability_score     # Can it unfold cleanly?
    + w7 * shape_similarity        # If targeting a specific shape
)
```
