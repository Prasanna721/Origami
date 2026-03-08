# FOLD File Format — Our State Representation

**Spec**: https://edemaine.github.io/fold/doc/spec.html (v1.2)
**Format**: JSON | **GitHub**: https://github.com/edemaine/fold

The standard interchange format for computational origami. We use this as our internal state.

## Key Fields

**Vertices:**
- `vertices_coords`: `[[x,y], [x,y,z], ...]`

**Edges:**
- `edges_vertices`: `[[u,v], ...]` — endpoint pairs
- `edges_assignment`: fold type per edge:
  - `"M"` Mountain | `"V"` Valley | `"B"` Boundary | `"F"` Flat | `"U"` Unassigned
- `edges_foldAngle`: degrees [-180, 180]; positive=valley, negative=mountain, 0=flat

**Faces:**
- `faces_vertices`: vertex IDs in counterclockwise order

**Layer Ordering:**
- `faceOrders`: `[f, g, s]` triples — s=+1 (f above g), s=-1 (f below g)

**Metadata:**
- `frame_classes`: `"creasePattern"`, `"foldedForm"`, `"graph"`
- `frame_unit`: `"unit"`, `"mm"`, `"cm"`, `"m"`

## Python Usage

```python
import json

with open("model.fold") as f:
    data = json.load(f)

vertices = data["vertices_coords"]
edges = data["edges_vertices"]
assignments = data["edges_assignment"]    # ["M", "V", "B", ...]
fold_angles = data.get("edges_foldAngle") # [-180, 180, 0, ...]
faces = data["faces_vertices"]
```

## Why It Matters
- Lingua franca of all origami software
- JSON = trivially parseable, serializable for replay buffers
- Encodes everything: geometry, fold types, angles, layer ordering
- Both Ghassaei's simulator and PyOri support it
