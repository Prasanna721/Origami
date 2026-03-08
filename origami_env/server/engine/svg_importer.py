"""Parse OrigamiSimulator SVG crease patterns into PaperState objects.

SVG Stroke-Color Convention (from OrigamiSimulator):

| Stroke Color        | Assignment | Fold Angle           |
|---------------------|------------|----------------------|
| #000000 / black     | B (Boundary)   | N/A              |
| #FF0000 / red       | M (Mountain)   | -opacity * 180   |
| #0000FF / blue      | V (Valley)     | +opacity * 180   |
| #FFFF00 / yellow    | F (Facet)      | 0                |
| #FF00FF / magenta   | U (Unassigned) | 0                |
| #00FF00 / green     | C (Cut->Boundary) | N/A           |
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .paper import PaperState, _faces_from_edges


# ── Color -> assignment mapping ──────────────────────────────────────────

_COLOR_MAP: dict[str, str] = {
    "#000000": "B",
    "#000":    "B",
    "black":   "B",
    "rgb(0,0,0)": "B",
    "#ff0000": "M",
    "#f00":    "M",
    "red":     "M",
    "rgb(255,0,0)": "M",
    "#0000ff": "V",
    "#00f":    "V",
    "blue":    "V",
    "rgb(0,0,255)": "V",
    "#00ff00": "C",
    "#0f0":    "C",
    "green":   "C",
    "rgb(0,255,0)": "C",
    "#ffff00": "F",
    "#ff0":    "F",
    "yellow":  "F",
    "rgb(255,255,0)": "F",
    "#ff00ff": "U",
    "#f0f":    "U",
    "magenta": "U",
    "rgb(255,0,255)": "U",
}

# Also handle 3-char hex with mixed case (e.g. "#000080" maps via 6-char)
# We normalize to lowercase 6-char hex in _normalize_color().


# ── Public API ───────────────────────────────────────────────────────────

def parse_svg(svg_content: str, vertex_tol: float | None = None) -> PaperState:
    """Parse SVG crease pattern string into PaperState.

    Parameters
    ----------
    svg_content : str
        Raw SVG XML string.
    vertex_tol : float, optional
        Distance threshold for merging nearby vertices.
        Default: max(width, height) / 500.

    Returns
    -------
    PaperState
        Parsed crease pattern as a PaperState object.
    """
    root = ET.fromstring(svg_content)

    # Collect raw vertices and segments (edges with assignment + fold angle)
    raw_vertices: list[tuple[float, float]] = []
    raw_segments: list[tuple[int, int, str, float]] = []  # (v0, v1, assignment, foldAngle)

    # Process all supported SVG element types
    _process_all_elements(root, raw_vertices, raw_segments)

    if not raw_vertices or not raw_segments:
        raise ValueError("No valid geometry found in SVG content.")

    # Convert to numpy arrays for processing
    verts = np.array(raw_vertices, dtype=np.float64)  # (N, 2)

    # Compute merge tolerance
    if vertex_tol is None:
        x_range = verts[:, 0].max() - verts[:, 0].min()
        y_range = verts[:, 1].max() - verts[:, 1].min()
        max_dim = max(x_range, y_range)
        vertex_tol = max_dim / 500.0 if max_dim > 0 else 1e-6

    # Merge nearby vertices
    verts, index_map = _merge_vertices(verts, vertex_tol)

    # Remap edge vertex indices
    edges_vertices = []
    edges_assignment = []
    edges_foldAngle = []

    for v0, v1, assignment, angle in raw_segments:
        new_v0 = index_map[v0]
        new_v1 = index_map[v1]

        # Skip loop edges (both endpoints merged to same vertex)
        if new_v0 == new_v1:
            continue

        # Canonical ordering for duplicate detection
        edge_key = (min(new_v0, new_v1), max(new_v0, new_v1))
        edges_vertices.append((new_v0, new_v1))
        edges_assignment.append(assignment)
        edges_foldAngle.append(angle)

    # Remove duplicate edges (keep first occurrence with priority: B > M > V > C > F > U)
    edges_vertices, edges_assignment, edges_foldAngle = _remove_duplicate_edges(
        edges_vertices, edges_assignment, edges_foldAngle
    )

    if not edges_vertices:
        raise ValueError("No valid edges remain after merging and deduplication.")

    # Find edge-edge intersections and insert new vertices at crossing points.
    # This handles cases like diagonals crossing in origami patterns.
    verts, edges_vertices, edges_assignment, edges_foldAngle = _find_intersections(
        verts, edges_vertices, edges_assignment, edges_foldAngle, vertex_tol
    )

    # Merge any newly-coincident vertices
    verts, index_map2 = _merge_vertices(verts, vertex_tol)
    edges_vertices = [(index_map2[v0], index_map2[v1]) for v0, v1 in edges_vertices]
    # Remove loops created by merge
    filtered = [(e, a, f) for e, a, f in zip(edges_vertices, edges_assignment, edges_foldAngle) if e[0] != e[1]]
    if filtered:
        edges_vertices, edges_assignment, edges_foldAngle = zip(*filtered)
        edges_vertices = list(edges_vertices)
        edges_assignment = list(edges_assignment)
        edges_foldAngle = list(edges_foldAngle)
    else:
        edges_vertices, edges_assignment, edges_foldAngle = [], [], []

    # Remove duplicates after intersection
    edges_vertices, edges_assignment, edges_foldAngle = _remove_duplicate_edges(
        edges_vertices, edges_assignment, edges_foldAngle
    )

    if not edges_vertices:
        raise ValueError("No valid edges remain after merging and deduplication.")

    # Split edges at T-intersections (vertices lying on edges that aren't endpoints).
    # This is necessary for proper face detection.
    verts, edges_vertices, edges_assignment, edges_foldAngle = _split_edges_at_vertices(
        verts, edges_vertices, edges_assignment, edges_foldAngle, vertex_tol
    )

    # Remove duplicates again after splitting
    edges_vertices, edges_assignment, edges_foldAngle = _remove_duplicate_edges(
        edges_vertices, edges_assignment, edges_foldAngle
    )

    # Normalize coordinates to 0-1 range
    verts_norm = _normalize_coords(verts)

    # Build 3D coords (x, y, 0) for PaperState
    coords_3d = np.column_stack([verts_norm, np.zeros(len(verts_norm))])

    # Build FOLD-compatible dict and construct PaperState
    edges_np = np.array(edges_vertices, dtype=np.int32)

    # Find faces via half-edge traversal
    faces = _faces_from_edges(coords_3d, edges_np)

    fold_data = {
        "vertices_coords": coords_3d.tolist(),
        "edges_vertices": edges_np.tolist(),
        "edges_assignment": list(edges_assignment),
        "edges_foldAngle": list(edges_foldAngle),
        "faces_vertices": faces,
    }

    return PaperState.from_fold_json(fold_data)


def load_svg_file(path: str) -> PaperState:
    """Load SVG file from disk and parse into PaperState.

    Parameters
    ----------
    path : str
        Path to an SVG file.

    Returns
    -------
    PaperState
    """
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_svg(content)


def list_bundled_examples() -> list[dict]:
    """List bundled OrigamiSimulator SVG examples.

    Returns
    -------
    list[dict]
        Each dict has keys: ``name``, ``category``, ``path``.
    """
    examples_dir = Path(__file__).resolve().parent.parent.parent / "examples"
    results = []

    if not examples_dir.is_dir():
        return results

    for category_dir in sorted(examples_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for svg_file in sorted(category_dir.glob("*.svg")):
            results.append({
                "name": svg_file.stem,
                "category": category,
                "path": str(svg_file),
            })

    return results


# ── SVG element processing ───────────────────────────────────────────────

def _process_all_elements(
    root: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Walk the SVG tree and extract geometry from all supported elements."""
    # Process elements recursively (handles <g> groups, nested elements, etc.)
    _process_element_recursive(root, raw_vertices, raw_segments)


def _process_element_recursive(
    element: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Recursively process an SVG element and its children."""
    tag = _local_tag(element)

    if tag == "line":
        _parse_line(element, raw_vertices, raw_segments)
    elif tag == "path":
        _parse_path(element, raw_vertices, raw_segments)
    elif tag == "rect":
        _parse_rect(element, raw_vertices, raw_segments)
    elif tag == "polygon":
        _parse_polygon(element, raw_vertices, raw_segments)
    elif tag == "polyline":
        _parse_polyline(element, raw_vertices, raw_segments)

    # Recurse into children (handles <g>, <svg>, etc.)
    for child in element:
        _process_element_recursive(child, raw_vertices, raw_segments)


def _parse_line(
    elem: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Parse an SVG <line> element."""
    stroke = _get_stroke(elem)
    assignment = _assignment_for_stroke(stroke)
    if assignment is None:
        return

    opacity = _get_opacity(elem)
    fold_angle = _fold_angle(assignment, opacity)

    x1 = float(elem.get("x1", "0"))
    y1 = float(elem.get("y1", "0"))
    x2 = float(elem.get("x2", "0"))
    y2 = float(elem.get("y2", "0"))

    idx_start = len(raw_vertices)
    raw_vertices.append((x1, y1))
    raw_vertices.append((x2, y2))
    raw_segments.append((idx_start, idx_start + 1, assignment, fold_angle))


def _parse_rect(
    elem: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Parse an SVG <rect> element into 4 edges."""
    stroke = _get_stroke(elem)
    assignment = _assignment_for_stroke(stroke)
    if assignment is None:
        return

    opacity = _get_opacity(elem)
    fold_angle = _fold_angle(assignment, opacity)

    x = float(elem.get("x", "0"))
    y = float(elem.get("y", "0"))
    w = float(elem.get("width", "0"))
    h = float(elem.get("height", "0"))

    idx = len(raw_vertices)
    raw_vertices.append((x, y))
    raw_vertices.append((x + w, y))
    raw_vertices.append((x + w, y + h))
    raw_vertices.append((x, y + h))

    raw_segments.append((idx, idx + 1, assignment, fold_angle))
    raw_segments.append((idx + 1, idx + 2, assignment, fold_angle))
    raw_segments.append((idx + 2, idx + 3, assignment, fold_angle))
    raw_segments.append((idx + 3, idx, assignment, fold_angle))


def _parse_polygon(
    elem: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Parse an SVG <polygon> element (closed shape)."""
    stroke = _get_stroke(elem)
    assignment = _assignment_for_stroke(stroke)
    if assignment is None:
        return

    opacity = _get_opacity(elem)
    fold_angle = _fold_angle(assignment, opacity)
    points = _parse_points_attr(elem.get("points", ""))
    if len(points) < 2:
        return

    idx_start = len(raw_vertices)
    for pt in points:
        raw_vertices.append(pt)

    for i in range(len(points) - 1):
        raw_segments.append((idx_start + i, idx_start + i + 1, assignment, fold_angle))
    # Close the polygon
    raw_segments.append((idx_start + len(points) - 1, idx_start, assignment, fold_angle))


def _parse_polyline(
    elem: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Parse an SVG <polyline> element (open shape)."""
    stroke = _get_stroke(elem)
    assignment = _assignment_for_stroke(stroke)
    if assignment is None:
        return

    opacity = _get_opacity(elem)
    fold_angle = _fold_angle(assignment, opacity)
    points = _parse_points_attr(elem.get("points", ""))
    if len(points) < 2:
        return

    idx_start = len(raw_vertices)
    for pt in points:
        raw_vertices.append(pt)

    for i in range(len(points) - 1):
        raw_segments.append((idx_start + i, idx_start + i + 1, assignment, fold_angle))


def _parse_path(
    elem: ET.Element,
    raw_vertices: list[tuple[float, float]],
    raw_segments: list[tuple[int, int, str, float]],
) -> None:
    """Parse an SVG <path> element.

    Supports commands: M, m, L, l, H, h, V, v, Z, z.
    Curves (C, S, Q, T, A) are not supported and are skipped.
    """
    stroke = _get_stroke(elem)
    assignment = _assignment_for_stroke(stroke)
    if assignment is None:
        return

    opacity = _get_opacity(elem)
    fold_angle = _fold_angle(assignment, opacity)

    d = elem.get("d", "")
    if not d.strip():
        return

    commands = _tokenize_path_d(d)
    if not commands:
        return

    # Current point (absolute coordinates)
    cx, cy = 0.0, 0.0
    # Start of current subpath (for Z command)
    start_vertex_idx: int | None = None
    first_command = True

    for cmd_type, args in commands:
        if cmd_type == "M":
            # Absolute moveto: each pair after the first is an implicit L
            pairs = _pairs(args)
            for i, (x, y) in enumerate(pairs):
                cx, cy = x, y
                if i == 0:
                    start_vertex_idx = len(raw_vertices)
                    raw_vertices.append((cx, cy))
                else:
                    # Implicit lineto
                    raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                    raw_vertices.append((cx, cy))
            first_command = False

        elif cmd_type == "m":
            # Relative moveto
            pairs = _pairs(args)
            for i, (dx, dy) in enumerate(pairs):
                if i == 0 and first_command:
                    # First "m" in path is treated as absolute
                    cx, cy = dx, dy
                else:
                    if i == 0:
                        cx += dx
                        cy += dy
                    else:
                        # Implicit relative lineto
                        raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                        cx += dx
                        cy += dy
                if i == 0:
                    start_vertex_idx = len(raw_vertices)
                    raw_vertices.append((cx, cy))
                else:
                    raw_vertices.append((cx, cy))
            first_command = False

        elif cmd_type == "L":
            # Absolute lineto
            pairs = _pairs(args)
            for x, y in pairs:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cx, cy = x, y
                raw_vertices.append((cx, cy))

        elif cmd_type == "l":
            # Relative lineto
            pairs = _pairs(args)
            for dx, dy in pairs:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cx += dx
                cy += dy
                raw_vertices.append((cx, cy))

        elif cmd_type == "H":
            # Absolute horizontal lineto
            for x in args:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cx = x
                raw_vertices.append((cx, cy))

        elif cmd_type == "h":
            # Relative horizontal lineto
            for dx in args:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cx += dx
                raw_vertices.append((cx, cy))

        elif cmd_type == "V":
            # Absolute vertical lineto
            for y in args:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cy = y
                raw_vertices.append((cx, cy))

        elif cmd_type == "v":
            # Relative vertical lineto
            for dy in args:
                raw_segments.append((len(raw_vertices) - 1, len(raw_vertices), assignment, fold_angle))
                cy += dy
                raw_vertices.append((cx, cy))

        elif cmd_type in ("Z", "z"):
            # Close path: connect current point back to subpath start
            if start_vertex_idx is not None and len(raw_vertices) > 0:
                raw_segments.append((len(raw_vertices) - 1, start_vertex_idx, assignment, fold_angle))
            start_vertex_idx = None

        # Skip unsupported commands (C, c, S, s, Q, q, T, t, A, a)


# ── Path d-attribute tokenizer ───────────────────────────────────────────

def _tokenize_path_d(d: str) -> list[tuple[str, list[float]]]:
    """Tokenize an SVG path d-attribute into (command, [args]) tuples.

    Handles the full range of number formats including:
    - Negative numbers as implicit separators: ``10-20`` -> ``10, -20``
    - Scientific notation: ``1e-3``
    - Decimal points as separators: ``1.5.3`` -> ``1.5, 0.3``
    """
    commands: list[tuple[str, list[float]]] = []
    # Match command letters and their subsequent numbers
    # Command letters: M m L l H h V v Z z C c S s Q q T t A a
    tokens = re.findall(r'[MmLlHhVvZzCcSsQqTtAa]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', d)

    current_cmd = None
    current_args: list[float] = []

    for token in tokens:
        if token.isalpha() and len(token) == 1:
            # Save previous command
            if current_cmd is not None:
                commands.append((current_cmd, current_args))
            current_cmd = token
            current_args = []
        else:
            try:
                current_args.append(float(token))
            except ValueError:
                pass

    # Save last command
    if current_cmd is not None:
        commands.append((current_cmd, current_args))

    return commands


def _pairs(args: list[float]) -> list[tuple[float, float]]:
    """Group a flat list of floats into (x, y) pairs."""
    result = []
    for i in range(0, len(args) - 1, 2):
        result.append((args[i], args[i + 1]))
    return result


def _parse_points_attr(points_str: str) -> list[tuple[float, float]]:
    """Parse an SVG ``points`` attribute (used by polygon/polyline)."""
    points_str = points_str.strip()
    if not points_str:
        return []
    # Points can be separated by spaces or commas
    nums = re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', points_str)
    result = []
    for i in range(0, len(nums) - 1, 2):
        result.append((float(nums[i]), float(nums[i + 1])))
    return result


# ── Stroke / color helpers ───────────────────────────────────────────────

def _local_tag(elem: ET.Element) -> str:
    """Get the local tag name, stripping any namespace."""
    tag = elem.tag
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return tag


def _get_stroke(elem: ET.Element) -> str | None:
    """Extract the stroke color from an SVG element.

    Checks the ``stroke`` attribute first, then parses the inline
    ``style`` attribute for a ``stroke:`` declaration.
    """
    # Direct attribute
    stroke = elem.get("stroke")
    if stroke:
        return _normalize_color(stroke)

    # Inline style
    style = elem.get("style", "")
    if style:
        match = re.search(r'(?:^|;)\s*stroke\s*:\s*([^;]+)', style, re.IGNORECASE)
        if match:
            return _normalize_color(match.group(1).strip())

    return None


def _get_opacity(elem: ET.Element) -> float:
    """Extract the effective opacity from an SVG element.

    Checks ``opacity``, ``stroke-opacity`` attributes and inline style.
    Returns the product of opacity and stroke-opacity (both default to 1.0).
    """
    opacity = _parse_opacity_value(elem.get("opacity"))
    stroke_opacity = _parse_opacity_value(elem.get("stroke-opacity"))

    # Also check inline style
    style = elem.get("style", "")
    if style:
        m_op = re.search(r'(?:^|;)\s*opacity\s*:\s*([^;]+)', style, re.IGNORECASE)
        if m_op:
            val = _parse_opacity_value(m_op.group(1).strip())
            if val is not None:
                opacity = val

        m_sop = re.search(r'(?:^|;)\s*stroke-opacity\s*:\s*([^;]+)', style, re.IGNORECASE)
        if m_sop:
            val = _parse_opacity_value(m_sop.group(1).strip())
            if val is not None:
                stroke_opacity = val

    op = opacity if opacity is not None else 1.0
    sop = stroke_opacity if stroke_opacity is not None else 1.0
    return op * sop


def _parse_opacity_value(val: str | None) -> float | None:
    """Parse a numeric opacity value, returning None if invalid."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _normalize_color(color: str) -> str:
    """Normalize a color string to lowercase.

    Handles hex shorthand (#abc -> #aabbcc), named colors, and rgb().
    """
    color = color.strip().lower()
    color = re.sub(r'\s+', '', color)  # remove whitespace

    # Expand 3-digit hex to 6-digit
    m = re.match(r'^#([0-9a-f])([0-9a-f])([0-9a-f])$', color)
    if m:
        color = f"#{m.group(1)*2}{m.group(2)*2}{m.group(3)*2}"

    # Normalize rgb() format: remove spaces
    m = re.match(r'^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$', color)
    if m:
        color = f"rgb({m.group(1)},{m.group(2)},{m.group(3)})"

    return color


def _assignment_for_stroke(stroke: str | None) -> str | None:
    """Map an SVG stroke color to a FOLD edge assignment.

    Returns None if the color is not recognized (element should be skipped).
    """
    if stroke is None:
        return None

    # Try direct lookup
    assignment = _COLOR_MAP.get(stroke)
    if assignment is not None:
        return assignment

    # Try interpreting as hex and matching to nearest known color
    hex_match = re.match(r'^#([0-9a-f]{6})$', stroke)
    if hex_match:
        r = int(hex_match.group(1)[0:2], 16)
        g = int(hex_match.group(1)[2:4], 16)
        b = int(hex_match.group(1)[4:6], 16)

        # Use thresholds for fuzzy matching to standard origami colors
        # Black: low all channels
        if r < 64 and g < 64 and b < 64:
            return "B"
        # Red: high R, low G, low B
        if r > 192 and g < 64 and b < 64:
            return "M"
        # Blue: low R, low G, high B (also handle navy #000080)
        if r < 64 and g < 64 and b > 64:
            return "V"
        # Green: low R, high G, low B
        if r < 64 and g > 192 and b < 64:
            return "C"
        # Yellow: high R, high G, low B
        if r > 192 and g > 192 and b < 64:
            return "F"
        # Magenta: high R, low G, high B
        if r > 192 and g < 64 and b > 192:
            return "U"

    return None


def _fold_angle(assignment: str, opacity: float) -> float:
    """Compute the target fold angle from assignment and opacity.

    Mountain: -opacity * 180
    Valley:   +opacity * 180
    Others:   0.0
    """
    if assignment == "M":
        return -opacity * 180.0
    elif assignment == "V":
        return opacity * 180.0
    else:
        return 0.0


# ── Vertex merging ───────────────────────────────────────────────────────

def _merge_vertices(
    verts: np.ndarray, tol: float
) -> tuple[np.ndarray, list[int]]:
    """Merge vertices within tolerance distance.

    Parameters
    ----------
    verts : np.ndarray
        (N, 2) array of 2D vertex positions.
    tol : float
        Distance threshold for merging.

    Returns
    -------
    merged_verts : np.ndarray
        (M, 2) array of unique vertex positions.
    index_map : list[int]
        Maps old vertex index -> new vertex index (length N).
    """
    n = len(verts)
    index_map = list(range(n))

    # Union-find: group vertices that are within tol
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # For small vertex counts, brute-force O(n^2) is fine.
    # For large counts, use a spatial approach.
    if n <= 5000:
        for i in range(n):
            for j in range(i + 1, n):
                dx = verts[i, 0] - verts[j, 0]
                dy = verts[i, 1] - verts[j, 1]
                if dx * dx + dy * dy <= tol * tol:
                    union(i, j)
    else:
        # Grid-based spatial hashing for larger models
        inv_tol = 1.0 / tol if tol > 0 else 1e12
        grid: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            gx = int(math.floor(verts[i, 0] * inv_tol))
            gy = int(math.floor(verts[i, 1] * inv_tol))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        for j in grid[key]:
                            ddx = verts[i, 0] - verts[j, 0]
                            ddy = verts[i, 1] - verts[j, 1]
                            if ddx * ddx + ddy * ddy <= tol * tol:
                                union(i, j)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

    # Flatten and compute merged positions (average of group)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    new_verts = []
    new_idx_map = [0] * n
    for group_indices in groups.values():
        new_id = len(new_verts)
        # Average position of group
        avg = np.mean(verts[group_indices], axis=0)
        new_verts.append(avg)
        for idx in group_indices:
            new_idx_map[idx] = new_id

    return np.array(new_verts, dtype=np.float64), new_idx_map


# ── Edge deduplication ───────────────────────────────────────────────────

_ASSIGNMENT_PRIORITY = {"B": 0, "M": 1, "V": 2, "C": 3, "F": 4, "U": 5}


def _remove_duplicate_edges(
    edges: list[tuple[int, int]],
    assignments: list[str],
    fold_angles: list[float],
) -> tuple[list[tuple[int, int]], list[str], list[float]]:
    """Remove duplicate edges, keeping the one with highest priority assignment."""
    seen: dict[tuple[int, int], int] = {}  # canonical edge -> index in output
    out_edges = []
    out_assign = []
    out_angles = []

    for i, (v0, v1) in enumerate(edges):
        key = (min(v0, v1), max(v0, v1))
        if key in seen:
            existing_idx = seen[key]
            existing_pri = _ASSIGNMENT_PRIORITY.get(out_assign[existing_idx], 99)
            new_pri = _ASSIGNMENT_PRIORITY.get(assignments[i], 99)
            if new_pri < existing_pri:
                # Replace with higher priority
                out_edges[existing_idx] = (v0, v1)
                out_assign[existing_idx] = assignments[i]
                out_angles[existing_idx] = fold_angles[i]
        else:
            seen[key] = len(out_edges)
            out_edges.append((v0, v1))
            out_assign.append(assignments[i])
            out_angles.append(fold_angles[i])

    return out_edges, out_assign, out_angles


# ── Edge splitting at T-intersections ────────────────────────────────────

def _split_edges_at_vertices(
    verts: np.ndarray,
    edges: list[tuple[int, int]],
    assignments: list[str],
    fold_angles: list[float],
    tol: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[str], list[float]]:
    """Split edges where a vertex lies on the edge interior (T-intersection).

    This handles the common case in origami SVGs where a crease line
    endpoint lies on a boundary edge but is not an explicit boundary
    vertex (e.g. a fold line meeting the paper boundary).

    Parameters
    ----------
    verts : np.ndarray
        (N, 2) vertex positions.
    edges, assignments, fold_angles : list
        Edge data parallel arrays.
    tol : float
        Distance tolerance for point-on-edge test.

    Returns
    -------
    Updated (verts, edges, assignments, fold_angles).
    """
    # Build a set of all vertex indices that are edge endpoints
    n_verts = len(verts)

    # We may need multiple passes since splitting can create new collinear situations
    changed = True
    max_passes = 5
    pass_count = 0

    while changed and pass_count < max_passes:
        changed = False
        pass_count += 1

        new_edges: list[tuple[int, int]] = []
        new_assignments: list[str] = []
        new_fold_angles: list[float] = []

        for edge_idx, (v0, v1) in enumerate(edges):
            p0 = verts[v0]
            p1 = verts[v1]

            # Find vertices that lie on this edge (excluding endpoints)
            split_verts: list[tuple[float, int]] = []  # (t_param, vertex_index)

            edge_vec = p1 - p0
            edge_len_sq = edge_vec[0] ** 2 + edge_vec[1] ** 2
            if edge_len_sq < tol * tol:
                # Degenerate edge
                new_edges.append((v0, v1))
                new_assignments.append(assignments[edge_idx])
                new_fold_angles.append(fold_angles[edge_idx])
                continue

            for vi in range(len(verts)):
                if vi == v0 or vi == v1:
                    continue

                pt = verts[vi]
                # Project pt onto line p0-p1
                t = ((pt[0] - p0[0]) * edge_vec[0] + (pt[1] - p0[1]) * edge_vec[1]) / edge_len_sq

                if t < tol / math.sqrt(edge_len_sq) or t > 1.0 - tol / math.sqrt(edge_len_sq):
                    continue  # Not interior to edge

                # Check distance from pt to the line
                proj = p0 + t * edge_vec
                dist_sq = (pt[0] - proj[0]) ** 2 + (pt[1] - proj[1]) ** 2
                if dist_sq <= tol * tol:
                    split_verts.append((t, vi))

            if split_verts:
                changed = True
                # Sort by parameter along edge
                split_verts.sort(key=lambda x: x[0])

                # Create chain of sub-edges
                prev = v0
                for _, sv in split_verts:
                    new_edges.append((prev, sv))
                    new_assignments.append(assignments[edge_idx])
                    new_fold_angles.append(fold_angles[edge_idx])
                    prev = sv
                # Final sub-edge
                new_edges.append((prev, v1))
                new_assignments.append(assignments[edge_idx])
                new_fold_angles.append(fold_angles[edge_idx])
            else:
                new_edges.append((v0, v1))
                new_assignments.append(assignments[edge_idx])
                new_fold_angles.append(fold_angles[edge_idx])

        edges = new_edges
        assignments = new_assignments
        fold_angles = new_fold_angles

    return verts, edges, assignments, fold_angles


# ── Edge-edge intersection detection ────────────────────────────────────

def _find_intersections(
    verts: np.ndarray,
    edges: list[tuple[int, int]],
    assignments: list[str],
    fold_angles: list[float],
    tol: float,
) -> tuple[np.ndarray, list[tuple[int, int]], list[str], list[float]]:
    """Find all edge-edge intersections and insert new vertices.

    For each pair of non-adjacent edges that cross, inserts a new vertex
    at the intersection point and splits both edges.

    Returns updated (verts, edges, assignments, fold_angles).
    """
    verts_list = list(map(tuple, verts.tolist()))

    # Collect all intersection points per edge
    # edge_splits[edge_idx] = list of (t_param, new_vertex_idx)
    edge_splits: dict[int, list[tuple[float, int]]] = {}

    n_edges = len(edges)
    for i in range(n_edges):
        v0i, v1i = edges[i]
        p0 = verts[v0i]
        p1 = verts[v1i]

        for j in range(i + 1, n_edges):
            v0j, v1j = edges[j]

            # Skip edges that share a vertex (adjacent edges)
            if v0i == v0j or v0i == v1j or v1i == v0j or v1i == v1j:
                continue

            p2 = verts[v0j]
            p3 = verts[v1j]

            result = _segment_intersection(p0, p1, p2, p3, tol)
            if result is None:
                continue

            ti, tj, ix, iy = result

            # Add the intersection vertex
            new_idx = len(verts_list)
            verts_list.append((ix, iy))

            if i not in edge_splits:
                edge_splits[i] = []
            edge_splits[i].append((ti, new_idx))

            if j not in edge_splits:
                edge_splits[j] = []
            edge_splits[j].append((tj, new_idx))

    if not edge_splits:
        return verts, edges, assignments, fold_angles

    # Rebuild edges, splitting those that have intersection points
    new_edges: list[tuple[int, int]] = []
    new_assignments: list[str] = []
    new_fold_angles: list[float] = []

    for edge_idx in range(n_edges):
        v0, v1 = edges[edge_idx]
        a = assignments[edge_idx]
        fa = fold_angles[edge_idx]

        if edge_idx in edge_splits:
            splits = sorted(edge_splits[edge_idx], key=lambda x: x[0])
            prev = v0
            for _, sv in splits:
                new_edges.append((prev, sv))
                new_assignments.append(a)
                new_fold_angles.append(fa)
                prev = sv
            new_edges.append((prev, v1))
            new_assignments.append(a)
            new_fold_angles.append(fa)
        else:
            new_edges.append((v0, v1))
            new_assignments.append(a)
            new_fold_angles.append(fa)

    new_verts = np.array(verts_list, dtype=np.float64)
    return new_verts, new_edges, new_assignments, new_fold_angles


def _segment_intersection(
    p0: np.ndarray, p1: np.ndarray,
    p2: np.ndarray, p3: np.ndarray,
    tol: float,
) -> tuple[float, float, float, float] | None:
    """Find the intersection of segments p0-p1 and p2-p3.

    Returns (t, u, x, y) where t and u are parameters along each segment
    (both strictly in (0, 1) exclusive of endpoints), or None if no
    interior intersection exists.
    """
    dx1 = p1[0] - p0[0]
    dy1 = p1[1] - p0[1]
    dx2 = p3[0] - p2[0]
    dy2 = p3[1] - p2[1]

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        return None  # Parallel or coincident

    dx3 = p2[0] - p0[0]
    dy3 = p2[1] - p0[1]

    t = (dx3 * dy2 - dy3 * dx2) / denom
    u = (dx3 * dy1 - dy3 * dx1) / denom

    # Both parameters must be strictly between 0 and 1 (interior intersection)
    edge_len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    edge_len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    eps1 = tol / edge_len1 if edge_len1 > 0 else 0.01
    eps2 = tol / edge_len2 if edge_len2 > 0 else 0.01

    if t < eps1 or t > 1.0 - eps1 or u < eps2 or u > 1.0 - eps2:
        return None

    ix = p0[0] + t * dx1
    iy = p0[1] + t * dy1

    return t, u, ix, iy


# ── Coordinate normalization ─────────────────────────────────────────────

def _normalize_coords(verts: np.ndarray) -> np.ndarray:
    """Normalize 2D vertex coordinates to the 0-1 range.

    The longer axis is scaled to [0, 1] and the shorter axis is
    scaled proportionally (maintaining aspect ratio), centered.
    """
    if len(verts) == 0:
        return verts.copy()

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    ranges = maxs - mins

    scale = max(ranges[0], ranges[1])
    if scale < 1e-12:
        # All points coincide
        return np.zeros_like(verts)

    normalized = (verts - mins) / scale
    return normalized
