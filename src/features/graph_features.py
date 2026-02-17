"""
Graph-based feature extraction from OCP shapes.

Builds a Face Adjacency Graph (FAG):
  - Nodes  = BRep faces (with area, surface type)
  - Edges  = shared BRep edges between adjacent faces (with length)

Also computes fixed-size graph statistics for tabular ML:
  - Number of nodes/edges, average degree, max degree
  - Face type distribution (plane, cylinder, sphere, etc.)
  - Average / std of face areas and edge lengths
"""

from __future__ import annotations

from collections import deque
from typing import Any

from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer

from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
)

# Map GeomAbs surface types to readable names
_SURFACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_BezierSurface: "bezier",
}


def _face_area(face) -> float:
    """Compute surface area of a face."""
    props = GProp_GProps()
    try:
        BRepGProp.SurfaceProperties_s(face, props)
        return abs(props.Mass())
    except Exception:
        return 0.0


def _edge_length(edge) -> float:
    """Compute length of an edge."""
    props = GProp_GProps()
    try:
        BRepGProp.LinearProperties_s(edge, props)
        return abs(props.Mass())
    except Exception:
        return 0.0


def _surface_type(face) -> str:
    """Get the surface type name of a face."""
    try:
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()
        return _SURFACE_TYPE_NAMES.get(stype, "other")
    except Exception:
        return "unknown"


def extract_graph_features(shape) -> dict[str, Any]:
    """
    Extract Face Adjacency Graph and graph statistics from an OCP shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCP shape to analyse.

    Returns
    -------
    dict with keys:
        graph_nodes : list[dict]   -- per-face attributes (area, type)
        graph_edges : list[tuple]  -- adjacency pairs (i, j)
        graph_stats : dict[str, float] -- fixed-size graph statistics
    """
    # Collect faces via IndexedMap (gives stable indices)
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)

    n_faces = face_map.Extent()
    if n_faces == 0:
        return _empty_graph_result()

    # Node attributes
    nodes = []
    type_counts: dict[str, int] = {}
    areas: list[float] = []
    for i in range(1, n_faces + 1):  # 1-indexed in OCP
        face = TopoDS.Face_s(face_map.FindKey(i))
        area = _face_area(face)
        stype = _surface_type(face)
        nodes.append({"area": area, "type": stype})
        areas.append(area)
        type_counts[stype] = type_counts.get(stype, 0) + 1

    # Edge map: find shared edges between faces
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    adjacency: list[tuple[int, int]] = []
    edge_lengths: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()

    for e_idx in range(1, edge_face_map.Extent() + 1):
        edge = edge_face_map.FindKey(e_idx)
        adj_face_list = edge_face_map.FindFromIndex(e_idx)

        # Collect face indices using direct Python iteration
        adj_indices: list[int] = []
        for f in adj_face_list:
            idx = face_map.FindIndex(f)
            if idx > 0:
                adj_indices.append(idx - 1)  # convert to 0-indexed

        # Create pairwise edges
        for a in range(len(adj_indices)):
            for b in range(a + 1, len(adj_indices)):
                pair = (
                    min(adj_indices[a], adj_indices[b]),
                    max(adj_indices[a], adj_indices[b]),
                )
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    adjacency.append(pair)
                    edge_lengths.append(_edge_length(edge))

    n_graph_edges = len(adjacency)

    # Degree statistics
    degrees = [0] * n_faces
    for i, j in adjacency:
        degrees[i] += 1
        degrees[j] += 1

    avg_degree = sum(degrees) / n_faces if n_faces > 0 else 0.0
    max_degree = max(degrees) if degrees else 0
    degree_std = (
        (sum((d - avg_degree) ** 2 for d in degrees) / n_faces) ** 0.5
        if n_faces > 0
        else 0.0
    )
    isolated_faces = sum(1 for d in degrees if d == 0)

    # Connected components (BFS on adjacency)
    adj_list: dict[int, list[int]] = {i: [] for i in range(n_faces)}
    for i, j in adjacency:
        adj_list[i].append(j)
        adj_list[j].append(i)

    visited = [False] * n_faces
    n_components = 0
    for start in range(n_faces):
        if visited[start]:
            continue
        n_components += 1
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            for nb in adj_list[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    # Edge loops per face (count wires per face)
    wire_counts: list[int] = []
    for i in range(1, n_faces + 1):
        face = TopoDS.Face_s(face_map.FindKey(i))
        n_wires = 0
        wexp = TopExp_Explorer(face, TopAbs_WIRE)
        while wexp.More():
            n_wires += 1
            wexp.Next()
        wire_counts.append(n_wires)

    avg_loops = sum(wire_counts) / n_faces if n_faces > 0 else 0.0
    max_loops = max(wire_counts) if wire_counts else 0
    multi_loop_faces = sum(1 for w in wire_counts if w > 1)

    # Assemble graph statistics (fixed-size vector)
    stats: dict[str, float] = {
        "graph_n_nodes": float(n_faces),
        "graph_n_edges": float(n_graph_edges),
        "graph_avg_degree": avg_degree,
        "graph_max_degree": float(max_degree),
        "graph_density": (
            2.0 * n_graph_edges / (n_faces * (n_faces - 1)) if n_faces > 1 else 0.0
        ),
        "graph_n_components": float(n_components),
        "graph_isolated_faces": float(isolated_faces),
        "graph_degree_std": degree_std,
        "graph_avg_loops_per_face": avg_loops,
        "graph_max_loops_per_face": float(max_loops),
        "graph_multi_loop_faces": float(multi_loop_faces),
    }

    # Face type distribution
    for stype in ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]:
        stats[f"graph_frac_{stype}"] = type_counts.get(stype, 0) / n_faces

    # Area statistics
    if areas:
        mean_area = sum(areas) / len(areas)
        std_area = (sum((a - mean_area) ** 2 for a in areas) / len(areas)) ** 0.5
        stats["graph_mean_face_area"] = mean_area
        stats["graph_std_face_area"] = std_area
        stats["graph_max_face_area"] = max(areas)
        stats["graph_min_face_area"] = min(areas)
    else:
        stats["graph_mean_face_area"] = 0.0
        stats["graph_std_face_area"] = 0.0
        stats["graph_max_face_area"] = 0.0
        stats["graph_min_face_area"] = 0.0

    # Edge length statistics
    if edge_lengths:
        stats["graph_mean_edge_len"] = sum(edge_lengths) / len(edge_lengths)
        stats["graph_max_edge_len"] = max(edge_lengths)
    else:
        stats["graph_mean_edge_len"] = 0.0
        stats["graph_max_edge_len"] = 0.0

    return {
        "graph_nodes": nodes,
        "graph_edges": adjacency,
        "graph_stats": stats,
    }


def _empty_graph_result() -> dict[str, Any]:
    """Return empty graph result for shapes with no faces."""
    stats = {
        "graph_n_nodes": 0.0,
        "graph_n_edges": 0.0,
        "graph_avg_degree": 0.0,
        "graph_max_degree": 0.0,
        "graph_density": 0.0,
        "graph_n_components": 0.0,
        "graph_isolated_faces": 0.0,
        "graph_degree_std": 0.0,
        "graph_avg_loops_per_face": 0.0,
        "graph_max_loops_per_face": 0.0,
        "graph_multi_loop_faces": 0.0,
    }
    for stype in ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "other"]:
        stats[f"graph_frac_{stype}"] = 0.0
    stats["graph_mean_face_area"] = 0.0
    stats["graph_std_face_area"] = 0.0
    stats["graph_max_face_area"] = 0.0
    stats["graph_min_face_area"] = 0.0
    stats["graph_mean_edge_len"] = 0.0
    stats["graph_max_edge_len"] = 0.0
    return {"graph_nodes": [], "graph_edges": [], "graph_stats": stats}
