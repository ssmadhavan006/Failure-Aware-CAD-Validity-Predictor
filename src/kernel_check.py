"""
Kernel validity checker — wraps OCP BRepCheck_Analyzer and topology
inspection to classify CAD shapes as valid or assign a failure mode.

Extended checks beyond BRepCheck:
  - Multi-solid compound detection (non-manifold indicator)
  - Open shell / non-solid detection
  - Topology consistency checks
"""

from __future__ import annotations

from typing import Any

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer


def _count_entities(shape, entity_type) -> int:
    """Count topology entities of a given type in *shape*."""
    n = 0
    exp = TopExp_Explorer(shape, entity_type)
    while exp.More():
        n += 1
        exp.Next()
    return n


def count_topology(shape) -> dict[str, int]:
    """Return a dict of topology entity counts for *shape*."""
    return {
        "n_vertices": _count_entities(shape, TopAbs_VERTEX),
        "n_edges": _count_entities(shape, TopAbs_EDGE),
        "n_wires": _count_entities(shape, TopAbs_WIRE),
        "n_faces": _count_entities(shape, TopAbs_FACE),
        "n_shells": _count_entities(shape, TopAbs_SHELL),
        "n_solids": _count_entities(shape, TopAbs_SOLID),
        "n_compsolids": _count_entities(shape, TopAbs_COMPSOLID),
        "n_compounds": _count_entities(shape, TopAbs_COMPOUND),
    }


def _check_manifold(shape, topology: dict[str, int]) -> dict[str, Any]:
    """
    Check for non-manifold conditions beyond BRepCheck:
      - Multiple disconnected solids in a compound
      - Shapes without any solid (non-solid geometry)
      - Compounds without proper fusion
    """
    issues = []

    n_solids = topology["n_solids"]
    n_faces = topology["n_faces"]
    n_shells = topology["n_shells"]

    # Multiple solids in what should be a single body
    if n_solids > 1:
        issues.append(f"multi_solid_compound({n_solids})")

    # Shape has faces but no solid — indicates an open shell or sheet body
    if n_faces > 0 and n_solids == 0:
        issues.append("no_solid_topology")

    # Compound with no solids (just floating faces/edges/shells)
    if topology["n_compounds"] > 0 and n_solids == 0:
        issues.append("compound_no_solid")

    # More shells than solids can indicate incomplete topology
    if n_shells > n_solids and n_solids > 0:
        issues.append(f"extra_shells({n_shells}vs{n_solids})")

    return {
        "is_manifold": len(issues) == 0,
        "manifold_issues": issues,
    }


def check_shape(shape) -> dict[str, Any]:
    """
    Run BRepCheck_Analyzer + extended checks on *shape*.

    Returns
    -------
    dict with keys:
        is_valid     : bool — BRepCheck valid AND manifold
        brep_valid   : bool — raw BRepCheck result
        is_manifold  : bool — passes manifold checks
        manifold_issues : list[str]
        topology     : dict[str, int]
    """
    topology = count_topology(shape)

    # Core BRepCheck
    analyzer = BRepCheck_Analyzer(shape)
    brep_valid = analyzer.IsValid()

    # Extended manifold checks
    manifold = _check_manifold(shape, topology)

    # Overall validity: must pass BOTH checks
    is_valid = brep_valid and manifold["is_manifold"]

    return {
        "is_valid": is_valid,
        "brep_valid": brep_valid,
        "is_manifold": manifold["is_manifold"],
        "manifold_issues": manifold["manifold_issues"],
        "topology": topology,
    }
