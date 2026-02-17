"""
Base geometric feature extraction from OCP shapes.

Extracts a fixed-size feature vector:
  - Bounding box dimensions (L, W, H, volume)
  - Ratios (aspect ratio, compactness)
  - Topology counts (vertices, edges, faces, shells, solids)
  - Boolean flags (has_boolean, has_compound)
  - Tolerance-related metrics (min dimension, min_dim / default_tol)
"""

from __future__ import annotations

from typing import Any

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
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

# Default OCC linear tolerance
OCC_DEFAULT_TOLERANCE = 1e-7


def _count(shape, entity_type) -> int:
    """Count topology entities of a given type."""
    n = 0
    exp = TopExp_Explorer(shape, entity_type)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _bounding_box(shape) -> dict[str, float]:
    """Compute axis-aligned bounding box dimensions."""
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    if bbox.IsVoid():
        return {
            "bbox_lx": 0.0,
            "bbox_ly": 0.0,
            "bbox_lz": 0.0,
            "bbox_vol": 0.0,
            "bbox_diag": 0.0,
        }
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    lx = max(xmax - xmin, 0.0)
    ly = max(ymax - ymin, 0.0)
    lz = max(zmax - zmin, 0.0)
    vol = lx * ly * lz
    diag = (lx**2 + ly**2 + lz**2) ** 0.5
    return {
        "bbox_lx": lx,
        "bbox_ly": ly,
        "bbox_lz": lz,
        "bbox_vol": vol,
        "bbox_diag": diag,
    }


def _volume_and_area(shape) -> dict[str, float]:
    """Compute volume and surface area via GProp."""
    vol_props = GProp_GProps()
    area_props = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, vol_props)
        volume = vol_props.Mass()
    except Exception:
        volume = 0.0
    try:
        BRepGProp.SurfaceProperties_s(shape, area_props)
        area = area_props.Mass()
    except Exception:
        area = 0.0
    return {"volume": volume, "surface_area": area}


def extract_base_features(
    shape, record: dict[str, Any] | None = None
) -> dict[str, float]:
    """
    Extract base geometric features from an OCP shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCP shape to extract features from.
    record : dict, optional
        The dataset record (used for metadata flags like sub_family).

    Returns
    -------
    dict[str, float]
        Flat dictionary of numeric features.
    """
    features: dict[str, float] = {}

    # Bounding box
    bb = _bounding_box(shape)
    features.update(bb)

    dims = sorted([bb["bbox_lx"], bb["bbox_ly"], bb["bbox_lz"]])
    features["dim_min"] = dims[0]
    features["dim_mid"] = dims[1]
    features["dim_max"] = dims[2]

    # Ratios
    features["aspect_ratio"] = dims[2] / dims[0] if dims[0] > 1e-15 else 1e12
    features["mid_ratio"] = dims[1] / dims[0] if dims[0] > 1e-15 else 1e12

    vol_area = _volume_and_area(shape)
    features.update(vol_area)

    bbox_vol = bb["bbox_vol"]
    features["compactness"] = vol_area["volume"] / bbox_vol if bbox_vol > 1e-30 else 0.0

    # Topology counts
    features["n_vertices"] = _count(shape, TopAbs_VERTEX)
    features["n_edges"] = _count(shape, TopAbs_EDGE)
    features["n_wires"] = _count(shape, TopAbs_WIRE)
    features["n_faces"] = _count(shape, TopAbs_FACE)
    features["n_shells"] = _count(shape, TopAbs_SHELL)
    features["n_solids"] = _count(shape, TopAbs_SOLID)
    features["n_compsolids"] = _count(shape, TopAbs_COMPSOLID)
    features["n_compounds"] = _count(shape, TopAbs_COMPOUND)

    # Euler characteristic: V - E + F (indicative of genus/holes)
    features["euler_char"] = (
        features["n_vertices"] - features["n_edges"] + features["n_faces"]
    )

    # Boolean / construction flags
    if record:
        sf = record.get("sub_family", "")
        features["has_boolean_op"] = (
            1.0 if "boolean" in sf or "union" in sf or "mismatch" in sf else 0.0
        )
        features["has_compound"] = 1.0 if features["n_compounds"] > 0 else 0.0
        features["is_multi_solid"] = 1.0 if features["n_solids"] > 1 else 0.0
    else:
        features["has_boolean_op"] = 0.0
        features["has_compound"] = 1.0 if features["n_compounds"] > 0 else 0.0
        features["is_multi_solid"] = 1.0 if features["n_solids"] > 1 else 0.0

    # Tolerance metrics
    min_dim = features["dim_min"]
    features["min_dim_over_tol"] = (
        min_dim / OCC_DEFAULT_TOLERANCE if min_dim > 0 else 0.0
    )
    features["log_min_dim"] = (
        __import__("math").log10(min_dim) if min_dim > 1e-15 else -15.0
    )
    features["log_volume"] = (
        __import__("math").log10(abs(vol_area["volume"]))
        if abs(vol_area["volume"]) > 1e-30
        else -30.0
    )

    # Interaction features (cross-term combinations)
    # Product of smallest dim and tolerance proximity — predictive of tolerance errors
    features["dim_min_x_tol_ratio"] = min_dim * features["min_dim_over_tol"]

    # Volume / surface area ratio (sphericity indicator)
    sa = vol_area["surface_area"]
    features["vol_sa_ratio"] = vol_area["volume"] / sa if sa > 1e-15 else 0.0

    # Face-to-edge ratio (topology regularity)
    n_e = features["n_edges"]
    features["face_edge_ratio"] = features["n_faces"] / n_e if n_e > 0 else 0.0

    # Topology imbalance: shells vs solids
    n_sol = features["n_solids"]
    features["shell_solid_ratio"] = features["n_shells"] / n_sol if n_sol > 0 else 0.0

    # Log aspect ratio (more numerically stable for ML)
    features["log_aspect_ratio"] = (
        __import__("math").log10(features["aspect_ratio"])
        if features["aspect_ratio"] > 0 and features["aspect_ratio"] < 1e15
        else 15.0
    )

    # Surface area per face (homogeneity indicator)
    n_f = features["n_faces"]
    features["area_per_face"] = sa / n_f if n_f > 0 else 0.0

    # Product of smallest dimension and default tolerance
    # (captures micro-scale geometry prone to tolerance errors)
    features["dim_min_x_tolerance"] = min_dim * OCC_DEFAULT_TOLERANCE

    # Aspect ratio × compactness (elongated-yet-hollow geometries)
    features["aspect_x_compactness"] = (
        features["aspect_ratio"] * features["compactness"]
    )

    return features
