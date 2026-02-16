"""
Feature extraction package for CAD validity prediction.

Provides:
  - extract_base_features(shape, record) -> dict of base geometric features
  - extract_graph_features(shape) -> dict with graph_stats, graph_nodes, graph_edges
  - extract_all_features(shape, record) -> dict merging base + graph stats
  - extract_features(params, shape_optional) -> numpy array (fixed length)
  - build_pipeline() -> sklearn Pipeline with StandardScaler
  - FEATURE_NAMES: ordered list of all feature column names
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_features import extract_base_features
from .graph_features import extract_graph_features


def extract_all_features(
    shape, record: dict[str, Any] | None = None
) -> dict[str, float]:
    """
    Extract all features (base + graph) from an OCP shape.

    Returns a flat dict of float values suitable for converting to a numpy row.
    """
    features: dict[str, float] = {}

    try:
        base = extract_base_features(shape, record)
        features.update(base)
    except Exception:
        pass

    try:
        graph = extract_graph_features(shape)
        features.update(graph["graph_stats"])
    except Exception:
        pass

    return features


# ── Canonical feature order ────────────────────────────────────────
# Defines the EXACT column order for X matrices. Every feature
# extractor output key should appear here exactly once.
FEATURE_NAMES: list[str] = [
    # Bounding box (5)
    "bbox_lx",
    "bbox_ly",
    "bbox_lz",
    "bbox_vol",
    "bbox_diag",
    # Sorted dims (3)
    "dim_min",
    "dim_mid",
    "dim_max",
    # Ratios (3)
    "aspect_ratio",
    "mid_ratio",
    "compactness",
    # Volume / area (2)
    "volume",
    "surface_area",
    # Topology counts (9)
    "n_vertices",
    "n_edges",
    "n_wires",
    "n_faces",
    "n_shells",
    "n_solids",
    "n_compsolids",
    "n_compounds",
    "euler_char",
    # Flags (3)
    "has_boolean_op",
    "has_compound",
    "is_multi_solid",
    # Tolerance metrics (3)
    "min_dim_over_tol",
    "log_min_dim",
    "log_volume",
    # Interaction features (8)
    "dim_min_x_tol_ratio",
    "vol_sa_ratio",
    "face_edge_ratio",
    "shell_solid_ratio",
    "log_aspect_ratio",
    "area_per_face",
    "dim_min_x_tolerance",
    "aspect_x_compactness",
    # Graph structure (11)
    "graph_n_nodes",
    "graph_n_edges",
    "graph_avg_degree",
    "graph_max_degree",
    "graph_density",
    "graph_n_components",
    "graph_isolated_faces",
    "graph_degree_std",
    "graph_avg_loops_per_face",
    "graph_max_loops_per_face",
    "graph_multi_loop_faces",
    # Graph face type distribution (7)
    "graph_frac_plane",
    "graph_frac_cylinder",
    "graph_frac_cone",
    "graph_frac_sphere",
    "graph_frac_torus",
    "graph_frac_bspline",
    "graph_frac_other",
    # Graph area/edge stats (6)
    "graph_mean_face_area",
    "graph_std_face_area",
    "graph_max_face_area",
    "graph_min_face_area",
    "graph_mean_edge_len",
    "graph_max_edge_len",
]


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a numpy array in canonical order."""
    return np.array(
        [features.get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def extract_features(
    params: dict[str, Any],
    shape=None,
    record: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Unified feature extraction entry-point.

    Parameters
    ----------
    params : dict
        Construction parameters (param_* fields from the dataset record).
        Currently used for metadata only; shape-derived features dominate.
    shape : TopoDS_Shape or None
        If provided, all geometric + graph features are extracted.
        If None, returns a zero-filled array of fixed length.
    record : dict, optional
        Full dataset record (for metadata flags like sub_family).

    Returns
    -------
    np.ndarray of shape (n_features,)
        Fixed-length feature vector matching FEATURE_NAMES order.
    """
    if shape is None:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    # Build record context from params if not supplied
    if record is None:
        record = {f"param_{k}": v for k, v in params.items()}

    features = extract_all_features(shape, record)
    return features_to_array(features)


def build_pipeline():
    """
    Build an sklearn Pipeline with StandardScaler for feature preprocessing.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with a single StandardScaler step. Fit on training data
        before use, or extend with a classifier step downstream.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )


__all__ = [
    "extract_base_features",
    "extract_graph_features",
    "extract_all_features",
    "extract_features",
    "features_to_array",
    "build_pipeline",
    "FEATURE_NAMES",
]
