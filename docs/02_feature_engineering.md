# Phase 2 — Feature Engineering

> **Scripts:** `scripts/extract_features.py`, `scripts/analyze_features.py`  
> **Source:** `src/features/` package  
> **Output:** `data/X.npy`, `data/y.npy`, `data/features.jsonl`, `data/feature_names.json`

---

## Overview

The feature engineering pipeline transforms raw CAD shapes into a fixed-length numeric feature vector of **60 dimensions** suitable for machine learning. Features are extracted from two complementary sources:

1. **Base Geometric Features** (36 features) — bounding box, topology counts, ratios, tolerance metrics, and interaction terms
2. **Graph-Based Features** (24 features) — derived from the Face Adjacency Graph (FAG) of the B-Rep

The pipeline reconstructs each shape from stored parameters, extracts both feature sets, and outputs numpy arrays (`X.npy`, `y.npy`) alongside a human-readable JSONL log.

---

## Pipeline Architecture

```
dataset.jsonl
    │
    ▼
┌──────────────────┐
│  Reconstruct     │  Match record → generator → build(params)
│  OCP Shape       │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  Base  │ │  Graph   │
│Features│ │ Features │
└────┬───┘ └────┬─────┘
     │          │
     └────┬─────┘
          ▼
  ┌───────────────┐
  │ 60-dim vector │  → X.npy (n_samples × 60)
  │ + label       │  → y.npy (n_samples,)
  └───────────────┘
```

### Shape Reconstruction

The pipeline uses the `_reconstruct_shape()` function to rebuild each shape from its stored parameters. It maps `intended_label_name` to the appropriate generator, extracts `param_*` fields, and calls `build()` to recreate the exact OCP shape. Samples with construction errors or missing shapes are skipped.

### Labeling Strategy

Labels use `intended_label` (not the kernel check result), because degenerate and tolerance shapes build successfully in OCC but still represent failure patterns the model must learn to detect.

---

## Feature Categories

### 1. Bounding Box Features (5)

| Feature | Description |
|---------|-------------|
| `bbox_lx`, `bbox_ly`, `bbox_lz` | Axis-aligned bounding box dimensions |
| `bbox_vol` | Bounding box volume (lx × ly × lz) |
| `bbox_diag` | Bounding box diagonal length |

### 2. Sorted Dimensions (3)

| Feature | Description |
|---------|-------------|
| `dim_min` | Smallest bounding box dimension |
| `dim_mid` | Middle bounding box dimension |
| `dim_max` | Largest bounding box dimension |

### 3. Geometric Ratios (3)

| Feature | Description |
|---------|-------------|
| `aspect_ratio` | `dim_max / dim_min` — measures elongation |
| `mid_ratio` | `dim_mid / dim_min` — measures flatness |
| `compactness` | `volume / bbox_vol` — how well the shape fills its bounding box |

### 4. Volume & Surface Area (2)

| Feature | Description |
|---------|-------------|
| `volume` | Shape volume via GProp |
| `surface_area` | Total surface area via GProp |

### 5. Topology Counts (9)

| Feature | Description |
|---------|-------------|
| `n_vertices` | Number of B-Rep vertices |
| `n_edges` | Number of B-Rep edges |
| `n_wires` | Number of wires (closed loops of edges) |
| `n_faces` | Number of B-Rep faces |
| `n_shells` | Number of shells |
| `n_solids` | Number of solids |
| `n_compsolids` | Number of compound solids |
| `n_compounds` | Number of compounds |
| `euler_char` | Euler characteristic: V − E + F |

### 6. Construction Flags (3)

| Feature | Description |
|---------|-------------|
| `has_boolean_op` | 1.0 if sub-family involves boolean/union operations |
| `has_compound` | 1.0 if shape contains compound entities |
| `is_multi_solid` | 1.0 if shape has more than 1 solid |

### 7. Tolerance Metrics (3)

| Feature | Description |
|---------|-------------|
| `min_dim_over_tol` | `dim_min / OCC_DEFAULT_TOLERANCE` — how far the smallest dimension is from tolerance |
| `log_min_dim` | log₁₀(dim_min) — log-scale minimum dimension |
| `log_volume` | log₁₀(|volume|) — log-scale volume |

### 8. Interaction Features (8)

Cross-term combinations designed to capture non-linear relationships:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `dim_min_x_tol_ratio` | dim_min × min_dim_over_tol | Tolerance sensitivity signal |
| `vol_sa_ratio` | volume / surface_area | Sphericity / shape efficiency |
| `face_edge_ratio` | n_faces / n_edges | Topology regularity |
| `shell_solid_ratio` | n_shells / n_solids | Topology consistency |
| `log_aspect_ratio` | log₁₀(aspect_ratio) | Numerically stable elongation |
| `area_per_face` | surface_area / n_faces | Face homogeneity |
| `dim_min_x_tolerance` | dim_min × 1e-7 | Micro-scale geometry indicator |
| `aspect_x_compactness` | aspect_ratio × compactness | Elongated-yet-hollow detection |

### 9. Graph Structure Features (11)

Derived from the Face Adjacency Graph (FAG):

| Feature | Description |
|---------|-------------|
| `graph_n_nodes` | Number of faces (graph nodes) |
| `graph_n_edges` | Number of shared edges (graph edges) |
| `graph_avg_degree` | Mean face connectivity |
| `graph_max_degree` | Maximum face connectivity |
| `graph_density` | Graph density (0–1) |
| `graph_n_components` | Number of disconnected components |
| `graph_isolated_faces` | Faces with no adjacency |
| `graph_degree_std` | Standard deviation of node degrees |
| `graph_avg_loops_per_face` | Average wire loops per face |
| `graph_max_loops_per_face` | Maximum wire loops on any face |
| `graph_multi_loop_faces` | Faces with more than one loop |

### 10. Graph Face Type Distribution (7)

Fraction of faces by surface type:

| Feature | Surface Type |
|---------|-------------|
| `graph_frac_plane` | Planar faces |
| `graph_frac_cylinder` | Cylindrical faces |
| `graph_frac_cone` | Conical faces |
| `graph_frac_sphere` | Spherical faces |
| `graph_frac_torus` | Toroidal faces |
| `graph_frac_bspline` | B-spline surfaces |
| `graph_frac_other` | Other surface types |

### 11. Graph Area & Edge Statistics (6)

| Feature | Description |
|---------|-------------|
| `graph_mean_face_area` | Mean face area |
| `graph_std_face_area` | Standard deviation of face areas |
| `graph_max_face_area` | Largest face area |
| `graph_min_face_area` | Smallest face area |
| `graph_mean_edge_len` | Mean shared-edge length |
| `graph_max_edge_len` | Maximum shared-edge length |

---

## Face Adjacency Graph (FAG)

The FAG is built by `src/features/graph_features.py`:

1. **Node creation** — Each B-Rep face becomes a graph node, annotated with its area and surface type
2. **Edge creation** — For every B-Rep edge, the two incident faces are identified; a graph edge is created between them with the edge length as weight
3. **Statistics** — Degree distribution, connectivity, component counts, and area/length statistics are computed

This graph representation captures structural topology patterns that distinguish valid shapes from non-manifold or self-intersecting ones (e.g., isolated face components, unusual connectivity patterns).

---

## Feature Importance Analysis

The `analyze_features.py` script performs:

1. **Data quality checks** — NaN/Inf counts, low-variance feature detection
2. **Gini importance** — Random Forest feature importances ranked by information gain
3. **Correlation analysis** — Flags pairs with |r| > 0.95
4. **Zero-importance pruning** — Removes features with zero Gini importance
5. **Pipeline serialization** — Saves a `StandardScaler + RandomForest` pipeline to `models/feature_pipeline.joblib`

### Top Features (by Gini Importance)

The most predictive features for failure classification tend to be:

- **`compactness`** — Primary separator for valid vs. failed shapes
- **`n_solids`** — Multi-solid indicator (non-manifold signal)
- **`n_shells`** — Shell count excess (topology abnormality)
- **`is_multi_solid`** — Binary multi-solid flag
- **`graph_n_components`** — Disconnected face subgraphs

---

## Usage

```bash
# Extract features from dataset
python scripts/extract_features.py

# Custom input/output paths
python scripts/extract_features.py --input data/dataset.jsonl --output-dir data

# Run feature importance analysis
python scripts/analyze_features.py --data-dir data
```

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `X.npy` | NumPy array (n × 60) | Feature matrix |
| `y.npy` | NumPy array (n,) | Label vector (int64, values 0–4) |
| `features.jsonl` | JSON Lines | Per-sample feature records with metadata |
| `feature_names.json` | JSON array | Ordered list of 60 feature column names |
| `feature_importance.json` | JSON dict | Feature → Gini importance mapping |
| `selected_features.json` | JSON array | Features retained after zero-importance pruning |

---

## Canonical Feature Order

The `FEATURE_NAMES` list in `src/features/__init__.py` defines the **exact column order** for `X.npy`. Every model trained on this data expects features in this order. The `extract_features()` function ensures output arrays conform to this ordering by filling missing features with 0.0.
