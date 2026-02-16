# Phase 1 — Synthetic Data Generation

> **Script:** `scripts/generate_dataset.py`  
> **Output:** `data/dataset.jsonl`, `data/dataset_stats.json`

---

## Overview

The data generation pipeline creates a labeled dataset of **2,500 synthetic CAD samples** (500 per class) spanning 5 failure categories. Each sample is a parametrically generated 3D shape built with [CadQuery](https://cadquery.readthedocs.io/) and validated by the OpenCascade (OCP) kernel. The pipeline produces construction parameters, kernel validity results, and topology metadata — all serialized as JSON Lines for downstream feature extraction.

---

## Generator Architecture

All generators inherit from `BaseGenerator` (defined in `src/generators/base.py`), which enforces a two-method contract:

| Method | Responsibility |
|--------|---------------|
| `sample_params(rng)` | Draw random construction parameters using a NumPy RNG |
| `build(params)` | Construct the OCP shape from those parameters and return a `Sample` dataclass |

The `generate(rng)` convenience method chains both steps.

### Sample Dataclass

Each generator returns a `Sample` with the following fields:

```
params          dict        Construction parameters (flattened as param_* in output)
label           int         Intended failure class (0–4)
failure_mode    str         Human-readable failure name (e.g. "bowtie_extrude")
sub_family      str         Generator sub-family identifier
shape           TopoDS_Shape | None   The built OCP shape (None on construction error)
error_type      str | None  Exception class name if construction failed
error_msg       str | None  Truncated exception message
is_valid        bool | None Result of BRepCheck_Analyzer (set by kernel_check)
topology        dict        Topology entity counts (set by kernel_check)
```

### Generator Registry

All generators are registered in `src/generators/__init__.py` via the `ALL_GENERATORS` list, which the pipeline iterates over to produce balanced class samples.

---

## Failure Classes

| Label | Class Name | Generator File | Description |
|-------|-----------|----------------|-------------|
| 0 | `valid` | `valid.py` | Well-formed, kernel-valid geometries |
| 1 | `self_intersection` | `self_intersection.py` | Shapes with self-intersecting B-Rep faces |
| 2 | `non_manifold` | `non_manifold.py` | Topologically non-manifold constructions |
| 3 | `degenerate_face` | `degenerate.py` | Degenerate or near-zero-dimension geometry |
| 4 | `tolerance_error` | `tolerance.py` | Shapes that stress the kernel's tolerance limits |

---

## Sub-Families per Class

### Valid (label 0)
| Sub-Family | Parameters | Description |
|-----------|------------|-------------|
| `primitive_box` | length, width, height | Random axis-aligned box |
| `primitive_cylinder` | radius, height | Random cylinder |
| `primitive_sphere` | radius | Random sphere |
| `boolean_union` | l1, w1, h1, l2, w2, h2, ox, oy, oz | Fuse of two overlapping boxes |
| `filleted_box` | length, width, height, fillet_r | Box with safe edge fillets (5–35% of shortest edge) |

### Self-Intersection (label 1)
| Sub-Family | Description |
|-----------|-------------|
| `bowtie_extrude` | Bowtie polygon extruded to create self-crossing faces |
| `twisted_polygon` | Twisted polygon creating intersection artifacts |
| `multi_cross` | Multiple crossing geometry patterns |

### Non-Manifold (label 2)
| Sub-Family | Description |
|-----------|-------------|
| `face_sharing_compound` | Two solids sharing a common face |
| `edge_sharing_compound` | Two solids sharing a common edge |
| `vertex_sharing_compound` | Two solids sharing a single vertex |
| `open_shell` | Solid with faces removed, creating an open shell |
| `t_junction` | T-junction where a branch meets a main body |

### Degenerate (label 3)
| Sub-Family | Description |
|-----------|-------------|
| `zero_dim_box` | Box with one or more near-zero dimensions |
| `near_zero_extrude` | Extrusion with extremely small height |
| `extreme_aspect_ratio` | Box with extreme length-to-width ratio |
| `collinear_extrude` | Extrusion from collinear sketch edges |

### Tolerance (label 4)
| Sub-Family | Description |
|-----------|-------------|
| `sub_tolerance_box` | Box with dimensions below OCC default tolerance (1e-7) |
| `scale_mismatch_boolean` | Boolean of shapes at vastly different scales |
| `micro_fillet` | Fillet radius approaching tolerance limits |
| `near_coincident_boolean` | Boolean of nearly coincident shapes |

---

## Kernel Validation

After each shape is built, the pipeline runs `check_shape()` from `src/kernel_check.py`, which performs:

1. **BRepCheck_Analyzer** — The standard OpenCascade validity check
2. **Manifold checks** — Detects multi-solid compounds, missing solids, open shells, and extra shells
3. **Topology census** — Counts vertices, edges, wires, faces, shells, solids, compounds

A shape is considered valid only if it passes **both** BRepCheck and the manifold checks.

### Dual Labeling Strategy

The dataset records two labels per sample:

- **`intended_label`** — The failure class the generator was designed to produce
- **`label`** (kernel label) — If the kernel says the shape is valid and no construction error occurred, the kernel label is set to `valid` (0); otherwise it inherits the intended label

This dual-label approach lets the training pipeline use `intended_label` to learn the intended failure semantics, while `label` reflects actual kernel ground truth.

---

## Output Format

### `dataset.jsonl`

Each line is a JSON object with:

```json
{
  "label": 0,
  "intended_label": 0,
  "label_name": "valid",
  "intended_label_name": "valid",
  "failure_mode": "none",
  "sub_family": "primitive_box",
  "is_valid": true,
  "error_type": null,
  "error_msg": null,
  "param_family": "primitive_box",
  "param_length": 42.7,
  "param_width": 18.3,
  "param_height": 55.1,
  "topo_n_vertices": 8,
  "topo_n_edges": 12,
  "topo_n_faces": 6,
  "sample_id": "valid_0001"
}
```

### `dataset_stats.json`

Summary statistics per class: total samples, kernel-valid count, kernel-invalid count, and construction errors.

---

## Usage

```bash
# Default: 500 samples per class, seed 42
python scripts/generate_dataset.py

# Custom configuration
python scripts/generate_dataset.py --samples-per-class 1000 --seed 123

# Also export STEP files for CAD viewer inspection
python scripts/generate_dataset.py --export-step --step-dir data/step_files
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples-per-class` | 500 | Number of samples per failure class |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | `data/dataset.jsonl` | Output file path |
| `--export-step` | off | Also save each shape as a `.step` file |
| `--step-dir` | `data/step_files` | Directory for STEP file exports |

---

## Dataset Statistics (Current Build)

| Class | Total | Kernel Valid | Kernel Invalid | Construction Errors |
|-------|-------|-------------|----------------|-------------------|
| valid | 500 | 495 | 5 | 0 |
| self_intersection | 500 | 0 | 500 | 0 |
| non_manifold | 500 | 0 | 500 | 0 |
| degenerate_face | 500 | 500 | 0 | 0 |
| tolerance_error | 500 | 500 | 0 | 0 |
| **Total** | **2,500** | **1,495** | **1,005** | **0** |

> **Note:** Degenerate and tolerance shapes are kernel-valid (they build successfully in OCC) but represent problematic construction patterns. The model learns to identify them via their `intended_label`, not the kernel result.
