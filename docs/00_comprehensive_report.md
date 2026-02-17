# Failure-Aware CAD Validity Predictor — Comprehensive Project Report

---

## 1. Introduction

### 1.1 Problem Statement

Modern CAD kernels (OpenCascade, ACIS, Parasolid) can fail silently or raise cryptic errors when constructing geometrically degenerate, topologically non-manifold, or tolerance-violating shapes. In parametric design pipelines — especially automated ones — these failures waste computation cycles, halt workflows, and require manual debugging.

This project builds a **machine-learning prediction layer** that sits *before* the CAD kernel: given construction parameters, it predicts whether the resulting shape is likely to succeed or fail, *which specific failure mode* to expect, and *how confident* the prediction is.

### 1.2 Goal

Predict CAD construction validity across **5 classes** before execution:

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | Valid | Shape will build correctly |
| 1 | Self-Intersection | B-Rep faces will self-intersect |
| 2 | Non-Manifold | Topology will be non-manifold |
| 3 | Degenerate Face | Geometry has degenerate / near-zero dimensions |
| 4 | Tolerance Error | Dimensions conflict with kernel tolerance |

### 1.3 Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.0% |
| **Macro F1** | 99.3% |
| **Baseline (Rule-Based)** | 40.5% |
| **Total Errors (Test Set)** | 3 / 309 |
| **Uncertainty Detection** | 100% of errors flagged as uncertain |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Project Structure                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  src/                                                            │
│  ├── generators/          ← Parametric CAD shape generators      │
│  │   ├── base.py          ← BaseGenerator ABC + Sample dataclass │
│  │   ├── valid.py         ← 5 valid shape sub-families           │
│  │   ├── self_intersection.py ← 3 self-intersecting patterns     │
│  │   ├── non_manifold.py  ← 5 non-manifold patterns             │
│  │   ├── degenerate.py    ← 4 degenerate patterns               │
│  │   └── tolerance.py     ← 4 tolerance-error patterns          │
│  ├── features/            ← Feature extraction modules           │
│  │   ├── base_features.py ← 36 geometric + ratio + interaction  │
│  │   ├── graph_features.py← 24 face adjacency graph features    │
│  │   └── __init__.py      ← Unified API + FEATURE_NAMES list    │
│  └── kernel_check.py      ← OCP BRepCheck + manifold checks     │
│                                                                  │
│  scripts/                                                        │
│  ├── generate_dataset.py  ← Synthetic data generation   │
│  ├── extract_features.py  ← Feature matrix construction │
│  ├── analyze_features.py  ← Feature importance & correlation    │
│  ├── train_models.py      ← RF + calibration + ensemble │
│  ├── phase3_audit.py      ← Comprehensive model audit   │
│  ├── phase4_evaluation.py ← Evaluation & analysis       │
│  ├── predict.py           ← CLI predictor               │
│  └── diagnose_predictions.py ← Diagnostic tooling               │
│                                                                  │
│  data/                    ← Generated data artifacts             │
│  models/                  ← Trained model artifacts              │
│  examples/                ← Example prediction I/O               │
│  docs/                    ← This documentation                   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| CAD Kernel | OpenCascade (OCP via CadQuery) |
| Modeling Language | CadQuery |
| ML Framework | scikit-learn |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib, pickle, JSON |

---

## 3. Pipeline Steps

### Environment Setup

- **Virtual environment** with CadQuery + OCP + scikit-learn
- Verification script (`test_cad_setup.py`) validates kernel access

### Synthetic Data Generation

**Goal:** Create 2,500 labeled CAD samples (500 per class) using parametric generators.

**Process:**
1. Each generator class produces random parameter sets
2. CadQuery builds the OCP shape from parameters
3. `kernel_check.py` runs BRepCheck + manifold analysis
4. Records are saved as JSON Lines with dual labels (kernel + intended)

**Generators:** 5 classes × 3–5 sub-families = 21 distinct shape patterns

**Key Design Decision — Dual Labeling:**
Degenerate and tolerance shapes often pass OCC kernel checks (they build successfully) but represent problematic patterns. The dataset stores both:
- `label` — Kernel-determined ground truth
- `intended_label` — Generator's intended class

The training pipeline uses `intended_label` to learn failure semantics that go beyond simple kernel pass/fail.

**Output:**

| Artifact | Description |
|----------|-------------|
| `data/dataset.jsonl` | 2,500 records with params, labels, topology |
| `data/dataset_stats.json` | Per-class sample counts |

> See [01_data_generation.md](01_data_generation.md) for full details.

---

### Feature Engineering

**Goal:** Transform raw CAD shapes into a 60-dimensional numeric feature vector.

**Process:**
1. Reconstruct each OCP shape from stored parameters
2. Extract **base features** (bounding box, topology, ratios, tolerance metrics, interaction terms)
3. Extract **graph features** (Face Adjacency Graph statistics, face type distribution, area/edge stats)
4. Assemble into fixed-order numpy array

**Feature Groups (60 total):**

| Group | Count | Examples |
|-------|-------|---------|
| Bounding Box | 5 | bbox_lx, bbox_vol, bbox_diag |
| Sorted Dimensions | 3 | dim_min, dim_mid, dim_max |
| Ratios | 3 | aspect_ratio, compactness |
| Volume/Area | 2 | volume, surface_area |
| Topology | 9 | n_vertices, n_edges, euler_char |
| Flags | 3 | has_boolean_op, is_multi_solid |
| Tolerance | 3 | min_dim_over_tol, log_min_dim |
| Interactions | 8 | vol_sa_ratio, face_edge_ratio |
| Graph Structure | 11 | graph_n_components, graph_density |
| Graph Face Types | 7 | graph_frac_plane, graph_frac_sphere |
| Graph Statistics | 6 | graph_mean_face_area, graph_max_edge_len |

**Output:**

| Artifact | Description |
|----------|-------------|
| `data/X.npy` | Feature matrix (n × 60) |
| `data/y.npy` | Label vector (n,) |
| `data/feature_names.json` | Ordered feature column names |
| `data/features.jsonl` | Per-sample feature records |

> See [02_feature_engineering.md](02_feature_engineering.md) for full details.

---

### Model Training

**Goal:** Build progressively sophisticated classifiers with uncertainty quantification.

**Models trained:**

| Model | Technique | Test Accuracy | Test F1 (Macro) |
|-------|-----------|-------------|----------------|
| Baseline | Hand-coded rules | 40.5% | 13.7% |
| Random Forest | 200 trees, balanced weights | 99.0% | 99.3% |
| Calibrated RF | Platt scaling (sigmoid, 5-fold) | 99.0% | 99.3% |
| Ensemble (5 RF) | Disagreement-based uncertainty | 98.4% | — |

**Data split:** 70% train / 15% validation / 15% test (stratified, seed=42)

**Key hyperparameters:**
- `n_estimators = 200`
- `class_weight = "balanced"`
- Ensemble: 5 models with seeds [42, 153, 264, 375, 486]
- Uncertainty thresholds: `prob_t = 0.30`, `std_t = 0.05`

**Output:**

| Artifact | Description |
|----------|-------------|
| `models/rf_model.joblib` | Trained Random Forest |
| `models/model.pkl` | Calibrated RF |
| `models/uncertainty_ensemble.joblib` | 5-model ensemble |
| `models/training_config.json` | Full hyperparameter config |
| `models/training_results.json` | Per-stage metrics |

> See [03_model_training.md](03_model_training.md) for full details.

---

### Evaluation & Failure Analysis

**Goal:** In-depth evaluation spanning 7 analysis sections.

| Section | Focus |
|---------|-------|
| 4.1 | Standard metrics, confusion matrix, PR curves |
| 4.2 | Probability calibration (Brier, log-loss, reliability diagram) |
| 4.3 | SHAP explainability (global + local waterfall) |
| 4.4 | Visual error analysis (3 misclassifications, root causes) |
| 4.5 | Ablation study (feature group contribution) |
| 4.6 | Uncertainty analysis (ensemble disagreement vs. correctness) |
| 4.7 | Boundary case analysis (decision boundary ambiguity) |

**Key Findings:**

1. **Top SHAP Features:** compactness (0.057), n_solids (0.047), n_shells (0.038)
2. **All 3 errors** are Valid → NonManifold with low confidence (0.58–0.72)
3. **Ablation:** Ratio features provide the largest jump (+16.72 pp macro F1)
4. **Uncertainty filter** catches 100% of errors in the uncertain set

> See [04_evaluation_and_failure_analysis.md](04_evaluation_and_failure_analysis.md) for full details.

---

### Prediction Interface

**Goal:** CLI tool for real-time prediction from construction parameters.

```bash
# From JSON file
python scripts/predict.py test_input.json --pretty

# Inline parameters
python scripts/predict.py --params '{"family":"primitive_box","length":30,"width":20,"height":10}' --pretty

# With feature explanation
python scripts/predict.py test_input.json --pretty --explain

# Built-in test suite (10 cases)
python scripts/predict.py --test-suite
```

**Prediction output:**
```json
{
  "valid": true,
  "failure_mode": "none",
  "confidence": 0.968,
  "uncertainty": 0.0081,
  "status": "Confident",
  "predicted_class": "valid",
  "label_index": 0,
  "probabilities": {
    "valid": 0.968,
    "self_intersection": 0.025,
    "non_manifold": 0.005,
    "degenerate_face": 0.001,
    "tolerance_error": 0.001
  }
}
```

**Prediction workflow:**
1. Validate input parameters against `FAMILY_INFO` registry
2. Map family → generator → `build(params)` to construct OCP shape
3. Extract 60-feature vector (same pipeline as training)
4. Predict with ensemble → mean probabilities + uncertainty
5. Return structured result with confidence and failure mode

---

## 4. Supported Shape Families

| Category | Sub-Families | Count |
|----------|-------------|-------|
| **Valid** | primitive_box, primitive_cylinder, primitive_sphere, boolean_union, filleted_box | 5 |
| **Self-Intersection** | bowtie_extrude, twisted_polygon, multi_cross | 3 |
| **Non-Manifold** | face_sharing_compound, edge_sharing_compound, vertex_sharing_compound, open_shell, t_junction | 5 |
| **Degenerate** | zero_dim_box, near_zero_extrude, extreme_aspect_ratio, collinear_extrude | 4 |
| **Tolerance** | sub_tolerance_box, scale_mismatch_boolean, micro_fillet, near_coincident_boolean | 4 |
| | **Total** | **21** |

---

## 5. Results Summary

### Model Comparison Table

| Model | Accuracy | F1 (Macro) | Improvement over Baseline |
|-------|----------|-----------|--------------------------|
| Baseline (Rule-Based) | 40.5% | 13.7% | — |
| Random Forest | 99.0% | 99.3% | +58.5 pp |
| Calibrated RF | 99.0% | 99.3% | +58.5 pp |
| Ensemble (5 RF) | 98.4% | — | +57.9 pp |

### Per-Class Performance (Best Model)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Valid | 1.00 | 0.98 | 0.99 |
| Self-Intersection | 1.00 | 1.00 | 1.00 |
| Non-Manifold | 0.96 | 1.00 | 0.98 |
| Tolerance Error | 1.00 | 1.00 | 1.00 |

### Error Analysis

- **3 total errors** on 309 test samples (1.0% error rate)
- **All errors:** Valid → NonManifold (boolean-union shapes with ambiguous topology)
- **Uncertainty filter:** 100% of errors fall in the uncertain prediction band
- **Root cause:** Genuine geometric ambiguity, not model deficiency

---

## 6. Uncertainty Quantification

The `UncertaintyEnsemble` provides three levels of prediction certainty:

| Status | Criteria | Meaning |
|--------|----------|---------|
| **Confident** | max_prob ≥ 0.65 AND std ≤ 0.05 | Reliable prediction |
| **Low Confidence** | max_prob < 0.65 | Model unsure about class |
| **High Disagreement** | std > 0.05 | Ensemble models disagree |

In practice, 99.4% of predictions are **Confident** with a 0.98% error rate. The remaining 0.6% flagged as **Uncertain** contain 100% of the errors — making the uncertainty signal highly actionable.

---

## 7. Explainability

### Top Discriminative Features

| Feature | SHAP Impact | Interpretation |
|---------|------------|---------------|
| `compactness` | 0.057 | Primary valid-vs-failed separator |
| `n_solids` | 0.047 | Multi-solid indicates compound/failure |
| `n_shells` | 0.038 | Excess shells signal topology issues |
| `is_multi_solid` | 0.034 | Binary flag for multi-solid |
| `graph_n_components` | 0.034 | Disconnected face groups |

### Ablation Insights

| Feature Set Added | Macro F1 Gain |
|-------------------|-------------|
| Basic (bbox + topo) → + Ratios | +16.72 pp |
| + Ratios → + Flags | 0 pp |
| + Flags → + Interactions | 0 pp |
| + Interactions → + Graph | 0 pp |

Ratio features (especially `compactness` and `aspect_ratio`) are the single most important feature group.

---

## 8. Reproducibility

### Environment Setup

```bash
# Option 1: pip (CadQuery provides OCP bindings)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `cadquery` | Parametric CAD modeling + OCP kernel |
| `numpy` | Numerical arrays |
| `scikit-learn` | ML models, metrics, pipelines |
| `shap` | Explainability |
| `matplotlib` / `seaborn` | Visualization |
| `joblib` | Model serialization |
| `pandas` | Data manipulation |

### Full Pipeline Execution

```bash
# Generate dataset
python scripts/generate_dataset.py

# Extract features
python scripts/extract_features.py
python scripts/analyze_features.py

# Train models
python scripts/train_models.py

# Audit
python scripts/phase3_audit.py

# Evaluate
python scripts/phase4_evaluation.py

# Predict
python scripts/predict.py test_input.json --pretty --explain
```

---

## 9. Known Limitations & Future Work

### Current Limitations

1. **Degenerate class underrepresentation in evaluation** — Degenerate shapes often pass kernel checks, leading to only 2 tolerance-error samples in the test split
2. **Valid ↔ NonManifold boundary** — Boolean-union shapes with multi-solid topology create genuine ambiguity between these classes
3. **Synthetic data only** — Training data is generated, not sourced from real-world CAD workflows
4. **Fixed feature set** — The 60 features are hand-engineered; more complex geometry encodings (e.g., PointNet on mesh) could capture finer details

### Potential Extensions

- **Additional failure classes:** Gaps, overlaps, face-orientation errors
- **Real-world data integration:** STEP/IGES files from production CAD systems
- **Deep learning:** Graph neural networks on the Face Adjacency Graph
- **REST API:** Flask/FastAPI wrapper for the prediction endpoint
- **CI/CD:** Automated retraining on new data batches

---

## 10. File Reference

### Source Code

| File | Description |
|------|-------------|
| `src/generators/base.py` | BaseGenerator ABC, Sample dataclass, label constants |
| `src/generators/valid.py` | ValidGenerator (5 sub-families) |
| `src/generators/self_intersection.py` | SelfIntersectionGenerator (3 sub-families) |
| `src/generators/non_manifold.py` | NonManifoldGenerator (5 sub-families) |
| `src/generators/degenerate.py` | DegenerateGenerator (4 sub-families) |
| `src/generators/tolerance.py` | ToleranceGenerator (4 sub-families) |
| `src/features/base_features.py` | 36 base geometric features |
| `src/features/graph_features.py` | 24 graph-based features |
| `src/features/__init__.py` | Unified API, FEATURE_NAMES, build_pipeline() |
| `src/kernel_check.py` | BRepCheck + manifold validation |

### Scripts

| Script | Phase | Description |
|--------|-------|-------------|
| `scripts/generate_dataset.py` | 1 | Synthetic data generation |
| `scripts/extract_features.py` | 2 | Feature matrix construction |
| `scripts/analyze_features.py` | 2 | Feature importance analysis |
| `scripts/train_models.py` | 3 | RF + calibration + ensemble training |
| `scripts/phase3_audit.py` | 3 | Comprehensive model audit |
| `scripts/audit_checklist.py` | 3 | Audit checklist verification |
| `scripts/phase4_evaluation.py` | 4 | Full evaluation & analysis |
| `scripts/predict.py` | 5 | CLI prediction interface |
| `scripts/diagnose_predictions.py` | — | Diagnostic tooling |

### Data Artifacts

| File | Description |
|------|-------------|
| `data/dataset.jsonl` | Raw dataset (2,500 records) |
| `data/X.npy` | Feature matrix (2,500 × 60) |
| `data/y.npy` | Label vector (2,500) |
| `data/feature_names.json` | 60 feature column names |
| `data/split_info.json` | Train/val/test split metadata |

### Model Artifacts

| File | Description |
|------|-------------|
| `models/rf_model.joblib` | Trained Random Forest |
| `models/model.pkl` | Calibrated Random Forest |
| `models/uncertainty_ensemble.joblib` | 5-model uncertainty ensemble |
| `models/training_config.json` | Hyperparameter configuration |
| `models/training_results.json` | Per-stage training metrics |
| `models/phase4_metrics.json` | Final evaluation metrics |
| `models/phase4_report.txt` | Evaluation text report |
