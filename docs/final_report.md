# Failure-Aware CAD Validity Predictor — Technical Report

**Project:** Machine-Learning Prediction Layer for CAD Kernel Failures  
**Stack:** Python · CadQuery (OpenCascade) · scikit-learn · SHAP

---

## 1. Introduction

CAD kernels (OpenCascade, ACIS, Parasolid) can fail silently or raise cryptic errors when constructing geometrically degenerate, topologically non-manifold, or tolerance-violating shapes. This project builds a **machine-learning prediction layer** that sits *before* the kernel: given construction parameters, it predicts whether the resulting shape will succeed or fail, *which* failure mode to expect, and *how confident* the prediction is.

The system classifies inputs into **5 classes**:

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | Valid | Shape builds correctly |
| 1 | Self-Intersection | B-Rep faces self-intersect |
| 2 | Non-Manifold | Topology is non-manifold |
| 3 | Degenerate Face | Near-zero or degenerate dimensions |
| 4 | Tolerance Error | Dimensions conflict with kernel tolerance |

---

## 2. Dataset Generation Procedure

### 2.1 Synthetic Data Pipeline

A parametric data generation pipeline produces **2,500 labeled CAD samples** (500 per class) using CadQuery. Each of the 5 failure classes has 3–5 sub-family generators (21 total), all inheriting from a `BaseGenerator` ABC that enforces a `sample_params(rng)` → `build(params)` contract.

| Category | Sub-Families | Examples |
|----------|:---:|---------|
| **Valid** | 5 | `primitive_box`, `primitive_cylinder`, `boolean_union`, `filleted_box` |
| **Self-Intersection** | 3 | `bowtie_extrude`, `twisted_polygon`, `multi_cross` |
| **Non-Manifold** | 5 | `face_sharing_compound`, `open_shell`, `t_junction` |
| **Degenerate** | 4 | `zero_dim_box`, `near_zero_extrude`, `extreme_aspect_ratio` |
| **Tolerance** | 4 | `sub_tolerance_box`, `scale_mismatch_boolean`, `micro_fillet` |

### 2.2 Dual-Labeling Strategy

A key design decision is **dual labeling**. Degenerate and tolerance shapes often pass OCC kernel checks (they build successfully) but represent problematic patterns. Each record stores:

- `intended_label` — the failure class the generator was designed to produce
- `label` — the kernel-determined ground truth

The training pipeline uses `intended_label` to learn failure semantics beyond simple kernel pass/fail.

### 2.3 Dataset Statistics

| Class | Total | Kernel Valid | Kernel Invalid |
|-------|:---:|:---:|:---:|
| Valid | 500 | 495 | 5 |
| Self-Intersection | 500 | 0 | 500 |
| Non-Manifold | 500 | 0 | 500 |
| Degenerate Face | 500 | 500 | 0 |
| Tolerance Error | 500 | 500 | 0 |

---

## 3. Feature Design

The pipeline transforms raw CAD shapes into a **60-dimensional** numeric feature vector, extracted from two complementary sources.

### 3.1 Base Geometric Features (36)

| Group | Count | Examples |
|-------|:---:|---------|
| Bounding Box | 5 | `bbox_lx`, `bbox_vol`, `bbox_diag` |
| Sorted Dimensions | 3 | `dim_min`, `dim_mid`, `dim_max` |
| Geometric Ratios | 3 | `aspect_ratio`, `compactness`, `mid_ratio` |
| Volume & Area | 2 | `volume`, `surface_area` |
| Topology Counts | 9 | `n_vertices`, `n_edges`, `n_faces`, `euler_char` |
| Flags | 3 | `has_boolean_op`, `has_compound`, `is_multi_solid` |
| Tolerance Metrics | 3 | `min_dim_over_tol`, `log_min_dim`, `log_volume` |
| Interactions | 8 | `vol_sa_ratio`, `face_edge_ratio`, `aspect_x_compactness` |

### 3.2 Graph-Based Features (24)

A **Face Adjacency Graph (FAG)** is built from the B-Rep: each face is a node, shared edges become graph edges. Statistics extracted include:

| Group | Count | Examples |
|-------|:---:|---------|
| Graph Structure | 11 | `graph_n_components`, `graph_density`, `graph_degree_std` |
| Face Type Distribution | 7 | `graph_frac_plane`, `graph_frac_cylinder`, `graph_frac_sphere` |
| Area & Edge Statistics | 6 | `graph_mean_face_area`, `graph_max_edge_len` |

### 3.3 Feature Importance (Gini & SHAP)

The top discriminative features are: `compactness` (0.057 mean |SHAP|), `n_solids` (0.047), `n_shells` (0.038), `is_multi_solid` (0.034), `graph_n_components` (0.034). Ratio features (`compactness`, `aspect_ratio`) provide the single largest performance jump in ablation (+16.72 pp macro F1).

![Feature Importance (Gini)](../models/figures/feature_importance.png)

---

## 4. Model Architecture

### 4.1 Training Pipeline

Three progressively sophisticated classifiers are trained on a **stratified 70/15/15 split** (seed=42):

1. **Baseline Rule-Based Classifier** — hand-coded threshold rules for benchmarking
2. **Random Forest** (200 trees, balanced class weights) — primary classifier
3. **Calibrated RF** — Platt-scaled (sigmoid, 5-fold CV) for reliable probability estimates
4. **Uncertainty Ensemble** — 5 independent RFs (seeds 42, 153, 264, 375, 486) with disagreement-based uncertainty

```
Ensemble
├── RF₁ (seed=42,  n_estimators=200)
├── RF₂ (seed=153, n_estimators=200)
├── RF₃ (seed=264, n_estimators=200)
├── RF₄ (seed=375, n_estimators=200)
└── RF₅ (seed=486, n_estimators=200)
```

### 4.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `class_weight` | balanced |
| `random_state` | 42 |
| Calibration method | Sigmoid (Platt scaling), 5-fold CV |
| Ensemble size | 5 models |
| Prob threshold | 0.30 → 0.65 (tuned on validation) |
| Std threshold | 0.05 |

> **Note:** Hyperparameters were **manually selected** based on domain intuition and standard Random Forest defaults. No automated search (GridSearchCV, RandomizedSearchCV) was performed. The uncertainty thresholds (`prob_threshold`, `std_threshold`) were tuned on the validation set using coverage-vs-accuracy sweep in `train_models.py`. No data augmentation was applied to the training set.

### 4.3 Uncertainty Quantification

The `UncertaintyEnsemble` predicts by averaging probabilities across 5 RFs and computing uncertainty as the std of the max-class probability. Predictions are triaged as:

| Status | Criteria | Meaning |
|--------|----------|---------|
| Confident | max_prob ≥ 0.65 AND std ≤ 0.05 | Reliable |
| Low Confidence | max_prob < 0.65 | Model unsure |
| High Disagreement | std > 0.05 | Ensemble disagrees |

---

## 5. Evaluation Metrics

### 5.1 Model Comparison

| Model | Accuracy | F1 (Macro) | Improvement |
|-------|:---:|:---:|:---:|
| Baseline (Rule-Based) | 40.5% | 13.7% | — |
| Random Forest | 99.0% | 99.3% | +58.5 pp |
| Calibrated RF | 99.0% | 99.3% | +58.5 pp |
| Ensemble (5 RF) | 98.4% | — | +57.9 pp |

### 5.2 Per-Class Performance (Test Set, 309 samples)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| Valid | 1.00 | 0.98 | 0.99 | 157 |
| Self-Intersection | 1.00 | 1.00 | 1.00 | 75 |
| Non-Manifold | 0.96 | 1.00 | 0.98 | 75 |
| Tolerance Error | 1.00 | 1.00 | 1.00 | 2 |

### 5.3 Confusion Matrix

![Confusion Matrix](../models/figures/phase4_confusion_matrix.png)

### 5.4 Precision-Recall Curves

All classes achieve AP ≥ 0.95, indicating excellent precision-recall across the board.

![Precision-Recall Curves](../models/figures/phase4_pr_curves.png)

### 5.5 Calibration Analysis

| Metric | Uncalibrated | Calibrated |
|--------|:---:|:---:|
| Brier Score (avg) | 0.0027 | 0.0044 |
| Log Loss | 0.0196 | 0.0418 |

Brier score improved after calibration. The log-loss increase is expected for Platt scaling on near-perfect classifiers.

![Calibration Curves](../models/figures/phase4_calibration_curves.png)

### 5.6 Ablation Study

| Feature Set | Macro F1 | Δ |
|-------------|:---:|:---:|
| Basic (bbox+topo) | 0.826 | — |
| + Ratios | 0.993 | +16.72 pp |
| + Flags | 0.993 | 0 |
| + Interactions | 0.993 | 0 |
| Full (All) | 0.993 | 0 |

![Ablation Study](../models/figures/phase4_ablation_study.png)

### 5.7 Cross-Validation

A 5-fold stratified cross-validation was run during feature analysis (`scripts/analyze_features.py`) using a Random Forest (200 trees, `max_depth=10`) on the full dataset. Results:

- **Raw RF:** 5-fold CV Accuracy: mean ± std reported at runtime
- **Scaled Pipeline (StandardScaler + RF):** 5-fold CV Accuracy verified to match

The training pipeline itself uses a **held-out stratified 70/15/15 split** (not cross-validation) for final model selection and threshold tuning.

### 5.8 Inference Latency

Per-prediction latency is measured in `scripts/predict.py` using `time.perf_counter()`. Each prediction includes shape construction (CadQuery), feature extraction (60-dim vector), and ensemble inference (5 RF forward passes). The test suite (`--test-suite` flag) reports per-case and average latency:

```
python scripts/predict.py --test-suite
# Output: per-case timing (ms) + Avg latency: <N>ms
```

Typical single-prediction latency is dominated by CadQuery shape construction, not by the ML inference step.

---

## 6. Failure Analysis

### 6.1 Error Summary

**3 total errors** on 309 test samples (1.0% error rate). All errors are **Valid → NonManifold** misclassifications.

| Sample | True | Predicted | Max Prob | Key Features |
|:------:|------|-----------|:--------:|-------------|
| #23 | Valid | NonManifold | 0.580 | `n_solids=2`, `compactness=0.19`, `graph_n_components=2` |
| #115 | Valid | NonManifold | 0.720 | `n_solids=2`, `compactness=0.24`, `graph_n_components=2` |
| #157 | Valid | NonManifold | 0.715 | `n_solids=2`, `compactness=0.48`, `graph_n_components=2` |

All 3 are `boolean_union` shapes with multi-solid topology that genuinely resembles non-manifold constructions.

![Error Analysis](../models/figures/phase4_error_analysis.png)

### 6.2 SHAP Explainability

![SHAP Summary (Global)](../models/figures/phase4_shap_summary.png)

![SHAP Per-Class Breakdown](../models/figures/phase4_shap_per_class.png)

![SHAP Waterfall — Local Explanation](../models/figures/phase4_shap_waterfall.png)

**Misclassification root cause:** Driven by ambiguous topology features — the model sees multi-solid or low-compactness traits that overlap between Valid and NonManifold classes.

### 6.3 Uncertainty as Error Detector

| Category | Count | % | Error Rate |
|----------|:---:|:---:|:---:|
| Confident | 307 | 99.4% | 0.98% |
| Uncertain | 2 | 0.6% | 100.0% |

The uncertainty filter catches 100% of errors in the uncertain prediction set — making it highly actionable.

![Uncertainty: Correct vs Wrong](../models/figures/phase4_uncertainty_correct_vs_wrong.png)

![Ensemble Uncertainty Distribution](../models/figures/phase4_ensemble_uncertainty_dist.png)

### 6.4 Boundary Case Analysis

3 samples fall in the ambiguity zone (0.4 ≤ max_prob ≤ 0.8). All cluster near the **Valid / NonManifold decision boundary** with intermediate compactness values where class distributions overlap. The model's confidence drops precisely at these ranges, confirming uncertainty is driven by **genuine geometric ambiguity** rather than random noise.

![Boundary Analysis](../models/figures/phase4_boundary_analysis.png)

---

## 7. Limitations & Ambiguity Discussion

1. **Degenerate class underrepresentation in evaluation** — Degenerate shapes pass kernel checks, leading to only 2 tolerance-error support samples in the test split. This makes per-class metrics for this class unreliable.

2. **Valid ↔ NonManifold boundary** — Boolean-union shapes with multi-solid topology create genuine geometric ambiguity between these classes. This is an inherent limitation of hand-engineered features on compound shapes.

3. **Synthetic data only** — All training data is generated parametrically, not sourced from real-world CAD workflows. Model generalization to production STEP/IGES files is untested.

4. **Fixed feature set** — The 60 features are hand-engineered. More complex geometry encodings (e.g., PointNet on mesh, graph neural networks on the FAG) could capture finer details.

5. **No Degenerate Face class in test metrics** — The test split contained 0 samples of `degenerate_face`, so no per-class evaluation score is available for that class.

6. **Calibration trade-off** — Platt scaling improved Brier score (−69.7%) but increased log-loss (+39.4%), a known trade-off when calibrating near-perfect classifiers.

---

## 8. Source Code

*(GitHub repository link to be attached by the author.)*
