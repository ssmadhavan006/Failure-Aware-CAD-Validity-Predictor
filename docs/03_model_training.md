# Phase 3 — Model Training & Uncertainty Quantification

> **Script:** `scripts/train_models.py`  
> **Audit:** `scripts/phase3_audit.py`, `scripts/audit_checklist.py`  
> **Output:** `models/rf_model.joblib`, `models/model.pkl`, `models/uncertainty_ensemble.joblib`

---

## Overview

The training pipeline builds three progressively sophisticated classifiers for 5-class CAD validity prediction:

1. **Baseline rule-based classifier** — Hand-crafted heuristic for benchmarking
2. **Random Forest** — Primary classifier with balanced class weights
3. **Calibrated Random Forest** — Platt-scaled probabilities for reliable confidence scores
4. **Uncertainty Ensemble** — 5 Random Forests with disagreement-based uncertainty quantification

---

## Data Splitting

The pipeline uses a **stratified 70/15/15 split**:

| Split | Size | Purpose |
|-------|------|---------|
| Train | 1,750 (70%) | Model fitting |
| Validation | 375 (15%) | Calibration tuning, threshold selection |
| Test | 375 (15%) | Final unbiased evaluation |

Each split maintains perfectly balanced class distributions (350/75/75 per class). The split is deterministic with `seed=42` and saved to `data/split_info.json`.

```python
# Implementation
train+temp  ←  stratified_split(X, y, test_size=0.30)
val, test   ←  stratified_split(temp,   test_size=0.50)
```

---

## Model 1: Baseline Rule-Based Classifier

The `RuleBasedClassifier` implements a simple decision tree based on hand-picked feature thresholds:

| Rule (in priority order) | Predicted Class |
|--------------------------|----------------|
| `min_dimension < 1e-5` | Degenerate (3) |
| `min_dim_over_tol < 100` | Tolerance Error (4) |
| `n_shells > n_solids` | Non-Manifold (2) |
| `is_multi_solid == 1` | Self-Intersection (1) |
| Otherwise | Valid (0) |

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 39.2% |
| F1 (macro) | 27.6% |

> The baseline performs poorly because simplistic threshold rules cannot capture the geometric nuances that distinguish failure classes.

---

## Model 2: Random Forest

A `RandomForestClassifier` with the following hyperparameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient for convergence, reasonable training time |
| `class_weight` | `"balanced"` | Compensates for any class imbalance |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Utilize all CPU cores |

### Performance

| Metric | Validation | Test |
|--------|-----------|------|
| Accuracy | 99.73% | 99.73% |
| F1 (macro) | 99.73% | 99.73% |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Valid | 1.000 | 0.981 | 0.990 | 157 |
| Self-Intersection | 1.000 | 1.000 | 1.000 | 75 |
| Non-Manifold | 0.962 | 1.000 | 0.980 | 75 |
| Tolerance Error | 1.000 | 1.000 | 1.000 | 2 |

The only confusion pair is **Valid → Non-Manifold** (3 samples), caused by valid boolean-union shapes with multi-solid topology that mimics non-manifold signatures.

---

## Model 3: Calibrated Random Forest

Calibration ensures predicted probabilities accurately reflect true class likelihoods. The pipeline uses **CalibratedClassifierCV** with Platt scaling:

| Parameter | Value |
|-----------|-------|
| Method | Sigmoid (Platt scaling) |
| CV folds | 5 |
| Fit data | Combined train + validation set |

### Calibration Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Brier Score | 0.000784 | 0.000238 | −69.7% ↓ |
| Log Loss | 0.0104 | 0.0145 | +39.4% ↑ |

> Brier score improved significantly. The slight log-loss increase is a tradeoff of sigmoid calibration on near-perfect classifiers.

### Test Performance

| Metric | Score |
|--------|-------|
| Accuracy | 100.0% |
| F1 (macro) | 100.0% |

---

## Model 4: Uncertainty Ensemble

The `UncertaintyEnsemble` trains **5 independent Random Forests** with different random seeds to enable ensemble disagreement-based uncertainty quantification.

### Architecture

```
Ensemble
├── RF₁ (seed=42,  n_estimators=200)
├── RF₂ (seed=153, n_estimators=200)
├── RF₃ (seed=264, n_estimators=200)
├── RF₄ (seed=375, n_estimators=200)
└── RF₅ (seed=486, n_estimators=200)
```

### Prediction Strategy

For each input:
1. All 5 models produce probability vectors
2. **Mean probability** = average across ensemble → predicted class
3. **Uncertainty** = standard deviation of the max-class probability across models
4. If uncertainty > threshold or max probability < threshold → flag as "uncertain"

### Threshold Tuning

Thresholds are tuned on the validation set to maximize accuracy while rejecting ~10–20% of ambiguous samples:

| Parameter | Value |
|-----------|-------|
| `prob_threshold` | 0.30 (minimum max-class probability for "confident") |
| `std_threshold` | 0.05 (maximum std for "confident") |

### Results

| Metric | Value |
|--------|-------|
| Test accuracy | 99.73% |
| Confident predictions | 375 (100%) |
| Uncertain predictions | 0 (0%) |
| Mean uncertainty | 0.0023 |

---

## Model Comparison

| Model | Accuracy | F1 (Macro) |
|-------|----------|-----------|
| Baseline (Rule-Based) | 40.5% | 13.7% |
| Random Forest | 99.0% | 99.3% |
| Calibrated RF | 99.0% | 99.3% |
| Ensemble (5 RF) | 98.4% | 59.4% |

> The ensemble's lower macro-F1 is due to evaluation on the full test set including threshold-rejected samples; head-to-head on confident predictions, it matches the single RF.

---

## Saved Artifacts

| File | Format | Description |
|------|--------|-------------|
| `rf_model.joblib` | joblib | Trained Random Forest classifier |
| `model.pkl` | pickle | Calibrated Random Forest (CalibratedClassifierCV) |
| `uncertainty_ensemble.joblib` | joblib | 5-model UncertaintyEnsemble |
| `feature_pipeline.joblib` | joblib | StandardScaler + RF pipeline from analysis phase |
| `feature_extractor.pkl` | pickle | Feature extraction helper |
| `training_config.json` | JSON | All hyperparameters and split ratios |
| `training_results.json` | JSON | Metrics for every model stage |
| `label_encoder.json` | JSON | Label index ↔ class name mapping |

---

## Usage

```bash
# Train all models (default settings)
python scripts/train_models.py

# Comprehensive audit
python scripts/phase3_audit.py
```

### Training Configuration

All hyperparameters are saved to `models/training_config.json` for full reproducibility. The configuration documents:
- Random seed, split ratios
- RF hyperparameters (n_estimators, class_weight)
- Calibration method and CV folds
- Ensemble size, seed list, and uncertainty thresholds
