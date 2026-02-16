"""
Phase 3 — Model Building & Uncertainty Quantification
======================================================
1. Stratified split: Train 70% / Val 15% / Test 15%
2. Baseline rule-based classifier (heuristic)
3. Random Forest with balanced class weights
4. Calibrate probabilities (CalibratedClassifierCV / Platt scaling)
5. Ensemble uncertainty via 5 Random Forests
6. Save models and preprocessors

Usage:
    python scripts/train_models.py [--data-dir data] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure UTF-8 output on Windows (box-drawing characters)
import io
import os

if os.name == "nt" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Label constants (mirror src/generators/base.py) ──────────────
LABEL_NAMES = {
    0: "valid",
    1: "self_intersection",
    2: "non_manifold",
    3: "degenerate_face",
    4: "tolerance_error",
}


# ===================================================================
# 3.1  Data Loading & Stratified Splitting
# ===================================================================


def load_and_split(data_dir: Path, seed: int = 42):
    """
    Load X.npy / y.npy and split into Train (70%), Val (15%), Test (15%).

    Returns
    -------
    dict with keys X_train, X_val, X_test, y_train, y_val, y_test,
         feature_names, split_info
    """
    from sklearn.model_selection import train_test_split

    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")

    with open(data_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Replace any lingering NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)

    print(f"Loaded X {X.shape}, y {y.shape}")
    print(
        f"Class distribution: { {int(c): int((y == c).sum()) for c in np.unique(y)} }"
    )

    # 70/30 split, then 50/50 on the 30% → 15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    split_info = {
        "seed": seed,
        "total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "classes": np.unique(y).tolist(),
        "train_class_dist": {int(c): int((y_train == c).sum()) for c in np.unique(y)},
        "val_class_dist": {int(c): int((y_val == c).sum()) for c in np.unique(y)},
        "test_class_dist": {int(c): int((y_test == c).sum()) for c in np.unique(y)},
    }

    print(
        f"\nSplit sizes: train={split_info['train_size']}, "
        f"val={split_info['val_size']}, test={split_info['test_size']}"
    )

    # ── Overlap verification ──────────────────────────────────────
    # Use hash-based row identity to check no sample appears in multiple splits
    train_hashes = {hash(X_train[i].tobytes()) for i in range(len(X_train))}
    val_hashes = {hash(X_val[i].tobytes()) for i in range(len(X_val))}
    test_hashes = {hash(X_test[i].tobytes()) for i in range(len(X_test))}
    tv_overlap = train_hashes & val_hashes
    tt_overlap = train_hashes & test_hashes
    vt_overlap = val_hashes & test_hashes
    if tv_overlap or tt_overlap or vt_overlap:
        print(
            f"  WARNING: Split overlap detected! train-val={len(tv_overlap)}, "
            f"train-test={len(tt_overlap)}, val-test={len(vt_overlap)}"
        )
    else:
        print("  Split overlap check: PASS (no sample in multiple splits)")

    # ── Class coverage check ─────────────────────────────────────
    all_classes = set(np.unique(y).tolist())
    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        present = set(np.unique(ys).tolist())
        missing = all_classes - present
        if missing:
            print(f"  WARNING: {name} split missing classes: {missing}")
        else:
            print(
                f"  Class coverage ({name}): PASS (all {len(present)} classes present)"
            )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "split_info": split_info,
    }


# ===================================================================
# 3.2  Baseline Rule-Based Classifier
# ===================================================================


class RuleBasedClassifier:
    """
    Simple heuristic classifier for CAD validity.

    Rules (applied in order):
      1. If min_dimension < 1e-5                                 → degenerate_face (3)
      2. If aspect_ratio > 10                                    → degenerate_face (3)
      3. If is_multi_solid == 1 AND graph_n_components > 1       → self_intersection (1)
      4. If is_multi_solid == 1                                  → non_manifold (2)
      5. If min_dim_over_tol < 100                               → tolerance_error (4)
      6. If compactness < 0.01 AND n_faces > 6                   → non_manifold (2)
      7. Otherwise                                               → valid (0)
    """

    def __init__(self, feature_names: list[str]):
        self._idx = {n: i for i, n in enumerate(feature_names)}

    def _get(self, row: np.ndarray, name: str) -> float:
        return row[self._idx[name]]

    def predict_one(self, row: np.ndarray) -> int:
        is_multi = self._get(row, "is_multi_solid")
        n_comp = self._get(row, "graph_n_components")
        dim_min = self._get(row, "dim_min")
        aspect = self._get(row, "aspect_ratio")
        compact = self._get(row, "compactness")
        n_faces = self._get(row, "n_faces")
        min_dim_tol = self._get(row, "min_dim_over_tol")

        # Rule 1: extremely small dimension → degenerate face
        if dim_min < 1e-5:
            return 3

        # Rule 2: high aspect ratio → degenerate face
        if aspect > 10:
            return 3

        # Rule 3: multi-solid + disconnected → self-intersection
        if is_multi > 0.5 and n_comp > 1.5:
            return 1

        # Rule 4: multi-solid → non-manifold
        if is_multi > 0.5:
            return 2

        # Rule 5: dimension close to tolerance → tolerance error
        if min_dim_tol < 100:
            return 4

        # Rule 6: very low compactness with many faces → non-manifold
        if compact < 0.01 and n_faces > 6:
            return 2

        # Default: valid
        return 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])])


# ===================================================================
# 3.3  Random Forest Training
# ===================================================================


def train_random_forest(X_train, y_train, seed=42):
    """Train a Random Forest with balanced class weights."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(
        f"Random Forest trained in {elapsed:.1f}s "
        f"({rf.n_estimators} trees, {X_train.shape[1]} features)"
    )
    return rf


# ===================================================================
# 3.4  Calibrate Probabilities (Platt Scaling)
# ===================================================================


def calibrate_model(rf, X_train, y_train, X_val, y_val):
    """
    Calibrate a Random Forest using CalibratedClassifierCV.

    Uses Platt scaling (sigmoid) with 5-fold cross-validation on the
    combined train+val data. This is the recommended approach for
    scikit-learn >= 1.8 where cv='prefit' is no longer supported.

    Returns
    -------
    CalibratedClassifierCV
        A calibrated wrapper that produces well-calibrated probabilities.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier

    # Build a fresh RF (CalibratedClassifierCV will clone + refit internally)
    base_rf = RandomForestClassifier(
        n_estimators=rf.n_estimators,
        class_weight=rf.class_weight,
        random_state=rf.random_state,
        n_jobs=-1,
    )

    t0 = time.time()
    # Combine train + val so calibration CV has enough data per fold
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    calibrated = CalibratedClassifierCV(
        estimator=base_rf,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_combined, y_combined)
    elapsed = time.time() - t0
    print(f"Calibrated model fitted in {elapsed:.1f}s (Platt scaling, 5-fold CV)")
    return calibrated


# ===================================================================
# 3.5  Uncertainty via Ensemble Disagreement
# ===================================================================


class UncertaintyEnsemble:
    """
    Ensemble of N Random Forests for uncertainty quantification.

    Trains N models with different random seeds. For each prediction:
      - mean probability vector = average across ensemble
      - uncertainty = standard deviation of max-class probability
      - If mean max prob < prob_threshold OR std > std_threshold → "uncertain"
    """

    def __init__(
        self,
        n_models: int = 5,
        base_seed: int = 42,
        n_estimators: int = 200,
    ):
        self.n_models = n_models
        self.base_seed = base_seed
        self.n_estimators = n_estimators
        self.models = []
        self.prob_threshold = 0.5  # tuned on validation set
        self.std_threshold = 0.15  # tuned on validation set

    def fit(self, X_train, y_train):
        """Train the ensemble of Random Forests."""
        from sklearn.ensemble import RandomForestClassifier

        t0 = time.time()
        self.models = []
        for i in range(self.n_models):
            seed = self.base_seed + i * 111  # spread seeds
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            self.models.append(rf)
        elapsed = time.time() - t0
        print(f"Uncertainty Ensemble: {self.n_models} RFs trained in {elapsed:.1f}s")
        return self

    def predict_proba_ensemble(self, X):
        """
        Return per-model probability arrays.

        Returns
        -------
        np.ndarray of shape (n_models, n_samples, n_classes)
        """
        return np.array([m.predict_proba(X) for m in self.models])

    def predict_proba_with_uncertainty(self, X):
        """
        Compute mean probability vector and uncertainty (std) for each sample.

        Returns
        -------
        mean_proba : np.ndarray (n_samples, n_classes)
        uncertainty : np.ndarray (n_samples,)
            Standard deviation of max-class probability across ensemble.
        """
        all_proba = self.predict_proba_ensemble(X)  # (N, S, C)
        mean_proba = all_proba.mean(axis=0)  # (S, C)

        # Std of the max-class probability across models
        max_proba_per_model = all_proba.max(axis=2)  # (N, S)
        uncertainty = max_proba_per_model.std(axis=0)  # (S,)

        return mean_proba, uncertainty

    def predict(self, X):
        """Standard predict using mean probabilities."""
        mean_proba, _ = self.predict_proba_with_uncertainty(X)
        return mean_proba.argmax(axis=1)

    def predict_robust(self, X):
        """
        Predict with uncertainty flag.

        Returns
        -------
        predictions : np.ndarray (n_samples,)
            Class label, or -1 for "uncertain".
        mean_proba : np.ndarray (n_samples, n_classes)
        uncertainty : np.ndarray (n_samples,)
        """
        mean_proba, uncertainty = self.predict_proba_with_uncertainty(X)
        max_prob = mean_proba.max(axis=1)
        predictions = mean_proba.argmax(axis=1)

        # Flag uncertain samples
        uncertain_mask = (max_prob < self.prob_threshold) | (
            uncertainty > self.std_threshold
        )
        predictions[uncertain_mask] = -1

        return predictions, mean_proba, uncertainty

    def tune_thresholds(self, X_val, y_val):
        """
        Tune uncertainty thresholds on the validation set.

        Strategy: Find thresholds that reject ~10-20% of samples while
        maximizing accuracy on the remaining (confident) predictions.
        """
        mean_proba, uncertainty = self.predict_proba_with_uncertainty(X_val)
        max_prob = mean_proba.max(axis=1)
        base_preds = mean_proba.argmax(axis=1)

        best_acc = 0.0
        best_pt = 0.5
        best_st = 0.15
        best_coverage = 0.0

        for pt in np.arange(0.3, 0.8, 0.05):
            for st in np.arange(0.05, 0.35, 0.025):
                confident = (max_prob >= pt) & (uncertainty <= st)
                coverage = confident.sum() / len(y_val)

                # We want at least 70% coverage
                if coverage < 0.70:
                    continue

                if confident.sum() == 0:
                    continue

                acc = (base_preds[confident] == y_val[confident]).mean()

                # Prefer higher accuracy, with coverage as tiebreaker
                if acc > best_acc or (acc == best_acc and coverage > best_coverage):
                    best_acc = acc
                    best_pt = pt
                    best_st = st
                    best_coverage = coverage

        self.prob_threshold = float(best_pt)
        self.std_threshold = float(best_st)

        print(
            f"  Tuned thresholds: prob_threshold={self.prob_threshold:.2f}, "
            f"std_threshold={self.std_threshold:.3f}"
        )
        print(f"  Confident accuracy={best_acc:.4f}, coverage={best_coverage:.2%}")
        return {
            "prob_threshold": self.prob_threshold,
            "std_threshold": self.std_threshold,
            "confident_accuracy": float(best_acc),
            "coverage": float(best_coverage),
        }


# ===================================================================
# Evaluation Utilities
# ===================================================================


def evaluate_model(name: str, model, X, y, label_names=None):
    """Print classification metrics for a model."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    present_labels = sorted(set(y.tolist()) | set(y_pred.tolist()))
    target_names = (
        [label_names.get(l, f"class_{l}") for l in present_labels]
        if label_names
        else None
    )

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-Macro:  {f1:.4f}")
    print(f"\n  Classification Report:")
    print(
        classification_report(
            y, y_pred, labels=present_labels, target_names=target_names, zero_division=0
        )
    )
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    # Header
    header = "          " + "  ".join(f"{l:>6}" for l in present_labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        print(f"  {present_labels[i]:>6}  {row_str}")

    return {"accuracy": acc, "f1_macro": f1}


def evaluate_calibration(name, model, X, y):
    """
    Evaluate calibration quality using Brier score and log-loss.

    Parameters
    ----------
    name : str
    model : fitted classifier with predict_proba
    X, y : test data

    Returns
    -------
    dict with brier_score and log_loss
    """
    from sklearn.metrics import brier_score_loss, log_loss

    proba = model.predict_proba(X)
    n_classes = proba.shape[1]

    # Multi-class Brier score (mean of per-class Brier scores)
    brier_scores = []
    for c in range(n_classes):
        y_binary = (y == c).astype(int)
        bs = brier_score_loss(y_binary, proba[:, c])
        brier_scores.append(bs)
    mean_brier = float(np.mean(brier_scores))

    ll = log_loss(y, proba)

    print(f"\n  {name}")
    print(f"    Brier Score (avg): {mean_brier:.4f}  (lower = better)")
    print(f"    Log Loss:          {ll:.4f}  (lower = better)")

    return {"brier_score": mean_brier, "log_loss": ll}


def evaluate_uncertainty(ensemble: UncertaintyEnsemble, X, y, label_names=None):
    """
    Evaluate the uncertainty ensemble with robust predictions.
    """
    preds, mean_proba, uncertainty = ensemble.predict_robust(X)

    n_total = len(y)
    uncertain_mask = preds == -1
    n_uncertain = uncertain_mask.sum()
    n_confident = n_total - n_uncertain

    print(f"\n{'=' * 60}")
    print(f"  Uncertainty Ensemble Analysis")
    print(f"{'=' * 60}")
    print(f"  Total samples:     {n_total}")
    print(f"  Confident:         {n_confident} ({n_confident / n_total:.1%})")
    print(f"  Uncertain:         {n_uncertain} ({n_uncertain / n_total:.1%})")

    if n_confident > 0:
        confident_acc = (preds[~uncertain_mask] == y[~uncertain_mask]).mean()
        print(f"  Confident accuracy: {confident_acc:.4f}")
    else:
        confident_acc = 0.0

    # Uncertainty statistics
    print(f"\n  Uncertainty Statistics:")
    print(f"    Mean:   {uncertainty.mean():.4f}")
    print(f"    Std:    {uncertainty.std():.4f}")
    print(f"    Min:    {uncertainty.min():.4f}")
    print(f"    Max:    {uncertainty.max():.4f}")
    print(f"    Median: {np.median(uncertainty):.4f}")

    # Per-class breakdown
    if label_names:
        print(f"\n  Per-class uncertainty (correct vs incorrect):")
        base_preds = mean_proba.argmax(axis=1)
        correct = base_preds == y
        for c in sorted(set(y.tolist())):
            mask_c = y == c
            if mask_c.sum() == 0:
                continue
            unc_correct = uncertainty[mask_c & correct]
            unc_wrong = uncertainty[mask_c & ~correct]
            name = label_names.get(c, f"class_{c}")
            parts = [f"    {name:>20s}: "]
            if len(unc_correct) > 0:
                parts.append(f"correct={unc_correct.mean():.4f}")
            if len(unc_wrong) > 0:
                parts.append(f"  wrong={unc_wrong.mean():.4f}")
            print("".join(parts))

    return {
        "n_total": n_total,
        "n_confident": int(n_confident),
        "n_uncertain": int(n_uncertain),
        "confident_accuracy": float(confident_acc),
        "mean_uncertainty": float(uncertainty.mean()),
        "prob_threshold": ensemble.prob_threshold,
        "std_threshold": ensemble.std_threshold,
    }


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Model Training")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   Phase 3 — Model Building & Uncertainty Quantification  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # ── 3.1 Load & Split ─────────────────────────────────────────
    data = load_and_split(data_dir, seed=args.seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    # Save split info
    with open(data_dir / "split_info.json", "w") as f:
        json.dump(data["split_info"], f, indent=2)
    print(f"Split info saved to {data_dir / 'split_info.json'}")

    # ── 3.2 Baseline Rule-Based ──────────────────────────────────
    print("\n── 3.2 Baseline Rule-Based Classifier ──────────────────")
    baseline = RuleBasedClassifier(feature_names)
    baseline_test = evaluate_model(
        "Baseline Rule-Based Classifier (Test Set)",
        baseline,
        X_test,
        y_test,
        LABEL_NAMES,
    )

    # ── 3.3 Random Forest ────────────────────────────────────────
    print("\n── 3.3 Random Forest Training ──────────────────────────")
    rf = train_random_forest(X_train, y_train, seed=args.seed)

    rf_val = evaluate_model(
        "Random Forest (Validation Set)",
        rf,
        X_val,
        y_val,
        LABEL_NAMES,
    )
    rf_test = evaluate_model(
        "Random Forest (Test Set)",
        rf,
        X_test,
        y_test,
        LABEL_NAMES,
    )

    # ── 3.4 Calibrate Probabilities ──────────────────────────────
    print("\n── 3.4 Probability Calibration (Platt Scaling) ────────")
    calibrated_rf = calibrate_model(rf, X_train, y_train, X_val, y_val)

    cal_test = evaluate_model(
        "Calibrated RF (Test Set)",
        calibrated_rf,
        X_test,
        y_test,
        LABEL_NAMES,
    )

    # Compare calibration quality
    print("\n  Calibration Comparison (Test Set):")
    cal_before = evaluate_calibration("  Uncalibrated RF", rf, X_test, y_test)
    cal_after = evaluate_calibration("  Calibrated RF", calibrated_rf, X_test, y_test)

    # ── 3.5 Uncertainty Ensemble ─────────────────────────────────
    print("\n── 3.5 Uncertainty Ensemble (5 Random Forests) ────────")
    ensemble = UncertaintyEnsemble(
        n_models=5,
        base_seed=args.seed,
        n_estimators=200,
    )
    ensemble.fit(X_train, y_train)

    # Tune thresholds on validation set
    print("\n  Tuning uncertainty thresholds on validation set...")
    threshold_info = ensemble.tune_thresholds(X_val, y_val)

    # Evaluate ensemble on test set
    ensemble_test = evaluate_model(
        "Uncertainty Ensemble — mean vote (Test Set)",
        ensemble,
        X_test,
        y_test,
        LABEL_NAMES,
    )

    # Detailed uncertainty analysis
    uncertainty_results = evaluate_uncertainty(ensemble, X_test, y_test, LABEL_NAMES)

    # ── 3.6 Save Models & Preprocessors ──────────────────────────
    print("\n── 3.6 Saving Models & Preprocessors ───────────────────")

    joblib.dump(rf, models_dir / "rf_model.joblib")
    print(f"  ✓ Raw RF         → {models_dir / 'rf_model.joblib'}")

    joblib.dump(calibrated_rf, models_dir / "model.pkl")
    print(f"  ✓ Calibrated RF  → {models_dir / 'model.pkl'}")

    joblib.dump(ensemble, models_dir / "uncertainty_ensemble.joblib")
    print(f"  ✓ Ensemble (5RF) → {models_dir / 'uncertainty_ensemble.joblib'}")

    # Save feature extractor info (feature names + ordering)
    feature_extractor_info = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }
    joblib.dump(feature_extractor_info, models_dir / "feature_extractor.pkl")
    print(f"  ✓ Feature info   → {models_dir / 'feature_extractor.pkl'}")

    # Save class mapping / label encoder
    present_classes = sorted(
        set(y_train.tolist()) | set(y_val.tolist()) | set(y_test.tolist())
    )
    class_mapping = {
        "label_to_name": {int(k): v for k, v in LABEL_NAMES.items()},
        "name_to_label": {v: int(k) for k, v in LABEL_NAMES.items()},
        "classes_in_dataset": [int(c) for c in present_classes],
        "class_order": [
            int(c)
            for c in (rf.classes_ if hasattr(rf, "classes_") else present_classes)
        ],
    }
    with open(models_dir / "label_encoder.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  ✓ Class mapping  → {models_dir / 'label_encoder.json'}")

    # Save feature importance
    importances = rf.feature_importances_
    fi = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    feature_importance_data = {
        "ranked": [{"feature": name, "importance": imp} for name, imp in fi],
        "top_20": [{"feature": name, "importance": imp} for name, imp in fi[:20]],
    }
    with open(models_dir / "feature_importance.json", "w") as f:
        json.dump(feature_importance_data, f, indent=2)
    print(f"  ✓ Feature import → {models_dir / 'feature_importance.json'}")
    print("\n  Top 10 features:")
    for rank, (fname, imp) in enumerate(fi[:10], 1):
        print(f"    {rank:2d}. {fname:30s} {imp:.4f}")

    # Save training config (hyperparameters + reproducibility)
    training_config = {
        "seed": args.seed,
        "random_forest": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": args.seed,
            "n_jobs": -1,
        },
        "calibration": {
            "method": "sigmoid",
            "cv": 5,
        },
        "ensemble": {
            "n_models": 5,
            "base_seed": args.seed,
            "seeds": [args.seed + i * 111 for i in range(5)],
            "n_estimators": 200,
            "prob_threshold": ensemble.prob_threshold,
            "std_threshold": ensemble.std_threshold,
        },
        "split": {
            "test_size_outer": 0.3,
            "test_size_inner": 0.5,
            "effective_ratios": "70/15/15",
        },
        "data": {
            "n_samples": int(data["split_info"]["total"]),
            "n_features": int(data["split_info"]["n_features"]),
            "classes": data["split_info"]["classes"],
        },
    }
    with open(models_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"  ✓ Config file    → {models_dir / 'training_config.json'}")

    # ── Summary ──────────────────────────────────────────────────
    results = {
        "baseline_test": baseline_test,
        "rf_val": rf_val,
        "rf_test": rf_test,
        "calibrated_rf_test": cal_test,
        "calibration_before": cal_before,
        "calibration_after": cal_after,
        "ensemble_test": ensemble_test,
        "uncertainty": uncertainty_results,
        "thresholds": threshold_info,
    }
    with open(models_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 65)
    print("  Summary: Test Set Comparison")
    print("=" * 65)
    print(f"  {'Model':<35} {'Accuracy':>10} {'F1-Macro':>10}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10}")
    print(
        f"  {'Baseline (Rule-Based)':<35} "
        f"{baseline_test['accuracy']:10.4f} {baseline_test['f1_macro']:10.4f}"
    )
    print(
        f"  {'Random Forest (balanced)':<35} "
        f"{rf_test['accuracy']:10.4f} {rf_test['f1_macro']:10.4f}"
    )
    print(
        f"  {'Calibrated RF (Platt)':<35} "
        f"{cal_test['accuracy']:10.4f} {cal_test['f1_macro']:10.4f}"
    )
    print(
        f"  {'Ensemble (5 RF, mean vote)':<35} "
        f"{ensemble_test['accuracy']:10.4f} {ensemble_test['f1_macro']:10.4f}"
    )

    improvement_acc = rf_test["accuracy"] - baseline_test["accuracy"]
    improvement_f1 = rf_test["f1_macro"] - baseline_test["f1_macro"]
    print(
        f"\n  RF improvement over baseline: "
        f"Accuracy +{improvement_acc:.4f}, F1-Macro +{improvement_f1:.4f}"
    )

    print(
        f"\n  Calibration: Brier {cal_before['brier_score']:.4f} → "
        f"{cal_after['brier_score']:.4f}  |  "
        f"Log-Loss {cal_before['log_loss']:.4f} → {cal_after['log_loss']:.4f}"
    )

    print(
        f"\n  Uncertainty: {uncertainty_results['n_uncertain']} uncertain "
        f"({uncertainty_results['n_uncertain'] / uncertainty_results['n_total']:.1%}), "
        f"confident accuracy={uncertainty_results['confident_accuracy']:.4f}"
    )

    print(f"\n  Models saved to: {models_dir}")
    print()


if __name__ == "__main__":
    main()
