"""Feature importance analysis, correlation, and pipeline construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_features(data_dir: str = "data") -> None:
    """Run feature importance and quality analysis."""

    ddir = Path(PROJECT_ROOT) / data_dir
    models_dir = Path(PROJECT_ROOT) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(ddir / "X.npy")
    y = np.load(ddir / "y.npy")

    with open(ddir / "feature_names.json", "r") as f:
        names = json.load(f)

    print(f"X.shape = {X.shape},  y.shape = {y.shape}")
    print(f"Classes: {np.unique(y).tolist()}")
    print(f"Features: {len(names)}")

    # ----------------------------------------------------------------
    # 1. Data quality checks
    # ----------------------------------------------------------------
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print(f"\nNaN count:  {n_nan}")
    print(f"Inf count:  {n_inf}")

    # Replace inf with large finite values for analysis
    X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)

    # Low-variance features (variance < 1e-10)
    variances = X.var(axis=0)
    low_var = [
        (names[i], variances[i]) for i in range(len(names)) if variances[i] < 1e-10
    ]
    if low_var:
        print(f"\nLow-variance features ({len(low_var)}):")
        for name, v in low_var:
            print(f"  {name}: var={v:.2e}")
    else:
        print("\nNo low-variance features found.")

    # ----------------------------------------------------------------
    # 2. Feature importance via Random Forest
    # ----------------------------------------------------------------
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        print(f"\n{'=' * 60}")
        print("Feature Importance (Random Forest, Gini)")
        print(f"{'=' * 60}")
        print(f"{'Rank':>4}  {'Feature':<35}  {'Importance':>10}")
        print(f"{'-' * 4}  {'-' * 35}  {'-' * 10}")
        for rank, idx in enumerate(sorted_idx, 1):
            print(f"{rank:4d}  {names[idx]:<35}  {importances[idx]:10.4f}")

        # Quick accuracy estimate
        scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
        print(f"\n5-fold CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Save importances
        importance_data = {names[i]: float(importances[i]) for i in sorted_idx}
        with open(ddir / "feature_importance.json", "w") as f:
            json.dump(importance_data, f, indent=2)
        print(f"\nImportances saved to {ddir / 'feature_importance.json'}")

        # Prune zero-importance features
        selected = [names[i] for i in range(len(names)) if importances[i] > 0.0]
        pruned = [names[i] for i in range(len(names)) if importances[i] == 0.0]

        print(f"\n{'=' * 60}")
        print("Feature Selection (Zero-Importance Pruning)")
        print(f"{'=' * 60}")
        print(f"  Retained:  {len(selected)} features")
        print(f"  Pruned:    {len(pruned)} features")
        if pruned:
            for p in pruned:
                print(f"    [drop] {p}")

        with open(ddir / "selected_features.json", "w") as f:
            json.dump(selected, f, indent=2)
        print(f"\nSelected features saved to {ddir / 'selected_features.json'}")

        # Build and save reusable pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)

        pipeline_path = models_dir / "feature_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        print(f"Pipeline saved to {pipeline_path}")

        # Verify pipeline accuracy matches
        pipe_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        print(
            f"Pipeline 5-fold CV Accuracy: "
            f"{pipe_scores.mean():.3f} +/- {pipe_scores.std():.3f}"
        )

    except ImportError:
        print("\nsklearn not installed -- skipping feature importance analysis.")
        print("Install with: pip install scikit-learn")

    # ----------------------------------------------------------------
    # 3. Correlation analysis (flag highly correlated pairs)
    # ----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("High Correlation Pairs (|r| > 0.95)")
    print(f"{'=' * 60}")

    # Only compute for finite-variance features
    valid_cols = [i for i in range(X.shape[1]) if variances[i] > 1e-10]
    X_valid = X[:, valid_cols]
    valid_names = [names[i] for i in valid_cols]

    if len(valid_cols) > 1:
        corr = np.corrcoef(X_valid, rowvar=False)
        high_corr = []
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                r = abs(corr[i, j])
                if r > 0.95:
                    high_corr.append((valid_names[i], valid_names[j], corr[i, j]))

        if high_corr:
            for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2])):
                print(f"  {a} <-> {b}  r={r:.3f}")
        else:
            print("  None found.")
    else:
        print("  Not enough valid features to compute correlations.")


def main():
    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    analyze_features(args.data_dir)


if __name__ == "__main__":
    main()
