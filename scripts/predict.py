"""CAD validity predictor CLI."""

from __future__ import annotations

import argparse
import json
import sys
import os
import io
import time
from pathlib import Path

import numpy as np

# UTF-8 on Windows
if os.name == "nt" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Required for model deserialization
from train_models import UncertaintyEnsemble, RuleBasedClassifier  # noqa: E402
import __main__

__main__.UncertaintyEnsemble = UncertaintyEnsemble
__main__.RuleBasedClassifier = RuleBasedClassifier

LABEL_NAMES = {
    0: "valid",
    1: "self_intersection",
    2: "non_manifold",
    3: "degenerate_face",
    4: "tolerance_error",
}

# Supported families (must match actual generator sub-family names)
FAMILY_INFO = {
    # Valid generators
    "primitive_box": {"generator": "valid", "required": ["length", "width", "height"]},
    "primitive_cylinder": {"generator": "valid", "required": ["radius", "height"]},
    "primitive_sphere": {"generator": "valid", "required": ["radius"]},
    "boolean_union": {
        "generator": "valid",
        "required": ["l1", "w1", "h1", "l2", "w2", "h2", "ox", "oy", "oz"],
    },
    "filleted_box": {
        "generator": "valid",
        "required": ["length", "width", "height", "fillet_r"],
    },
    # Self-intersection generators
    "bowtie_extrude": {
        "generator": "self_intersection",
        "required": ["w", "h", "extrude", "cx", "cy"],
    },
    "twisted_polygon": {
        "generator": "self_intersection",
        "required": ["n_points", "radius", "extrude", "twist_seed"],
    },
    "multi_cross": {
        "generator": "self_intersection",
        "required": ["n_segments", "amplitude", "wavelength", "extrude"],
    },
    # Non-manifold generators
    "face_sharing_compound": {"generator": "non_manifold", "required": ["s1", "s2"]},
    "edge_sharing_compound": {"generator": "non_manifold", "required": ["s1", "s2"]},
    "vertex_sharing_compound": {"generator": "non_manifold", "required": ["s1", "s2"]},
    "open_shell": {
        "generator": "non_manifold",
        "required": ["length", "width", "height", "n_faces_keep"],
    },
    "t_junction": {
        "generator": "non_manifold",
        "required": ["main_size", "branch_w", "branch_h", "branch_d"],
    },
    # Degenerate generators
    "zero_dim_box": {
        "generator": "degenerate",
        "required": ["length", "width", "height"],
    },
    "near_zero_extrude": {
        "generator": "degenerate",
        "required": ["base_l", "base_w", "height"],
    },
    "extreme_aspect_ratio": {
        "generator": "degenerate",
        "required": ["length", "width", "height"],
    },
    "collinear_extrude": {
        "generator": "degenerate",
        "required": ["length", "extrude_h", "deviation"],
    },
    # Tolerance generators
    "sub_tolerance_box": {
        "generator": "tolerance",
        "required": ["length", "width", "height"],
    },
    "scale_mismatch_boolean": {
        "generator": "tolerance",
        "required": ["large_size", "tiny_size", "offset_x", "offset_y", "offset_z"],
    },
    "micro_fillet": {"generator": "tolerance", "required": ["box_size", "fillet_r"]},
    "near_coincident_boolean": {
        "generator": "tolerance",
        "required": ["size", "offset"],
    },
}


def load_models(models_dir: Path | None = None):
    """Load trained models and feature names with clear error messages."""
    import joblib

    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"

    # RF model
    rf_path = models_dir / "rf_model.joblib"
    if not rf_path.exists():
        raise FileNotFoundError(
            f"RF model not found at {rf_path}. Run train_models.py first."
        )
    rf = joblib.load(rf_path)

    # Ensemble (optional but recommended)
    ensemble = None
    ens_path = models_dir / "uncertainty_ensemble.joblib"
    if ens_path.exists():
        ensemble = joblib.load(ens_path)
    else:
        print(
            f"  [WARN] Ensemble not found at {ens_path}; uncertainty will be 0.",
            file=sys.stderr,
        )

    # Feature names
    fn_path = PROJECT_ROOT / "data" / "feature_names.json"
    if not fn_path.exists():
        raise FileNotFoundError(
            f"Feature names not found at {fn_path}. Run extract_features.py first."
        )
    with open(fn_path) as f:
        feature_names = json.load(f)

    return rf, ensemble, feature_names


def validate_input(params: dict) -> list[str]:
    """Validate input params. Returns list of error strings (empty = valid)."""
    errors = []

    if "family" not in params:
        errors.append("Missing required key 'family'.")
        return errors

    family = params["family"]
    if family not in FAMILY_INFO:
        errors.append(
            f"Unknown family '{family}'. Supported: {list(FAMILY_INFO.keys())}"
        )
        return errors

    info = FAMILY_INFO[family]
    for key in info["required"]:
        if key not in params:
            errors.append(f"Missing required parameter '{key}' for family '{family}'.")
        else:
            val = params[key]
            if not isinstance(val, (int, float)):
                errors.append(
                    f"Parameter '{key}' must be numeric, got {type(val).__name__}."
                )
            elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                errors.append(f"Parameter '{key}' must be finite, got {val}.")

    # Range validation — only for valid-generator families
    # (degenerate/tolerance families intentionally use zero/near-zero dims)
    if info["generator"] == "valid":
        for key in [
            "length",
            "width",
            "height",
            "radius",
            "l1",
            "w1",
            "h1",
            "l2",
            "w2",
            "h2",
        ]:
            if key in params and isinstance(params[key], (int, float)):
                if params[key] <= 0:
                    errors.append(
                        f"Parameter '{key}' must be positive, got {params[key]}."
                    )

    return errors


def build_shape(params: dict):
    """Build a CAD shape from parameter dict using the generators."""
    from src.generators import ALL_GENERATORS

    family = params["family"]
    info = FAMILY_INFO[family]
    gen_name = info["generator"]

    gen_map = {g.name: g for g in [G() for G in ALL_GENERATORS]}
    if gen_name not in gen_map:
        raise ValueError(
            f"Generator '{gen_name}' not found. Available: {list(gen_map.keys())}"
        )
    gen = gen_map[gen_name]

    # Pass ALL params INCLUDING 'family' — build() reads params["family"]
    sample = gen.build(params)
    return sample


def extract_feature_vector(shape, params: dict, feature_names: list[str]):
    """Extract features and return ordered numpy array matching training order."""
    from src.features import extract_all_features

    if shape is None:
        return np.zeros(len(feature_names), dtype=np.float64)

    # Build a record dict that mirrors what the training pipeline provides.
    # Critical: include "sub_family" so that has_boolean_op and other
    # metadata-derived features are computed identically to training.
    record = {f"param_{k}": v for k, v in params.items()}
    family = params.get("family", "")
    record["sub_family"] = family  # training pipeline stores this field

    feat_dict = extract_all_features(shape, record)

    arr = np.array(
        [feat_dict.get(name, 0.0) for name in feature_names],
        dtype=np.float64,
    )
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e12, neginf=-1e12)

    # Assert length matches
    assert len(arr) == len(feature_names), (
        f"Feature vector length {len(arr)} != expected {len(feature_names)}"
    )
    return arr


def predict(params: dict, rf=None, ensemble=None, feature_names=None) -> dict:
    """
    Predict CAD validity from construction parameters.

    Returns dict with: valid, failure_mode, confidence, uncertainty, status,
                       predicted_class, label_index, probabilities
    """
    if rf is None:
        rf, ensemble, feature_names = load_models()

    # 1. Validate input
    errors = validate_input(params)
    if errors:
        return {"error": True, "messages": errors}

    # 2. Build shape
    try:
        sample = build_shape(params)
    except Exception as e:
        return {"error": True, "messages": [f"Shape build failed: {e}"]}

    # 3. Extract features (same pipeline as training)
    X = extract_feature_vector(sample.shape, params, feature_names).reshape(1, -1)

    # 4. Get probabilities
    proba = rf.predict_proba(X)[0]
    pred_class = int(proba.argmax())
    max_prob = float(proba.max())

    # 5. Ensemble uncertainty (uses tuned thresholds from training)
    uncertainty = 0.0
    prob_threshold = 0.6  # default fallback
    std_threshold = 0.15  # default fallback

    if ensemble is not None:
        mean_proba, unc = ensemble.predict_proba_with_uncertainty(X)
        uncertainty = float(unc[0])
        proba = mean_proba[0]
        pred_class = int(proba.argmax())
        max_prob = float(proba.max())
        # Use the TUNED thresholds from the ensemble (set during training)
        prob_threshold = ensemble.prob_threshold
        std_threshold = ensemble.std_threshold

    # 6. Uncertainty rule
    is_uncertain = (max_prob < prob_threshold) or (uncertainty > std_threshold)
    status = "Uncertain" if is_uncertain else "Confident"

    # 7. Determine output fields
    pred_name = LABEL_NAMES.get(pred_class, "unknown")
    is_valid = pred_name == "valid" and not is_uncertain
    failure_mode = (
        "none" if is_valid else (pred_name if not is_uncertain else "unknown")
    )

    prob_dict = {}
    for i, name in LABEL_NAMES.items():
        prob_dict[name] = round(float(proba[i]), 4) if i < len(proba) else 0.0

    result = {
        "valid": "uncertain" if is_uncertain else is_valid,
        "failure_mode": failure_mode,
        "confidence": round(max_prob, 4),
        "uncertainty": round(uncertainty, 4),
        "status": status,
        "predicted_class": pred_name,
        "label_index": pred_class,
        "probabilities": prob_dict,
        "thresholds_used": {
            "prob_threshold": round(prob_threshold, 4),
            "std_threshold": round(std_threshold, 4),
        },
    }

    if sample.error_type:
        result["build_warning"] = sample.error_type

    return result


def explain_prediction(params: dict, rf, feature_names: list[str]) -> dict:
    """Return top-10 most important features for this prediction."""
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    explanation = {}
    for i in top_idx:
        explanation[feature_names[i]] = round(float(importances[i]), 4)
    return explanation


def main():
    parser = argparse.ArgumentParser(
        description="CAD Validity Predictor — predict shape validity from parameters",
        epilog="Example: python scripts/predict.py test_input.json --pretty",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to JSON file with shape parameters",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Inline JSON string with shape parameters",
    )
    parser.add_argument("--models-dir", default=None, help="Path to models directory")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print output JSON"
    )
    parser.add_argument(
        "--explain", action="store_true", help="Include top feature importances"
    )
    parser.add_argument(
        "--test-suite",
        action="store_true",
        help="Run built-in test suite on known samples",
    )
    args = parser.parse_args()

    # Load models
    models_dir = Path(args.models_dir) if args.models_dir else None
    rf, ensemble, feature_names = load_models(models_dir)

    # Test suite mode
    if args.test_suite:
        run_test_suite(rf, ensemble, feature_names)
        return

    # Normal prediction mode
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": True, "messages": [f"Invalid JSON: {e}"]}))
            sys.exit(1)
    elif args.input_file:
        p = Path(args.input_file)
        if not p.exists():
            print(json.dumps({"error": True, "messages": [f"File not found: {p}"]}))
            sys.exit(1)
        try:
            with open(p, "r") as f:
                params = json.load(f)
        except json.JSONDecodeError as e:
            print(
                json.dumps({"error": True, "messages": [f"Invalid JSON in {p}: {e}"]})
            )
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    t0 = time.perf_counter()
    result = predict(params, rf=rf, ensemble=ensemble, feature_names=feature_names)
    latency = time.perf_counter() - t0
    result["latency_ms"] = round(latency * 1000, 1)

    if args.explain and not result.get("error"):
        result["explanation"] = explain_prediction(params, rf, feature_names)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

    if result.get("error"):
        sys.exit(1)


def run_test_suite(rf, ensemble, feature_names):
    """Run predictions on known test cases and report results."""
    test_cases = [
        # Valid shapes
        {
            "name": "Valid box (large)",
            "expected": "valid",
            "params": {
                "family": "primitive_box",
                "length": 50,
                "width": 40,
                "height": 30,
            },
        },
        {
            "name": "Valid box (small)",
            "expected": "valid",
            "params": {"family": "primitive_box", "length": 8, "width": 6, "height": 5},
        },
        {
            "name": "Valid cylinder",
            "expected": "valid",
            "params": {"family": "primitive_cylinder", "radius": 15, "height": 30},
        },
        {
            "name": "Valid sphere",
            "expected": "valid",
            "params": {"family": "primitive_sphere", "radius": 25},
        },
        {
            "name": "Valid filleted box",
            "expected": "valid",
            "params": {
                "family": "filleted_box",
                "length": 30,
                "width": 25,
                "height": 20,
                "fillet_r": 2.0,
            },
        },
        {
            "name": "Boolean union",
            "expected": "valid",
            "params": {
                "family": "boolean_union",
                "l1": 20,
                "w1": 20,
                "h1": 20,
                "l2": 15,
                "w2": 15,
                "h2": 15,
                "ox": 10,
                "oy": 10,
                "oz": 5,
            },
        },
        {
            "name": "Box (cube)",
            "expected": "valid",
            "params": {
                "family": "primitive_box",
                "length": 20,
                "width": 20,
                "height": 20,
            },
        },
        # Degenerate shapes
        {
            "name": "Zero-dim box (degenerate)",
            "expected": "degenerate_face",
            "params": {
                "family": "zero_dim_box",
                "length": 10,
                "width": 10,
                "height": 1e-5,
            },
        },
        {
            "name": "Extreme aspect ratio (degen)",
            "expected": "degenerate_face",
            "params": {
                "family": "extreme_aspect_ratio",
                "length": 500,
                "width": 500,
                "height": 0.001,
            },
        },
        # Tolerance errors
        {
            "name": "Sub-tolerance box",
            "expected": "tolerance_error",
            "params": {
                "family": "sub_tolerance_box",
                "length": 1e-4,
                "width": 1e-4,
                "height": 1e-4,
            },
        },
        {
            "name": "Micro fillet",
            "expected": "tolerance_error",
            "params": {
                "family": "micro_fillet",
                "box_size": 0.05,
                "fillet_r": 1e-4,
            },
        },
        # Self-intersection
        {
            "name": "Bowtie extrude (self-int)",
            "expected": "self_intersection",
            "params": {
                "family": "bowtie_extrude",
                "w": 20,
                "h": 20,
                "extrude": 10,
                "cx": 10,
                "cy": 10,
            },
        },
        # Non-manifold
        {
            "name": "Face-sharing compound (NM)",
            "expected": "non_manifold",
            "params": {
                "family": "face_sharing_compound",
                "s1": 20,
                "s2": 20,
            },
        },
    ]

    print("=" * 78)
    print("  Inference Test Suite")
    print("=" * 78)

    results_list = []
    n_correct = 0
    for tc in test_cases:
        t0 = time.perf_counter()
        result = predict(
            tc["params"], rf=rf, ensemble=ensemble, feature_names=feature_names
        )
        lat = (time.perf_counter() - t0) * 1000

        pred = result.get("predicted_class", "ERROR")
        conf = result.get("confidence", 0)
        unc = result.get("uncertainty", 0)
        st = result.get("status", "?")
        expected = tc.get("expected", "?")

        # Check if prediction matches expected
        matches = pred == expected
        if matches:
            n_correct += 1
        icon = "[PASS]" if matches else "[FAIL]"

        print(
            f"  {icon} {tc['name']:30s} → {pred:20s} "
            f"(exp={expected:20s}) conf={conf:.3f} unc={unc:.4f} [{st}] ({lat:.0f}ms)"
        )
        results_list.append({**tc, "result": result, "latency_ms": lat})

    # Summary
    n_total = len(test_cases)
    n_ok = sum(1 for r in results_list if not r["result"].get("error"))
    n_uncertain = sum(
        1 for r in results_list if r["result"].get("status") == "Uncertain"
    )
    avg_lat = np.mean([r["latency_ms"] for r in results_list])
    print(f"\n  Passed (no errors): {n_ok}/{n_total}")
    print(f"  Correct predictions: {n_correct}/{n_total}")
    print(f"  Uncertain: {n_uncertain}/{n_total}")
    print(f"  Avg latency: {avg_lat:.0f}ms")

    # Save example outputs
    examples_dir = PROJECT_ROOT / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Pick one valid, one failure-like, one uncertain
    for r in results_list:
        res = r["result"]
        if res.get("valid") and "example_valid" not in [
            f.stem for f in examples_dir.glob("*.json")
        ]:
            with open(examples_dir / "example_valid.json", "w") as f:
                json.dump({"input": r["params"], "output": res}, f, indent=2)
        if not res.get("valid") and res.get("status") != "Uncertain":
            with open(examples_dir / "example_failure.json", "w") as f:
                json.dump({"input": r["params"], "output": res}, f, indent=2)
        if res.get("status") == "Uncertain":
            with open(examples_dir / "example_uncertain.json", "w") as f:
                json.dump({"input": r["params"], "output": res}, f, indent=2)

    print(f"  Example outputs saved to {examples_dir}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
