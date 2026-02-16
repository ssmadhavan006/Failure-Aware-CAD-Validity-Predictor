"""Quick diagnostic: run predictions on failure cases and write to file."""

import sys, json, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_models import UncertaintyEnsemble, RuleBasedClassifier
import __main__

__main__.UncertaintyEnsemble = UncertaintyEnsemble
__main__.RuleBasedClassifier = RuleBasedClassifier

from scripts.predict import predict, load_models

rf, ensemble, feature_names = load_models()

test_cases = [
    (
        "Valid box",
        "valid",
        {"family": "primitive_box", "length": 50, "width": 40, "height": 30},
    ),
    ("Valid sphere", "valid", {"family": "primitive_sphere", "radius": 25}),
    (
        "Boolean union",
        "valid",
        {
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
    ),
    (
        "Zero-dim box",
        "degenerate_face",
        {"family": "zero_dim_box", "length": 10, "width": 10, "height": 1e-5},
    ),
    (
        "Extreme aspect",
        "degenerate_face",
        {
            "family": "extreme_aspect_ratio",
            "length": 500,
            "width": 500,
            "height": 0.001,
        },
    ),
    (
        "Sub-tolerance box",
        "tolerance_error",
        {"family": "sub_tolerance_box", "length": 1e-4, "width": 1e-4, "height": 1e-4},
    ),
    (
        "Micro fillet",
        "tolerance_error",
        {"family": "micro_fillet", "box_size": 0.05, "fillet_r": 1e-4},
    ),
    (
        "Bowtie extrude",
        "self_intersection",
        {
            "family": "bowtie_extrude",
            "w": 20,
            "h": 20,
            "extrude": 10,
            "cx": 10,
            "cy": 10,
        },
    ),
    (
        "Face-sharing compound",
        "non_manifold",
        {"family": "face_sharing_compound", "s1": 20, "s2": 20},
    ),
    (
        "Edge-sharing compound",
        "non_manifold",
        {"family": "edge_sharing_compound", "s1": 20, "s2": 20},
    ),
    (
        "T-junction",
        "non_manifold",
        {
            "family": "t_junction",
            "main_size": 25,
            "branch_w": 5,
            "branch_h": 5,
            "branch_d": 15,
        },
    ),
]

out_path = PROJECT_ROOT / "models" / "diagnostic_results.txt"
lines = []
lines.append("=" * 90)
lines.append("  DIAGNOSTIC: Prediction Test Results")
lines.append("=" * 90)

n_correct = 0
for name, expected, params in test_cases:
    result = predict(params, rf=rf, ensemble=ensemble, feature_names=feature_names)
    if result.get("error"):
        lines.append(f"  ERROR  {name:30s}  error={result['messages']}")
        continue
    pred = result["predicted_class"]
    conf = result["confidence"]
    unc = result["uncertainty"]
    match = "PASS" if pred == expected else "FAIL"
    if pred == expected:
        n_correct += 1
    lines.append(
        f"  {match:4s}  {name:30s}  pred={pred:20s}  exp={expected:20s}  conf={conf:.3f}  unc={unc:.4f}"
    )

lines.append(f"\n  Score: {n_correct}/{len(test_cases)} correct")
lines.append("=" * 90)

output = "\n".join(lines)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)

# Also print to stderr (avoids stdout wrapper issues)
sys.stderr.write(output + "\n")
