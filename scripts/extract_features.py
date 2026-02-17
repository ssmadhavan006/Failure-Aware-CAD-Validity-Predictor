"""Feature extraction pipeline: reconstructs shapes and extracts base + graph features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators import ALL_GENERATORS  # noqa: E402
from src.features import (  # noqa: E402
    extract_all_features,
    extract_features as extract_features_fn,
    FEATURE_NAMES,
)


def _build_generator_map() -> dict[str, object]:
    """Map generator name -> generator instance."""
    return {gen.name: gen for gen in [G() for G in ALL_GENERATORS]}


def _reconstruct_shape(record: dict, gen_map: dict):
    """
    Reconstruct an OCP shape from a dataset record.

    Extracts param_* fields from the record, finds the matching
    generator, and calls build() to recreate the shape.
    """
    params = {}
    for k, v in record.items():
        if k.startswith("param_"):
            params[k[6:]] = v

    intended = record.get("intended_label_name", "")
    label_to_gen = {
        "valid": "valid",
        "self_intersection": "self_intersection",
        "non_manifold": "non_manifold",
        "degenerate_face": "degenerate",
        "tolerance_error": "tolerance",
    }

    gen_name = label_to_gen.get(intended)
    if gen_name is None or gen_name not in gen_map:
        return None

    gen = gen_map[gen_name]
    try:
        sample = gen.build(params)
        return sample.shape
    except Exception:
        return None


def extract_features(
    input_path: str = "data/dataset.jsonl",
    output_dir: str = "data",
) -> None:
    """Run the full feature extraction pipeline."""

    inp = Path(PROJECT_ROOT) / input_path
    out_dir = Path(PROJECT_ROOT) / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "features.jsonl"
    out_x = out_dir / "X.npy"
    out_y = out_dir / "y.npy"
    out_names = out_dir / "feature_names.json"

    if not inp.exists():
        print(f"Error: {inp} not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(inp, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} records from {inp}")

    gen_map = _build_generator_map()

    total = len(records)
    n_ok = 0
    n_skip = 0
    t0 = time.time()

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    jsonl_records: list[dict] = []

    for idx, rec in enumerate(records):
        # Skip samples that had construction errors
        if rec.get("error_type") and rec["error_type"] != "None":
            feat_rec = {
                "sample_idx": idx,
                "label": rec["label"],
                "label_name": rec["label_name"],
                "intended_label_name": rec["intended_label_name"],
                "sub_family": rec["sub_family"],
                "is_valid": rec["is_valid"],
                "has_shape": False,
            }
            jsonl_records.append(feat_rec)
            n_skip += 1
            continue

        # Reconstruct shape
        shape = _reconstruct_shape(rec, gen_map)
        if shape is None:
            feat_rec = {
                "sample_idx": idx,
                "label": rec["label"],
                "label_name": rec["label_name"],
                "intended_label_name": rec["intended_label_name"],
                "sub_family": rec["sub_family"],
                "is_valid": rec["is_valid"],
                "has_shape": False,
            }
            jsonl_records.append(feat_rec)
            n_skip += 1
            continue

        # Extract all features
        params = {k[6:]: v for k, v in rec.items() if k.startswith("param_")}
        row = extract_features_fn(params, shape, rec)
        features = extract_all_features(shape, rec)

        feature_rows.append(row)
        labels.append(rec["intended_label"])  # Use intended label, not kernel result

        # JSONL record
        feat_rec = {
            "sample_idx": idx,
            "label": rec["label"],
            "label_name": rec["label_name"],
            "intended_label_name": rec["intended_label_name"],
            "sub_family": rec["sub_family"],
            "is_valid": rec["is_valid"],
            "has_shape": True,
        }
        feat_rec.update(features)
        jsonl_records.append(feat_rec)
        n_ok += 1

        # Progress
        if (idx + 1) % 100 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{idx + 1:5d}/{total}]  ok={n_ok}  skip={n_skip}  "
                f"({rate:.1f} samples/s)"
            )

    elapsed = time.time() - t0

    # Save JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")

    # Save numpy arrays
    X = np.vstack(feature_rows) if feature_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(labels, dtype=np.int64)
    np.save(out_x, X)
    np.save(out_y, y)

    # Save feature names
    with open(out_names, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s -- {n_ok} features extracted, {n_skip} skipped")
    print(f"  X.shape = {X.shape}")
    print(f"  y.shape = {y.shape}  classes = {np.unique(y).tolist()}")
    print(f"  Feature names: {len(FEATURE_NAMES)} columns")
    print(f"  Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument("--input", default="data/dataset.jsonl")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    extract_features(args.input, args.output_dir)


if __name__ == "__main__":
    main()
