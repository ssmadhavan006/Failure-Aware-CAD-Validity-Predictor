"""Generate a labeled dataset of synthetic CAD samples with kernel-checked labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.generators import ALL_GENERATORS
from src.generators.base import LABEL_NAMES
from src.kernel_check import check_shape


def generate_dataset(
    samples_per_class: int = 500,
    seed: int = 42,
    output_path: str = "data/dataset.jsonl",
    export_step: bool = False,
    step_dir: str = "data/step_files",
) -> None:
    """Generate the full dataset."""

    rng = np.random.default_rng(seed)
    output = Path(PROJECT_ROOT) / output_path
    output.parent.mkdir(parents=True, exist_ok=True)

    if export_step:
        step_path = Path(PROJECT_ROOT) / step_dir
        step_path.mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        "total": 0,
        "by_class": {},
        "by_subfamily": {},
        "construction_errors": 0,
        "kernel_invalid": 0,
        "kernel_valid": 0,
    }

    print("\nDataset Generation Pipeline\n")

    t_start = time.time()

    with open(output, "w") as f:
        for GenClass in ALL_GENERATORS:
            gen = GenClass()
            class_name = LABEL_NAMES.get(gen.label, f"class_{gen.label}")
            print(
                f"  Generating {samples_per_class} samples for class "
                f"'{class_name}' (label={gen.label}) ..."
            )

            class_valid = 0
            class_invalid = 0
            class_errors = 0
            class_start = time.time()

            for i in range(samples_per_class):
                sample = gen.generate(rng)

                # Run kernel check if shape was produced
                if sample.shape is not None:
                    try:
                        result = check_shape(sample.shape)
                        sample.is_valid = result["is_valid"]
                        sample.topology = result["topology"]
                        sample.topology["brep_valid"] = result["brep_valid"]
                        sample.topology["is_manifold"] = result["is_manifold"]
                        sample.topology["manifold_issues"] = ",".join(
                            result.get("manifold_issues", [])
                        )
                        if result["is_valid"]:
                            class_valid += 1
                        else:
                            class_invalid += 1
                    except Exception as exc:
                        sample.is_valid = False
                        sample.error_type = sample.error_type or type(exc).__name__
                        sample.error_msg = sample.error_msg or str(exc)[:200]
                        class_errors += 1
                else:
                    class_errors += 1

                # Export STEP file (optional)
                if export_step and sample.shape is not None:
                    try:
                        import cadquery as cq

                        step_file = (
                            Path(PROJECT_ROOT) / step_dir / f"{class_name}_{i:04d}.step"
                        )
                        solid = cq.Shape(sample.shape)
                        cq.exporters.export(
                            cq.Workplane().add(solid),
                            str(step_file),
                        )
                    except Exception:
                        pass  # non-critical

                # Write record
                record = sample.to_record()
                record["sample_id"] = f"{class_name}_{i:04d}"
                f.write(json.dumps(record) + "\n")

                stats["total"] += 1

            elapsed = time.time() - class_start
            stats["by_class"][class_name] = {
                "total": samples_per_class,
                "kernel_valid": class_valid,
                "kernel_invalid": class_invalid,
                "construction_errors": class_errors,
            }
            stats["kernel_valid"] += class_valid
            stats["kernel_invalid"] += class_invalid
            stats["construction_errors"] += class_errors

            print(
                f"    [ok] {samples_per_class} samples in {elapsed:.1f}s "
                f"(valid={class_valid}, invalid={class_invalid}, "
                f"errors={class_errors})"
            )

    total_time = time.time() - t_start

    # Summary
    print()
    print("=" * 55)
    print("  Dataset Summary")
    print("=" * 55)
    print(f"  Total samples:       {stats['total']}")
    print(f"  Kernel valid:        {stats['kernel_valid']}")
    print(f"  Kernel invalid:      {stats['kernel_invalid']}")
    print(f"  Construction errors: {stats['construction_errors']}")
    print(f"  Output file:         {output}")
    print(f"  Time elapsed:        {total_time:.1f}s")
    print()

    print("  Per-class breakdown:")
    for cls_name, cls_stats in stats["by_class"].items():
        print(
            f"    {cls_name:25s}  "
            f"total={cls_stats['total']:4d}  "
            f"valid={cls_stats['kernel_valid']:4d}  "
            f"invalid={cls_stats['kernel_invalid']:4d}  "
            f"errors={cls_stats['construction_errors']:4d}"
        )
    print()

    # Save stats alongside
    stats_path = output.parent / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled CAD dataset for validity prediction."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Number of samples per failure class (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.jsonl",
        help="Output JSONL file path (default: data/dataset.jsonl)",
    )
    parser.add_argument(
        "--export-step",
        action="store_true",
        help="Also export each sample as a STEP file",
    )
    parser.add_argument(
        "--step-dir",
        type=str,
        default="data/step_files",
        help="Directory for STEP files (default: data/step_files)",
    )
    args = parser.parse_args()

    generate_dataset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        output_path=args.output,
        export_step=args.export_step,
        step_dir=args.step_dir,
    )


if __name__ == "__main__":
    main()
