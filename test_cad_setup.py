"""Verify that the development environment is correctly set up."""

import sys
import os

PASS = "[ok]"
FAIL = "[FAIL]"

def check_imports():
    """Check that all core libraries can be imported."""
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scikit-learn": "sklearn",
        "shap": "shap",
        "cadquery": "cadquery",
    }
    optional = {
        "OCP (via cadquery)": "OCP",
    }

    all_ok = True
    print("=" * 55)
    print("  Library Import Checks")
    print("=" * 55)

    for name, module in required.items():
        try:
            __import__(module)
            print(f"  {PASS}  {name}")
        except ImportError:
            print(f"  {FAIL}  {name}  — MISSING (required)")
            all_ok = False

    for name, module in optional.items():
        try:
            __import__(module)
            print(f"  {PASS}  {name}")
        except ImportError:
            print(f"  -  {name}  — not installed (optional)")

    print()
    return all_ok


def create_and_export_box(output_path="box.step"):
    """Create a simple CadQuery box and export it as a STEP file."""
    print("=" * 55)
    print("  CadQuery Box Creation & STEP Export")
    print("=" * 55)

    try:
        import cadquery as cq

        # Create a simple parametric box
        box = cq.Workplane("XY").box(10, 20, 30)

        # Export to STEP
        cq.exporters.export(box, output_path)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  {PASS}  Created box (10 × 20 × 30)")
            print(f"  {PASS}  Exported to '{output_path}' ({size_kb:.1f} KB)")
        else:
            print(f"  {FAIL}  STEP file not created or is empty")
            return False

    except Exception as e:
        print(f"  {FAIL}  Error: {e}")
        return False

    print()
    return True


def check_kernel_validity():
    """
    Perform a basic OCP kernel validity check on the box shape.
    This demonstrates access to the OpenCascade kernel that CadQuery wraps.
    """
    print("=" * 55)
    print("  OpenCascade Kernel Validity Check")
    print("=" * 55)

    try:
        import cadquery as cq
        from OCP.BRepCheck import BRepCheck_Analyzer

        box = cq.Workplane("XY").box(10, 20, 30)
        shape = box.val().wrapped  # Get the underlying OCC TopoDS_Shape

        analyzer = BRepCheck_Analyzer(shape)
        is_valid = analyzer.IsValid()

        if is_valid:
            print(f"  {PASS}  Kernel validity check PASSED (shape is valid)")
        else:
            print(f"  {FAIL}  Kernel validity check FAILED (shape is invalid)")
            return False

        # Count topology entities as a sanity check
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

        face_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_count += 1
            explorer.Next()

        edge_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge_count += 1
            explorer.Next()

        vertex_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex_count += 1
            explorer.Next()

        print(f"  {PASS}  Topology: {face_count} faces, {edge_count} edges, {vertex_count} vertices")

    except ImportError:
        print(f"  -  OCP not available — skipping kernel check")
        print(f"     (Install pythonocc-core via conda for full kernel access)")
        return True  # Not a failure, just optional

    except Exception as e:
        print(f"  {FAIL}  Error: {e}")
        return False

    print()
    return True


def main():
    print("\nEnvironment Verification\n")

    results = {}
    results["imports"] = check_imports()
    results["box_export"] = create_and_export_box()
    results["kernel_check"] = check_kernel_validity()

    # Summary
    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    all_passed = all(results.values())
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")

    print()
    if all_passed:
        print(f"  {PASS}  ALL CHECKS PASSED — Environment is ready!")
        print(f"     Python {sys.version.split()[0]}")
    else:
        print(f"  {FAIL}  SOME CHECKS FAILED — see details above")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
