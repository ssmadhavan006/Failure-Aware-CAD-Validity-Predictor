# Failure-Aware CAD Validity Predictor — Project Context

## Project Overview

This is a **machine learning pipeline** that predicts whether a proposed CAD construction will succeed or fail inside a CAD kernel **before execution**. The project uses synthetic data generation, feature engineering, and Random Forest classification to achieve **99.0% accuracy** on validity prediction.

### Core Purpose
- Predict CAD model validity from parametric inputs before expensive geometry kernel operations
- Classify failures into 5 categories: `valid`, `self_intersection`, `non_manifold`, `degenerate_face`, `tolerance_error`
- Provide uncertainty quantification and SHAP-based explainability

### Technology Stack
| Category | Technologies |
|----------|-------------|
| **CAD Kernel** | CadQuery 2.2+ (includes OCP/OpenCascade bindings) |
| **ML Framework** | scikit-learn 1.0+ (Random Forest, calibration, pipelines) |
| **Data Processing** | numpy 1.21+, pandas 1.3+, joblib 1.1+ |
| **Visualization** | matplotlib 3.4+, seaborn 0.11+ |
| **Explainability** | SHAP 0.40+ |
| **Development** | Jupyter notebooks |

## Project Structure

```
cad/
├── README.md                 # High-level overview
├── requirements.txt          # Python dependencies
├── test_cad_setup.py         # Environment verification
├── test_input.json           # Example prediction input
├── GEMINI.md                 # AI assistant context
├── QWEN.md                   # This file — comprehensive project context
│
├── src/                      # Source code package
│   ├── __init__.py
│   ├── kernel_check.py       # OCP BRepCheck_Analyzer wrapper
│   ├── features/
│   │   ├── __init__.py       # extract_features(), FEATURE_NAMES, build_pipeline()
│   │   ├── base_features.py  # Bounding box, topology, ratios, tolerance metrics
│   │   └── graph_features.py # Face Adjacency Graph features
│   └── generators/
│       ├── __init__.py       # ALL_GENERATORS registry
│       ├── base.py           # BaseGenerator ABC, Sample dataclass, LABEL_NAMES
│       ├── valid.py          # ValidGenerator (5 sub-families)
│       ├── self_intersection.py  # Bowtie, twisted polygon, multi-cross
│       ├── non_manifold.py       # Face/edge/vertex sharing, open shell, T-junction
│       ├── degenerate.py         # Zero-dim, near-zero extrude, extreme aspect
│       └── tolerance.py          # Sub-tolerance, scale mismatch, micro fillet
│
├── scripts/                  # Executable pipeline scripts
│   ├── generate_dataset.py   # Synthetic data generation (2500+ samples)
│   ├── extract_features.py   # Feature matrix construction
│   ├── analyze_features.py   # Feature importance analysis
│   ├── train_models.py       # RF + calibration + ensemble
│   ├── audit_checklist.py    # Model audit
│   ├── phase3_audit.py       # Comprehensive audit
│   ├── phase4_evaluation.py  # Evaluation & analysis
│   ├── predict.py            # CLI predictor
│   └── diagnose_predictions.py  # Prediction diagnostics
│
├── data/                     # Generated artifacts (git-ignored)
│   ├── dataset.jsonl         # Raw dataset
│   ├── X.npy, y.npy          # Feature matrix and labels
│   ├── feature_names.json    # Feature name list
│   ├── dataset_stats.json    # Generation statistics
│   ├── feature_importance.json
│   ├── selected_features.json
│   ├── split_info.json       # Train/val/test split info
│   └── step_files/           # STEP file exports
│
├── models/                   # Trained models (git-ignored)
│   ├── rf_model.joblib       # Random Forest classifier
│   ├── model.pkl             # Calibrated model
│   ├── uncertainty_ensemble.joblib
│   └── figures/              # Evaluation plots
│
├── examples/                 # Example prediction I/O (git-ignored)
├── samples/                  # Sample CAD models
│   ├── valid/
│   └── invalid/
├── notebooks/                # Jupyter analysis notebooks
└── docs/                     # Detailed documentation
    ├── 00_comprehensive_report.md
    ├── 01_data_generation.md
    ├── 02_feature_engineering.md
    ├── 03_model_training.md
    ├── 04_evaluation_and_failure_analysis.md
    └── TECHNICAL_REPORT.md
```

## Building and Running

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify environment
python test_cad_setup.py
```

### Pipeline Execution Order

| Step | Script | Description |
|-------|--------|-------------|
| **0** | `test_cad_setup.py` | Environment verification |
| **1** | `scripts/generate_dataset.py` | Generate 2500+ synthetic CAD samples |
| **2** | `scripts/extract_features.py` | Extract 88 features per sample |
| **2** | `scripts/analyze_features.py` | Feature importance analysis |
| **3** | `scripts/train_models.py` | Train RF + calibration + ensemble |
| **3** | `scripts/audit_checklist.py` | Model audit |
| **4** | `scripts/phase4_evaluation.py` | Comprehensive evaluation |
| **5** | `scripts/predict.py` | CLI prediction interface |

### Prediction Usage

```bash
# From JSON file
python scripts/predict.py test_input.json --pretty

# Inline parameters
python scripts/predict.py --params '{"family":"primitive_box","length":30,"width":20,"height":10}' --pretty

# With SHAP explanation
python scripts/predict.py test_input.json --pretty --explain

# Run test suite (10 cases)
python scripts/predict.py --test-suite
```

### Example Output

```json
{
  "valid": true,
  "failure_mode": "none",
  "confidence": 0.968,
  "uncertainty": 0.0081,
  "status": "Confident",
  "predicted_class": "valid",
  "label_index": 0,
  "probabilities": { "valid": 0.968, "self_intersection": 0.025, ... }
}
```

## Key Components

### Supported Shape Families

| Category | Sub-Families |
|----------|-------------|
| **Valid** | `primitive_box`, `primitive_cylinder`, `primitive_sphere`, `boolean_union`, `filleted_box` |
| **Self-Intersection** | `bowtie_extrude`, `twisted_polygon`, `multi_cross` |
| **Non-Manifold** | `face_sharing_compound`, `edge_sharing_compound`, `vertex_sharing_compound`, `open_shell`, `t_junction` |
| **Degenerate** | `zero_dim_box`, `near_zero_extrude`, `extreme_aspect_ratio`, `collinear_extrude` |
| **Tolerance** | `sub_tolerance_box`, `scale_mismatch_boolean`, `micro_fillet`, `near_coincident_boolean` |

### Feature Engineering (88 features)

1. **Base Geometry** (5): Bounding box dimensions, volume, diagonal
2. **Sorted Dims** (3): min/mid/max dimensions
3. **Ratios** (3): aspect_ratio, mid_ratio, compactness
4. **Volume/Area** (2): volume, surface_area
5. **Topology** (9): vertices, edges, wires, faces, shells, solids, compsolids, compounds, euler_char
6. **Flags** (3): has_boolean_op, has_compound, is_multi_solid
7. **Tolerance** (3): min_dim_over_tol, log_min_dim, log_volume
8. **Interaction** (8): Cross-term combinations for tolerance/degeneracy detection
9. **Graph Structure** (11): Face Adjacency Graph statistics
10. **Face Type Distribution** (7): plane/cylinder/cone/sphere/torus/bspline/other fractions
11. **Area/Edge Stats** (6): face area and edge length statistics

### Model Architecture

- **Base Classifier**: Random Forest with balanced class weights
- **Calibration**: Platt scaling via CalibratedClassifierCV
- **Uncertainty**: Ensemble of 5 Random Forests with tuned thresholds
- **Decision Thresholds**: prob_threshold=0.6, std_threshold=0.15 (tuned during training)

## Development Conventions

### Code Style
- **Type hints**: All functions use `typing` module annotations
- **Docstrings**: Google-style docstrings with Parameters/Returns sections
- **Imports**: Organized with standard library → third-party → local; use `__future__` annotations
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants

### Testing Practices
- `test_cad_setup.py` verifies environment setup
- **Test Suite**: `predict.py --test-suite` runs 10+ predefined test cases
- **Audit Scripts**: `audit_checklist.py` validates feature coverage

### Git Conventions
- **Ignored**: `.venv/`, `__pycache__/`, `*.step`, `*.stl`, `data/*.npy`, `models/*.joblib`, `examples/`
- **Tracked**: Source code, scripts, documentation, configuration files

### Data Conventions
- **Feature Order**: `FEATURE_NAMES` list in `src/features/__init__.py` defines canonical order
- **Label Encoding**: 0=valid, 1=self_intersection, 2=non_manifold, 3=degenerate_face, 4=tolerance_error
- **Dataset Format**: JSONL with flattened params (`param_*`) and topology (`topo_*`)

## Important Notes

### OCP/CadQuery Integration
- CadQuery 2.2+ includes OCP (OpenCascade Python bindings) — no separate pythonocc-core needed
- Kernel validity checks use `BRepCheck_Analyzer` from OCP
- Topology counting uses `TopExp_Explorer` with `TopAbs_*` entity types

### Manual Dependencies
- **pythonocc-core**: Only available via conda (`conda install -c conda-forge pythonocc-core`)
- Not required for this project — CadQuery's bundled OCP is sufficient

### File Size Considerations
- Large artifacts (models, data files, STEP exports) are git-ignored
- Regenerate via pipeline scripts as needed
- Use `--export-step` flag in `generate_dataset.py` for STEP file exports

## Quick Reference

### Key Classes
- `BaseGenerator`: Abstract base for all shape generators
- `Sample`: Dataclass holding generated CAD sample data
- `UncertaintyEnsemble`: Ensemble wrapper for uncertainty quantification
- `RuleBasedClassifier`: Heuristic baseline classifier

### Key Functions
- `extract_features(params, shape)`: Main feature extraction entry point
- `check_shape(shape)`: Kernel validity check with manifold detection
- `predict(params)`: High-level prediction function
- `build_pipeline()`: Returns sklearn Pipeline with StandardScaler

### Configuration Files
- `feature_names.json`: 88 feature names in canonical order
- `split_info.json`: Train/val/test split metadata
- `dataset_stats.json`: Generation statistics per class/family
