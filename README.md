# Failure-Aware CAD Validity Predictor

A machine-learning–based prediction pipeline that determines whether a proposed
CAD construction is likely to succeed or fail inside a CAD kernel **before
execution**.

## Quick Start

### Local Setup (pip)

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify environment
python test_cad_setup.py
```

> **Note:** `pythonocc-core` is only available via conda. The pip-based setup
> uses CadQuery's built-in OCP bindings for kernel access, which is sufficient
> for this project. For full pythonocc-core, use conda.

## Prediction

### CLI

```bash
# From JSON file
python scripts/predict.py test_input.json --pretty

# Inline parameters
python scripts/predict.py --params '{"family":"primitive_box","length":30,"width":20,"height":10}' --pretty

# With feature explanation
python scripts/predict.py test_input.json --pretty --explain

# Run built-in test suite (10 cases)
python scripts/predict.py --test-suite
```

**Output format:**
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

## Project Status

This project is a complete and functional machine learning pipeline for predicting CAD model validity. All phases of the project, from data generation to prediction, are complete and have been documented. The model achieves an accuracy of **99.0%** on the test set and includes uncertainty quantification to identify ambiguous predictions.

## Documentation

This `README.md` file provides a high-level overview of the project. For more detailed documentation, please refer to the files in the `docs` directory:

*   **[Comprehensive Project Report](docs/comprehensive_report.md):** A detailed report covering all aspects of the project, including feature design, model architecture, evaluation metrics, and failure analysis.
*   **[Phase 1: Data Generation](docs/01_data_generation.md):** A detailed description of the synthetic data generation process.
*   **[Phase 2: Feature Engineering](docs/02_feature_engineering.md):** A detailed description of the feature engineering pipeline.
*   **[Phase 3: Model Training](docs/03_model_training.md):** A detailed description of the model training and calibration process.
*   **[Phase 4: Evaluation & Failure Analysis](docs/04_evaluation_and_failure_analysis.md):** An in-depth analysis of the model's performance, including error analysis and explainability.

## Project Structure

```
cad/
├── README.md
├── requirements.txt
├── .gitignore
├── test_cad_setup.py       # Phase 0 verification script
├── test_input.json         # Example input for predict.py
├── GEMINI.md               # Project context for AI assistants
├── src/
│   ├── __init__.py
│   ├── kernel_check.py     # OCP kernel validity utilities
│   ├── features/
│   │   ├── __init__.py     # extract_features(), FEATURE_NAMES, build_pipeline()
│   │   ├── base_features.py  # Bounding box, topology, ratios
│   │   └── graph_features.py # Face adjacency graph features
│   └── generators/
│       ├── __init__.py     # ALL_GENERATORS registry
│       ├── base.py         # BaseGenerator ABC
│       ├── valid.py        # ValidGenerator (box, cylinder, sphere, union, fillet)
│       ├── self_intersection.py  # Bowtie, twisted polygon, multi-cross
│       ├── non_manifold.py       # Face/edge/vertex sharing, open shell, T-junction
│       ├── degenerate.py         # Zero-dim, near-zero extrude, extreme aspect
│       └── tolerance.py          # Sub-tolerance, scale mismatch, micro fillet
├── scripts/
│   ├── generate_dataset.py   # Phase 1: synthetic data generation
│   ├── extract_features.py   # Phase 2: feature matrix construction
│   ├── analyze_features.py   # Phase 2: feature importance analysis
│   ├── train_models.py       # Phase 3: RF + calibration + ensemble
│   ├── audit_checklist.py    # Phase 3: model audit
│   ├── phase3_audit.py       # Phase 3: comprehensive audit
│   ├── phase4_evaluation.py  # Phase 4: evaluation & analysis
│   ├── predict.py            # Phase 5: CLI predictor
│   └── diagnose_predictions.py  # Prediction diagnostics
├── data/                     # Generated data (gitignored)
│   ├── dataset.jsonl
│   ├── X.npy, y.npy
│   ├── feature_names.json
│   ├── dataset_stats.json
│   ├── feature_importance.json
│   ├── selected_features.json
│   ├── split_info.json
│   ├── audit_report.txt
│   └── step_files/           # STEP file exports
├── models/                   # Trained models (gitignored)
│   ├── rf_model.joblib
│   ├── model.pkl
│   ├── uncertainty_ensemble.joblib
│   └── figures/              # Evaluation plots
├── examples/                 # Example prediction I/O (gitignored)
├── samples/                  # Sample CAD models
│   ├── valid/
│   └── invalid/
├── notebooks/                # Jupyter analysis notebooks
└── docs/                     # Documentation
    ├── 00_comprehensive_report.md
    ├── 01_data_generation.md
    ├── 02_feature_engineering.md
    ├── 03_model_training.md
    ├── 04_evaluation_and_failure_analysis.md
    └── TECHNICAL_REPORT.md
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment & Prerequisites | ✅ Complete |
| 1 | Synthetic Data Generation | ✅ Complete |
| 2 | Feature Extraction | ✅ Complete |
| 3 | Model Training & Evaluation | ✅ Complete |
| 4 | Explainability & Reporting | ✅ Complete |
| 5 | Prediction Script & Interface | ✅ Complete |

## Supported Shape Families

| Category | Families |
|----------|----------|
| **Valid** | `primitive_box`, `primitive_cylinder`, `primitive_sphere`, `boolean_union`, `filleted_box` |
| **Self-Intersection** | `bowtie_extrude`, `twisted_polygon`, `multi_cross` |
| **Non-Manifold** | `face_sharing_compound`, `edge_sharing_compound`, `vertex_sharing_compound`, `open_shell`, `t_junction` |
| **Degenerate** | `zero_dim_box`, `near_zero_extrude`, `extreme_aspect_ratio`, `collinear_extrude` |
| **Tolerance** | `sub_tolerance_box`, `scale_mismatch_boolean`, `micro_fillet`, `near_coincident_boolean` |
