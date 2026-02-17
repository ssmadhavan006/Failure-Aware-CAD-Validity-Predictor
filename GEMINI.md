# Gemini Project Context: Failure-Aware CAD Validity Predictor

## Project Overview

This project is a machine learning pipeline that predicts whether a given set of parameters will result in a valid CAD model *before* the expensive geometry kernel operation is executed. It is a classification problem with the following classes: `valid`, `self_intersection`, `non_manifold`, `degenerate_face`, and `tolerance_error`.

The project is structured into several stages:
1.  **Data Generation:** Synthetically create CAD models (both valid and invalid) from a set of predefined "families" (e.g., `primitive_box`, `bowtie_extrude`).
2.  **Feature Extraction:** For each generated shape, extract a comprehensive feature vector. This includes:
    *   **Base Features:** Bounding box dimensions, volume, surface area, topology counts (faces, edges, etc.), and various geometric ratios.
    *   **Graph Features:** A Face Adjacency Graph is constructed, and graph-based statistics (degree, centrality, etc.) are computed.
3.  **Model Training:** A Random Forest classifier is trained on the extracted features. The model is calibrated using Platt scaling, and an ensemble of models is used for uncertainty quantification.
4.  **Prediction:** A CLI is provided to predict the validity of a new set of parameters, returning the predicted class, confidence, and uncertainty.

## Key Technologies

*   **Language:** Python
*   **Core Libraries:**
    *   `cadquery`: For B-rep (Boundary Representation) solid modeling and geometry kernel operations.
    *   `scikit-learn`: For machine learning (Random Forest, calibration).
    *   `numpy`, `pandas`: For data manipulation.
    *   `joblib`: For saving and loading trained models.
    *   `shap`: For model explainability.

## Building and Running

### Local Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```
2.  **Activate the environment:**
    *   Windows: `.venv\Scripts\activate`
    *   Linux/macOS: `source .venv/bin/activate`
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Verify the setup:**
    ```bash
    python test_cad_setup.py
    ```

### Running Predictions

The primary prediction script is `scripts/predict.py`.

*   **From a JSON file:**
    ```bash
    python scripts/predict.py test_input.json --pretty
    ```
*   **With inline parameters:**
    ```bash
    python scripts/predict.py --params '{"family":"primitive_box","length":30,"width":20,"height":10}' --pretty
    ```
*   **With SHAP-based explanation:**
    ```bash
    python scripts/predict.py test_input.json --pretty --explain
    ```

## Development Conventions

*   **Pipeline Scripts:** The project is organized into distinct, sequential steps, with scripts for each:
    *   `scripts/generate_dataset.py` (data generation)
    *   `scripts/extract_features.py` (feature extraction)
    *   `scripts/train_models.py` (model training)
    *   `scripts/phase4_evaluation.py` (evaluation)
    *   `scripts/predict.py` (prediction CLI)
    *   `scripts/diagnose_predictions.py` (Diagnostics)
*   **Feature Engineering:** Features are defined in the `src/features/` directory. New features should be added there and integrated into the main feature extraction pipeline.
*   **Data and Models:** The `data/` and `models/` directories are intended for generated datasets and trained models, respectively. They are git-ignored by default.
*   **Uncertainty Quantification:** The model provides an "uncertainty" score based on the standard deviation of predictions from an ensemble of models. If the confidence is low or the uncertainty is high, the prediction status is marked as "Uncertain".
*   **Testing:** A simple verification script `test_cad_setup.py` ensures the environment is correctly configured. The prediction script also includes a `--test-suite` argument to run a set of predefined test cases.
*   **Documentation:** Detailed documentation is available in the `docs/` directory, including comprehensive reports and step-specific guides.
*   **Samples:** Sample CAD models for testing and demonstration are stored in the `samples/` directory, organized by validity (`valid/` and `invalid/`).
