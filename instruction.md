You must build a machine learning pipeline and inference API that integrates offline model tracking and statistical drift monitoring.

## System Requirements

The system consists of two primary components:
1. A Python CLI for model training.
2. A Django API for serving predictions and calculating drift.

### Component 1: Python CLI (`/app/environment/cli/train.py`)
You must implement a Python command-line interface that trains a scikit-learn tabular classifier and logs artifacts to Weights & Biases (WandB) in **offline mode**.

**Inputs:**
The script must accept three positional arguments:
1. `data_path`: Path to a CSV file containing the training data. The last column is the target variable `target`, and all preceding columns are numerical features.
2. `model_output_path`: Path where the serialized trained model (using `joblib`) must be saved.
3. `hist_output_path`: Path where the feature histograms (JSON format) must be saved for later drift calculation.

**Behavior:**
- Ensure WandB operates strictly in offline mode by setting `WANDB_MODE="offline"`.
- Initialize a WandB run.
- Train a scikit-learn classifier (e.g., `RandomForestClassifier`).
- Log the model's hyperparameters and evaluation metrics (e.g., `accuracy`) using `wandb.log()`.
- Log the confusion matrix using `wandb.plot.confusion_matrix()`.
- Save the trained model to `model_output_path` and log it as a WandB artifact.
- Compute histograms for each feature in the training data using 10 equal-width bins. Save the bin edges and training set bin counts/frequencies to `hist_output_path` as a JSON file.

### Component 2: Django API (`/app/environment/django_project/api/views.py`)
You must extend a Django REST Framework API with two endpoints. The application should load the trained model and the saved training histograms at startup.

**Endpoint 1: Predict**
- **URL:** `POST /api/predict/`
- **Behavior:** Receives a single instance's features and returns the model's prediction.
- **Request Body (JSON):**
  ```json
  {
    "features": [1.2, 3.4, 0.5, 2.1]
  }
  ```
- **Responses:**
  - **200 OK:**
    ```json
    {
      "prediction": 1
    }
    ```
  - **400 Bad Request:** (If "features" key is missing or not a list of numbers matching the expected feature count)
    ```json
    {
      "error": "Invalid input schema"
    }
    ```

**Endpoint 2: Batch Drift**
- **URL:** `POST /api/drift/`
- **Behavior:** Receives a batch of inferences, computes the Population Stability Index (PSI) against the training feature histograms, and returns the PSI score per feature.
- **Request Body (JSON):**
  ```json
  {
    "batch": [
      {"features": [1.2, 3.4, 0.5, 2.1]},
      {"features": [1.1, 3.2, 0.4, 2.0]}
    ]
  }
  ```
- **Responses:**
  - **200 OK:** Returns a mapping of feature indices (as strings) to their calculated PSI scores, and a boolean indicating if any feature's PSI exceeds `0.1`.
    ```json
    {
      "drift_metrics": {
        "0": 0.05,
        "1": 0.12,
        "2": 0.01,
        "3": 0.08
      },
      "is_drifted": true
    }
    ```
  - **400 Bad Request:** (If "batch" key is missing or improperly formatted)
    ```json
    {
      "error": "Invalid input schema"
    }
    ```

### Drift Calculation Details (Population Stability Index)
For a given feature, PSI is calculated across the 10 bins:
`PSI = sum((Actual_Pct - Expected_Pct) * ln(Actual_Pct / Expected_Pct))`
- `Expected_Pct` is the proportion of the training data in each bin (calculated during training).
- `Actual_Pct` is the proportion of the batch data in each bin (using the training bin edges).
- Add a small epsilon (`1e-6`) to actual and expected proportions before division and logarithm to avoid division by zero or `ln(0)`.
