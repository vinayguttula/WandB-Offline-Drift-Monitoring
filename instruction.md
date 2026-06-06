Hey there, I need you to implement an offline drift monitoring pipeline for our machine learning project! 

The system should have two main pieces. First, I need a Python CLI located at `/app/environment/cli/train.py`. This script should train a tabular `RandomForestClassifier` from a provided CSV data path (using the last column as the target). It must log the model's accuracy and confusion matrix to a local experiment tracking system without requiring internet access. The trained model artifact must be serialized and saved to a specified output path so it can be loaded later. While training, you also need to compute the training feature distributions for each numerical feature to serve as a baseline for drift detection. These baseline distribution statistics should be saved to a specified JSON output path.

Second, I need you to extend our API backend located in `/app/environment/django_project/api/views.py`. Set up a `/api/predict/` endpoint that receives a list of feature values, loads the saved model artifact, and returns the prediction. Then, add a `/api/drift/` endpoint that accepts a batch of feature arrays and computes the Population Stability Index (PSI) against the saved training baseline statistics. The drift endpoint should calculate and return the PSI for each feature by comparing the actual batch distributions against the expected training distributions. Make sure you handle numerical stability in ratio calculations to avoid division by zero errors. If any feature's PSI exceeds 0.1, flag the entire batch as drifted.

Make sure you validate the inputs appropriately and gracefully return standard HTTP 400 Bad Request errors if schemas or data formats are mismatched!

## Testing Specifications
- The command-line utility `train.py` must accept exactly three positional arguments in the following order: `data_path`, `model_output_path`, and `hist_output_path`.
- The baseline distribution output file must have the features keyed exactly by stringified indices (e.g., `"0"`, `"1"`, `"2"`, `"3"`).
- Each feature distribution must have exactly 10 bins (this means the counts array has a length of 10, and the bin_edges array has a length of 11).
- The prediction endpoint `/api/predict/` must return a prediction formatted as `{"prediction": <integer_prediction>}`. If the array has fewer than 4 features, it must return a standard dictionary of exactly `{"error": "Invalid input schema"}` alongside an HTTP 400.
- The drift endpoint `/api/drift/` must return the `drift_metrics` dictionary explicitly using integer string keys for the feature indices (e.g., `"0"`, `"1"`).
- The WandB run logging directory must be initiated explicitly under the standard local `wandb/` directory inside the repository environment.
