import os
import json
import subprocess
import pytest
from pathlib import Path
from django.test import Client

ENV_DIR = Path("/app/environment")

DATA_PATH = ENV_DIR / "data" / "train.csv"
MODEL_OUTPUT_PATH = ENV_DIR / "model.joblib"
HIST_OUTPUT_PATH = ENV_DIR / "hist.json"
TRAIN_SCRIPT = ENV_DIR / "cli" / "train.py"

@pytest.fixture(scope="session", autouse=True)
def run_training():
    """Run the training CLI before tests to setup artifact states."""
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = str(ENV_DIR)
    
    if MODEL_OUTPUT_PATH.exists():
        MODEL_OUTPUT_PATH.unlink()
    if HIST_OUTPUT_PATH.exists():
        HIST_OUTPUT_PATH.unlink()
        
    result = subprocess.run([
        "python", str(TRAIN_SCRIPT),
        str(DATA_PATH),
        str(MODEL_OUTPUT_PATH),
        str(HIST_OUTPUT_PATH)
    ], env=env, capture_output=True, text=True)
    
    yield result

    subprocess.run(["pkill", "-f", "wandb-service"], check=False)

def test_training_execution(run_training):
    """Verify that the training CLI executed successfully with zero exit code."""
    assert run_training.returncode == 0, f"Training failed:\n{run_training.stderr}"

def test_wandb_offline_artifacts(run_training):
    """Verify that WandB successfully created offline run artifacts in the environment directory."""
    wandb_dir = ENV_DIR / "wandb"
    assert wandb_dir.exists(), "WandB directory not found"
    offline_runs = list(wandb_dir.glob("offline-run-*"))
    assert len(offline_runs) > 0, "No offline WandB runs found"
    
    # Verify accuracy metrics were logged properly
    run_dir = sorted(offline_runs)[-1]
    
    # Check wandb-summary.json for the explicitly required accuracy metric
    summary_file = run_dir / "files" / "wandb-summary.json"
    history_file = run_dir / "logs" / "wandb-history.jsonl"
    
    found_accuracy = False
    
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if "accuracy" in summary or "acc" in summary:
            found_accuracy = True
            assert 0.0 <= float(summary.get("accuracy", summary.get("acc", 0.0))) <= 1.0
            
    if not found_accuracy and history_file.exists():
        # Fallback to history check if summary hasn't flushed properly in offline mode
        with open(history_file, 'r') as f:
            for line in f:
                if "accuracy" in line or "acc" in line:
                    found_accuracy = True
                    break
                    
    # Ultimate fallback: wandb offline mode can sometimes solely push to the compressed .wandb file
    if not found_accuracy:
        wandb_sqlite = run_dir / f"run-{run_dir.name.split('-')[-1]}.wandb"
        if wandb_sqlite.exists():
            try:
                content = wandb_sqlite.read_bytes()
                if b"accuracy" in content or b"acc" in content:
                    found_accuracy = True
            except Exception:
                pass
                
    assert found_accuracy, "Accuracy metric not logged to wandb history or summary"
    
    # Verify confusion matrix table was logged
    tables = list(run_dir.glob("files/media/table/*.table.json"))
    assert len(tables) > 0, "No confusion matrix table logged"

def test_output_files_created(run_training):
    """Verify that the serialized model and histogram output files are created with correct schemas."""
    assert MODEL_OUTPUT_PATH.exists(), "Model output not found"
    assert HIST_OUTPUT_PATH.exists(), "Histogram output not found"
    
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    model = joblib.load(MODEL_OUTPUT_PATH)
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier, got {type(model).__name__}"
    
    with open(HIST_OUTPUT_PATH, 'r') as f:
        hist_data = json.load(f)
        
    assert len(hist_data) == 4, "Expected histograms for 4 features"
    for feature, data in hist_data.items():
        assert "counts" in data and "bin_edges" in data
        assert len(data["counts"]) == 10
        assert len(data["bin_edges"]) == 11
        assert sum(data["counts"]) == 100 
        
@pytest.fixture
def api_client():
    """Setup Django test client fixture."""
    return Client()

def test_predict_endpoint_success(api_client, run_training):
    """Verify that the /api/predict/ endpoint correctly returns a prediction given valid features."""
    payload = {"features": [0.5, -1.2, 0.3, 0.9]}
    response = api_client.post('/api/predict/', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] in [0, 1], "Prediction must be a class label"
    
    import joblib
    model = joblib.load(MODEL_OUTPUT_PATH)
    expected = int(model.predict([payload["features"]])[0])
    assert data["prediction"] == expected

def test_predict_endpoint_invalid_schema(api_client, run_training):
    """Verify that the /api/predict/ endpoint rejects malformed input schemas with HTTP 400."""
    response = api_client.post('/api/predict/', data=json.dumps({"wrong": [1,2,3,4]}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
    
    response = api_client.post('/api/predict/', data=json.dumps({"features": [1,2,3]}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
    
def test_drift_endpoint_success(api_client, run_training):
    """Verify that the /api/drift/ endpoint successfully calculates Population Stability Index (PSI) drift."""
    batch = [
        {"features": [3.5, -1.2, 0.3, 0.9]},
        {"features": [3.6, -1.0, 0.2, 0.8]},
        {"features": [4.0, -1.1, 0.4, 0.7]},
        {"features": [3.8, -1.3, 0.1, 0.6]}
    ] * 5 
    
    payload = {"batch": batch}
    response = api_client.post('/api/drift/', data=json.dumps(payload), content_type='application/json')
    
    assert response.status_code == 200
    data = response.json()
    assert "drift_metrics" in data
    assert "is_drifted" in data
    
    metrics = data["drift_metrics"]
    assert len(metrics) == 4
    for key in ["0", "1", "2", "3"]:
        assert key in metrics
        assert float(metrics[key]) >= 0.0
        
    assert float(metrics["0"]) > 0.1
    assert data["is_drifted"] is True
    
    # Compute expected PSI from baseline hist and the test batch
    with open(HIST_OUTPUT_PATH, 'r') as f:
        hist_data = json.load(f)
        
    import numpy as np
    
    for key in ["0", "1", "2", "3"]:
        expected_counts = np.array(hist_data[key]["counts"])
        expected_pct = expected_counts / np.sum(expected_counts)
        bin_edges = hist_data[key]["bin_edges"]
        
        batch_feat = np.array([item["features"][int(key)] for item in batch])
        actual_counts = np.zeros_like(expected_counts)
        for j in range(len(expected_counts)):
            if j == len(expected_counts) - 1:
                actual_counts[j] = np.sum((batch_feat >= bin_edges[j]) & (batch_feat <= bin_edges[j+1]))
            else:
                actual_counts[j] = np.sum((batch_feat >= bin_edges[j]) & (batch_feat < bin_edges[j+1]))
                
        actual_total = np.sum(actual_counts)
        if actual_total == 0:
            actual_pct = np.zeros_like(actual_counts, dtype=float)
        else:
            actual_pct = actual_counts / actual_total
            
        eps = 1e-6
        expected_pct_eps = expected_pct + eps
        actual_pct_eps = actual_pct + eps
        
        psi_per_bin = (actual_pct_eps - expected_pct_eps) * np.log(actual_pct_eps / expected_pct_eps)
        expected_psi = np.sum(psi_per_bin)
        
        assert abs(float(metrics[key]) - expected_psi) < 0.01, f"PSI calculation incorrect for feature {key}"

def test_drift_endpoint_no_drift(api_client, run_training):
    """Verify that the /api/drift/ endpoint returns is_drifted=False when batch matches training distribution."""
    # Use exact values from training data to guarantee exactly 0.0 PSI mathematically
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    features_list = df.drop(columns=["target"]).values.tolist()
    
    # Use the full dataset to match the original histogram distributions identically
    batch = [{"features": f} for f in features_list]
    
    payload = {"batch": batch}
    response = api_client.post('/api/drift/', data=json.dumps(payload), content_type='application/json')
    
    assert response.status_code == 200
    data = response.json()
    assert "drift_metrics" in data
    for key, v in data["drift_metrics"].items():
        assert float(v) < 0.01, f"PSI should be ~0 for same data: {v}"
        
    assert data["is_drifted"] is False
    
    # Explicitly calculate expected hardcoded value (which should be epsilon derived effectively 0.0)
    assert abs(float(data["drift_metrics"]["0"]) - 0.0) < 0.01, "Expected PSI for identical distributions must be ~0.0"

def test_drift_numerical_stability(api_client, run_training):
    """Verify that the /api/drift/ endpoint safely handles numerical stability (zero division) in PSI calculation."""
    batch = [{"features": [0.0, 0.0, 0.0, 0.0]}] * 20
    response = api_client.post('/api/drift/', data=json.dumps({"batch": batch}), content_type='application/json')
    
    assert response.status_code == 200
    data = response.json()
    for v in data["drift_metrics"].values():
        val = float(v)
        assert val == val  # Not NaN
        assert val != float('inf') and val != float('-inf')

def test_drift_endpoint_invalid_schema(api_client, run_training):
    """Verify that the /api/drift/ endpoint safely handles invalid schema formats with HTTP 400."""
    response = api_client.post('/api/drift/', data=json.dumps({"wrong": []}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
