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
    
def test_output_files_created(run_training):
    """Verify that the serialized model and histogram output files are created with correct schemas."""
    assert MODEL_OUTPUT_PATH.exists(), "Model output not found"
    assert HIST_OUTPUT_PATH.exists(), "Histogram output not found"
    
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
    assert float(metrics["0"]) > 0.1
    assert data["is_drifted"] is True

def test_drift_endpoint_invalid_schema(api_client, run_training):
    """Verify that the /api/drift/ endpoint safely handles invalid schema formats with HTTP 400."""
    response = api_client.post('/api/drift/', data=json.dumps({"wrong": []}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
