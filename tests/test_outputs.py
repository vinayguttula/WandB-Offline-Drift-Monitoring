import os
import json
import subprocess
import pytest
from pathlib import Path
from django.test import Client

BASE_DIR = Path(__file__).resolve().parent.parent

# Set paths
DATA_PATH = BASE_DIR / "environment" / "data" / "train.csv"
PREDICT_DATA_PATH = BASE_DIR / "environment" / "data" / "predict.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "environment" / "model.joblib"
HIST_OUTPUT_PATH = BASE_DIR / "environment" / "hist.json"
TRAIN_SCRIPT = BASE_DIR / "environment" / "cli" / "train.py"

@pytest.fixture(scope="session", autouse=True)
def run_training():
    """Run the training CLI before tests."""
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = str(BASE_DIR / "environment")
    
    # Clean up previous runs
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

    # Cleanup logic if necessary (e.g., removing background wandb processes)
    subprocess.run(["pkill", "-f", "wandb-service"], check=False)

def test_training_execution(run_training):
    """Test that the CLI executed successfully."""
    assert run_training.returncode == 0, f"Training failed:\n{run_training.stderr}"

def test_wandb_offline_artifacts(run_training):
    """Test that WandB ran in offline mode and created local run directories."""
    wandb_dir = BASE_DIR / "environment" / "wandb"
    assert wandb_dir.exists(), "WandB directory not found"
    
    offline_runs = list(wandb_dir.glob("offline-run-*"))
    assert len(offline_runs) > 0, "No offline WandB runs found"
    
def test_output_files_created(run_training):
    """Test that the serialized model and histograms were created."""
    assert MODEL_OUTPUT_PATH.exists(), "Model output not found"
    assert HIST_OUTPUT_PATH.exists(), "Histogram output not found"
    
    with open(HIST_OUTPUT_PATH, 'r') as f:
        hist_data = json.load(f)
        
    # Expect 4 features
    assert len(hist_data) == 4, "Expected histograms for 4 features"
    
    for feature, data in hist_data.items():
        assert "counts" in data and "bin_edges" in data
        assert len(data["counts"]) == 10
        assert len(data["bin_edges"]) == 11
        assert sum(data["counts"]) == 100 # Total 100 rows in train.csv
        
@pytest.fixture
def api_client():
    """Django test client."""
    return Client()

def test_predict_endpoint_success(api_client, run_training):
    """Test prediction endpoint with valid schema."""
    payload = {"features": [0.5, -1.2, 0.3, 0.9]}
    response = api_client.post('/api/predict/', data=json.dumps(payload), content_type='application/json')
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))

def test_predict_endpoint_invalid_schema(api_client, run_training):
    """Test prediction endpoint handles bad schema."""
    # Missing 'features'
    response = api_client.post('/api/predict/', data=json.dumps({"wrong": [1,2,3,4]}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
    
    # Wrong number of features (expects 4)
    response = api_client.post('/api/predict/', data=json.dumps({"features": [1,2,3]}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
    
def test_drift_endpoint_success(api_client, run_training):
    """Test drift endpoint calculates PSI."""
    # Generate some shifted batch data
    batch = [
        {"features": [3.5, -1.2, 0.3, 0.9]},
        {"features": [3.6, -1.0, 0.2, 0.8]},
        {"features": [4.0, -1.1, 0.4, 0.7]},
        {"features": [3.8, -1.3, 0.1, 0.6]}
    ] * 5 # 20 items
    
    payload = {"batch": batch}
    response = api_client.post('/api/drift/', data=json.dumps(payload), content_type='application/json')
    
    assert response.status_code == 200
    data = response.json()
    assert "drift_metrics" in data
    assert "is_drifted" in data
    
    metrics = data["drift_metrics"]
    # Feature 0 is heavily shifted, should have high PSI
    assert float(metrics["0"]) > 0.1
    # Feature 1 is not shifted, should have low PSI (close to 0 but > 0 due to epsilon)
    # Actually, with a small sample size vs train it might vary, but is_drifted should be true
    assert data["is_drifted"] is True

def test_drift_endpoint_invalid_schema(api_client, run_training):
    """Test drift endpoint handles bad schema."""
    response = api_client.post('/api/drift/', data=json.dumps({"wrong": []}), content_type='application/json')
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid input schema"}
