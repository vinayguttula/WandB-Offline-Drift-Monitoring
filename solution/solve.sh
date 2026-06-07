#!/usr/bin/env bash
set -euo pipefail

# Get exactly where this script is located so we can resolve the root robustly
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE="$DIR/../environment"

# Ensure target directories exist using dynamic paths
mkdir -p "$BASE/cli"
mkdir -p "$BASE/django_project/api"
mkdir -p "$BASE/data"

# Write train script
cat << INNER_EOF > "$BASE/cli/train.py"
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import wandb
import joblib
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model_output_path")
    parser.add_argument("hist_output_path")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(project="drift-monitor")

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    wandb.log({"accuracy": accuracy})
    
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                        y_true=y.to_numpy(), preds=preds,
                        class_names=["0", "1"])})

    joblib.dump(model, args.model_output_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(args.model_output_path)
    run.log_artifact(artifact)

    histograms = {}
    for i, col in enumerate(X.columns):
        counts, bin_edges = np.histogram(X[col].dropna(), bins=10)
        histograms[str(i)] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist()
        }

    with open(args.hist_output_path, "w") as f:
        json.dump(histograms, f)

    wandb.finish()

if __name__ == "__main__":
    main()
INNER_EOF

# Write Django API views
cat << INNER_EOF > "$BASE/django_project/api/views.py"
import json
import math
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
from pathlib import Path
import numpy as np

# Dynamically resolve root relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"
HIST_PATH = BASE_DIR / "hist.json"

_model = None
_histograms = None

def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model

def get_histograms():
    global _histograms
    if _histograms is None and HIST_PATH.exists():
        with open(HIST_PATH, "r") as f:
            _histograms = json.load(f)
    return _histograms

@api_view(['POST'])
def predict(request):
    data = request.data
    features = data.get("features")
    
    if not isinstance(features, list) or len(features) != 4 or not all(isinstance(x, (int, float)) for x in features):
        return Response({"error": "Invalid input schema"}, status=400)
        
    model = get_model()
    if not model:
        return Response({"error": "Model not loaded"}, status=500)
        
    pred = model.predict([features])[0]
    return Response({"prediction": int(pred)})

@api_view(['POST'])
def batch_drift(request):
    data = request.data
    batch = data.get("batch")
    
    if not isinstance(batch, list) or len(batch) == 0:
        return Response({"error": "Invalid input schema"}, status=400)
        
    for item in batch:
        features = item.get("features")
        if not isinstance(features, list) or len(features) != 4:
            return Response({"error": "Invalid input schema"}, status=400)

    hists = get_histograms()
    if not hists:
        return Response({"error": "Histograms not loaded"}, status=500)

    X_batch = np.array([item["features"] for item in batch])
    
    drift_metrics = {}
    is_drifted = False
    
    for i in range(4):
        feat_idx = str(i)
        hist_data = hists.get(feat_idx)
        if not hist_data:
            continue
            
        train_counts = np.array(hist_data["counts"])
        bin_edges = hist_data["bin_edges"]
        
        expected_pct = train_counts / np.sum(train_counts)
        
        # We need bin edges to identically cover numpy's inclusive right edges for the final bin
        actual_counts = np.zeros_like(train_counts)
        for j in range(len(train_counts)):
            if j == len(train_counts) - 1:
                # Include rightmost edge
                actual_counts[j] = np.sum((X_batch[:, i] >= bin_edges[j]) & (X_batch[:, i] <= bin_edges[j+1]))
            else:
                actual_counts[j] = np.sum((X_batch[:, i] >= bin_edges[j]) & (X_batch[:, i] < bin_edges[j+1]))
                
        actual_total = np.sum(actual_counts)
        if actual_total == 0:
            actual_pct = np.zeros_like(actual_counts, dtype=float)
        else:
            actual_pct = actual_counts / actual_total
            
        eps = 1e-6
        expected_pct_eps = expected_pct + eps
        actual_pct_eps = actual_pct + eps
        
        psi_per_bin = (actual_pct_eps - expected_pct_eps) * np.log(actual_pct_eps / expected_pct_eps)
        psi = np.sum(psi_per_bin)
        
        if math.isnan(psi) or math.isinf(psi):
            psi = 0.0

        drift_metrics[feat_idx] = float(psi)
        if psi > 0.1:
            is_drifted = True
            
    return Response({
        "drift_metrics": drift_metrics,
        "is_drifted": is_drifted
    })
INNER_EOF

