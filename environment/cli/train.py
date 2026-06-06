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
