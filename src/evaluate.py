import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
import json  # Import json

# Load params
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")
params = yaml.safe_load(open(params_path))

train_params = params['train']
mflow_params = params['mlflow']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"]

    mlflow.set_tracking_uri(mflow_params["MLFLOW_TRACKING_URI"])
    model_name = "Best_RandomForestClassifier"
    model_version = "latest"

    # Load model
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # 1. Log to MLflow
    mlflow.log_metric("eval_accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")

    # 2. SAVE METRICS LOCALLY FOR DVC
    # DVC requires this file to exist to satisfy the 'outs' contract
    metrics = {"accuracy": accuracy}
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname("metrics/metrics.json"), exist_ok=True)
    
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Metrics saved to metrics/metrics.json")

if __name__ == "__main__":
    evaluate(train_params["data"], train_params["model_path"])