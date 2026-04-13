from fastapi import FastAPI
import pandas as pd
import mlflow
import os
import yaml
from contextlib import asynccontextmanager
import traceback

MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))
mlflow_params = params['mlflow']

mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("mlflow tracking URI set to:", mlflow.get_tracking_uri(), flush=True)
    print("Chargement du modèle depuis MLflow Registry...", flush=True)
    print("Model URI:", f"models:/{MODEL_NAME}/{MODEL_VERSION}", flush=True)

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        )
        print("✅ Modèle chargé avec succès", flush=True)

    except Exception as e:
        print("❌ ERROR loading model:", str(e), flush=True)
        traceback.print_exc()

    yield  # app runs here

    print("🔻 Shutting down app", flush=True)


app = FastAPI(
    title="API de prédiction ML",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "API MLflow opérationnelle"}


@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}
