import os
import pickle
import yaml
import pandas as pd
import mlflow
import boto3
import s3fs

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from mlflow.models import infer_signature

MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def open_df(data_path):
    if data_path.startswith("s3"):
        data = pd.read_csv(data_path)
    else:
        data = pd.read_csv(data_path)
    return data


def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=2,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def train(data_path, model_path, random_state, n_estimators, max_depth):
    print("STAGE TRAIN : Starting the training process : ")

    data = open_df(data_path)
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"]

    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state
        )

        signature = infer_signature(X_train, y_train)

        param_grid = {
            "min_samples_leaf": [1, 2]
        }

        mlflow.set_tag("mlflow.runName", "Best RandomForestClassifier")
        mlflow.set_tag("experiment_type", "grid_search_cv for best hyperparameters")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            "RandomForestClassifier with GridSearchCV For Hyperparameter Tuning"
        )

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy:{accuracy}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        cm = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(classification_rep), "classification_report.txt")

        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        mlflow.sklearn.log_model(
            best_model,
            MODEL_NAME,
            registered_model_name=MODEL_NAME,
            signature=signature
        )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)

        print(f"Model successfully saved locally to: {model_path}")


if __name__ == "__main__":
    params_path = os.path.join(BASE_DIR, "params.yaml")
    with open(params_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    train_params = params["train"]
    mlflow_params = params["mlflow"]
    aws_params = params["aws"]

    os.environ["AWS_DEFAULT_REGION"] = aws_params["region_name"]

    train(
        train_params["data"],
        train_params["model_path"],
        train_params["random_state"],
        train_params["n_estimators"],
        train_params["max_depth"]
    )