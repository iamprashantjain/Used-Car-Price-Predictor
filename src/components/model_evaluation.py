import os
import sys
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
from src.exception.exception import customexception
from src.logger.logging import logging


def evaluate_and_log(model_path, X_test_path, y_test_path, mlflow_uri, experiment_name="BaseLine Model"):
    try:
        mlflow.set_tracking_uri(mlflow_uri)

        # Check experiment existence
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist. Please create it on Dagshub.")
            sys.exit(1)
        else:
            mlflow.set_experiment(experiment_name)

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()
        model = joblib.load(model_path)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        logging.info(f"Evaluation completed: MSE={mse:.2f}, R2={r2:.2f}")
        print(f"Evaluation completed:\nMSE: {mse:.2f}, R2: {r2:.2f}")

        with mlflow.start_run():
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, "model")

    except Exception as e:
        logging.error("Model evaluation failed", exc_info=True)
        raise customexception(e, sys)



if __name__ == "__main__":
    evaluate_and_log(
        model_path="artifacts/model/model.pkl",
        X_test_path="artifacts/data_transformation/X_test_transformed.csv",
        y_test_path="artifacts/data_transformation/y_test.csv",
        mlflow_uri="https://dagshub.com/iamprashantjain/Used-Car-Price-Predictor.mlflow"
    )

