import os
import sys
import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.exception.exception import customexception
from src.logger.logging import logging


def load_params(params_file="params.yaml"):
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    logging.info("Model parameters loaded from params.yaml")
    return params["model_training"]["params"]


def load_data(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    logging.info("Training data loaded successfully.")
    return X_train, y_train


def train_and_save_model(X_train, y_train, params, output_path):
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info(f"Model trained and saved to {output_path}")


if __name__ == "__main__":
    try:
        params = load_params("params.yaml")
        X_train, y_train = load_data(
            "artifacts/data_transformation/X_train_transformed.csv",
            "artifacts/data_transformation/y_train.csv"
        )
        train_and_save_model(X_train, y_train, params, "artifacts/model/model.pkl")
    except Exception as e:
        raise customexception(e, sys)
