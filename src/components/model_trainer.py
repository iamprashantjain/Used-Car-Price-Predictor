# import os
# import sys
# import pandas as pd
# import joblib
# import yaml
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from src.exception.exception import customexception
# from src.logger.logging import logging


# def load_params(params_file="params.yaml"):
#     try:
#         with open(params_file, 'r') as f:
#             params = yaml.safe_load(f)
#         logging.info("Model parameters loaded from params.yaml")
#         return params["model_training"]["params"]
#     except Exception as e:
#         logging.error("Failed to load parameters from YAML", exc_info=True)
#         raise customexception(e, sys)


# def load_data(X_train_path, y_train_path, X_test_path, y_test_path):
#     try:
#         X_train = pd.read_csv(X_train_path)
#         y_train = pd.read_csv(y_train_path).squeeze()
#         X_test = pd.read_csv(X_test_path)
#         y_test = pd.read_csv(y_test_path).squeeze()
#         logging.info("Model training: Data loaded successfully.")
#         return X_train, y_train, X_test, y_test
#     except Exception as e:
#         logging.error("Failed to load data", exc_info=True)
#         raise customexception(e, sys)


# def train_model(X_train, y_train, params):
#     try:
#         model = RandomForestRegressor(
#             n_estimators=params['n_estimators'],
#             max_depth=params['max_depth'],
#             min_samples_split=params['min_samples_split'],
#             random_state=42,
#             n_jobs=-1
#         )
#         model.fit(X_train, y_train)
#         logging.info("Model trained successfully.")
#         return model
#     except Exception as e:
#         logging.error("Model training failed", exc_info=True)
#         raise customexception(e, sys)


# def evaluate_model(model, X_test, y_test):
#     try:
#         preds = model.predict(X_test)
#         mse = mean_squared_error(y_test, preds)
#         r2 = r2_score(y_test, preds)
#         logging.info(f"Model Evaluation: MSE={mse:.2f}, R2={r2:.2f}")
#         print(f"Model Evaluation Results:\nMSE: {mse:.2f}\nR2: {r2:.2f}")
#         return mse, r2
#     except Exception as e:
#         logging.error("Model evaluation failed", exc_info=True)
#         raise customexception(e, sys)


# def save_model(model, output_path):
#     try:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         joblib.dump(model, output_path)
#         logging.info(f"Model saved to {output_path}")
#     except Exception as e:
#         logging.error("Saving model failed", exc_info=True)
#         raise customexception(e, sys)


# if __name__ == "__main__":
#     X_train_path = "artifacts/data_transformation/X_train_transformed.csv"
#     y_train_path = "artifacts/data_transformation/y_train.csv"
#     X_test_path = "artifacts/data_transformation/X_test_transformed.csv"
#     y_test_path = "artifacts/data_transformation/y_test.csv"
#     model_output_path = "artifacts/model/model.pkl"

#     params = load_params("params.yaml")

#     X_train, y_train, X_test, y_test = load_data(X_train_path, y_train_path, X_test_path, y_test_path)

#     model = train_model(X_train, y_train, params)

#     evaluate_model(model, X_test, y_test)

#     save_model(model, model_output_path)

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
