import pandas as pd
import numpy as np
import os
import sys
import joblib
import yaml

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.logger.logging import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from src.utils.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    trained_model_type_path: str = os.path.join('artifacts', 'model_type.pkl')
    params_file_path: str = os.path.join('params.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.params = self.read_params()

    def read_params(self):
        try:
            with open(self.model_trainer_config.params_file_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.info(f"Parameters loaded from {self.model_trainer_config.params_file_path}")
            return params
        except Exception as e:
            logging.error("Error loading parameters from params.yaml")
            raise customexception(e, sys)

    def calculate_adjusted_r2(self, r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def initate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Model training started.')

            model_params = self.params.get('model_trainer', {})
            learning_rate = model_params.get('learning_rate', 0.05)
            max_depth = model_params.get('max_depth', 3)
            n_estimators = model_params.get('n_estimators', 100)
            subsample = model_params.get('subsample', 1.0)

            logging.info(f"Model Params - learning_rate: {learning_rate}, max_depth: {max_depth}, "
                         f"n_estimators: {n_estimators}, subsample: {subsample}")

            model = GradientBoostingRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                subsample=subsample,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            adjusted_r2 = self.calculate_adjusted_r2(r2, X_test.shape[0], X_test.shape[1])

            logging.info(f'R2: {r2:.4f}, Adjusted R2: {adjusted_r2:.4f}, MSE: {mse:.4f}')
            print(f"R2: {r2:.4f}, Adjusted R2: {adjusted_r2:.4f}, MSE: {mse:.4f}")

            save_object(self.model_trainer_config.trained_model_file_path, model)
            joblib.dump("GradientBoostingRegressor", self.model_trainer_config.trained_model_type_path)
            logging.info("Model and model type saved successfully.")

        except Exception as e:
            logging.error('Exception occurred during model training.')
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        X_train = np.load('artifacts/X_train.npy')
        X_test = np.load('artifacts/X_test.npy')
        y_train = np.load('artifacts/y_train.npy')
        y_test = np.load('artifacts/y_test.npy')

        trainer = ModelTrainer()
        trainer.initate_model_training(X_train, X_test, y_train, y_test)
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise customexception(e, sys)