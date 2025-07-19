import joblib
import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception.exception import customexception
from src.logger.logging import logging

def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()  # Convert to Series
        y_test = pd.read_csv(y_test_path).squeeze()
        logging.info("Data loaded successfully for transformation")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error loading data", exc_info=True)
        raise customexception(e, sys)


def data_transformation(X_train, X_test, y_train, y_test, save_path="artifacts/data_transformation"):
    try:
        os.makedirs(save_path, exist_ok=True)

        numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Save transformed datasets
        pd.DataFrame(X_train_transformed).to_csv(os.path.join(save_path, 'X_train_transformed.csv'), index=False)
        pd.DataFrame(X_test_transformed).to_csv(os.path.join(save_path, 'X_test_transformed.csv'), index=False)
        y_train.to_csv(os.path.join(save_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(save_path, 'y_test.csv'), index=False)

        # Save preprocessor
        joblib.dump(preprocessor, os.path.join(save_path, "preprocessor.pkl"))

        logging.info("Data transformation completed and preprocessor saved.")
        return (os.path.join(save_path, 'X_train_transformed.csv'),
                os.path.join(save_path, 'X_test_transformed.csv'),
                os.path.join(save_path, 'y_train.csv'),
                os.path.join(save_path, 'y_test.csv'),
                os.path.join(save_path, "preprocessor.pkl"))
    except Exception as e:
        logging.error("Data transformation failed", exc_info=True)
        raise customexception(e, sys)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(
        "artifacts/data_ingestion/X_train.csv",
        "artifacts/data_ingestion/X_test.csv",
        "artifacts/data_ingestion/y_train.csv",
        "artifacts/data_ingestion/y_test.csv"
    )
    X_train_file, X_test_file, y_train_file, y_test_file, preprocessor_file = data_transformation(
        X_train, X_test, y_train, y_test
    )
    print(f"Transformed files:\nX_train: {X_train_file}\nX_test: {X_test_file}\nPreprocessor: {preprocessor_file}")
