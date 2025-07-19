import sys
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import yaml
from io import BytesIO
from src.exception.exception import customexception
from src.logger.logging import logging

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["data_ingestion"]["test_size"]
random_state = params["data_ingestion"]["random_state"]
input_file_path = params["data_ingestion"]["input_file_path"]

def read_data_s3(path):
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path, storage_options={"anon": False})
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path, storage_options={"anon": False})
        else:
            raise ValueError("Unsupported file format for S3 path.")
        logging.info(f"Data read successfully from S3: {path}")
        return df
    except Exception as e:
        logging.error("Failed to read data from S3", exc_info=True)
        raise customexception(e, sys)


def split_data(df, target_column, test_size, random_state):
    try:
        df.dropna(subset=[target_column], inplace=True)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(f"Train-test split done with test_size={test_size}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Train-test split failed", exc_info=True)
        raise customexception(e, sys)


def data_ingestion_pipeline(path, target_column, output_dir="artifacts/data_ingestion"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        df = read_data_s3(path)
        X_train, X_test, y_train, y_test = split_data(df, target_column, test_size, random_state)

        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        logging.info(f"Data ingestion completed. Files saved in {output_dir}")
        return (f"{output_dir}/X_train.csv", f"{output_dir}/X_test.csv",
                f"{output_dir}/y_train.csv", f"{output_dir}/y_test.csv")
    except Exception as e:
        logging.error("Data ingestion pipeline failed", exc_info=True)
        raise customexception(e, sys)


if __name__ == "__main__":
    target_column = params["data_ingestion"]["target_column"]
    X_train, X_test, y_train, y_test = data_ingestion_pipeline(input_file_path, target_column)
    print(f"Data Saved:\nX_train: {X_train}\nX_test: {X_test}\ny_train: {y_train}\ny_test: {y_test}")