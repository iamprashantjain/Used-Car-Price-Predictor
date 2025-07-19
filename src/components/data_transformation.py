import joblib
import pandas as pd
import numpy as np
import os
import ast
import logging
import sys
import json

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

from src.exception.exception import customexception
from src.logger.logging import logging


def read_data(source_type, train_path, test_path):
    if source_type == 'path':
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Data read successfully from {train_path} and {test_path}")
            return train_df, test_df
        except Exception as e:
            logging.info("Failed to read data", exc_info=True)
            raise customexception(e, sys)
    else:
        logging.warning("Other source type not configured yet")


def data_cleaning(df):
    try:
        df['registrationDate'] = pd.to_datetime(df['registrationDate'], unit='ms', errors='coerce').dt.date
        df['odometer'] = df['odometer'].apply(lambda x: ast.literal_eval(x)['value'] if pd.notna(x) else np.nan)

        df['emiDetails'] = df['emiDetails'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
        emi_df = pd.json_normalize(df['emiDetails'])
        df = df.drop(columns='emiDetails').join(emi_df)

        df['transmissionType'] = df['transmissionType'].apply(
            lambda x: ast.literal_eval(x)['value'] if pd.notna(x) else np.nan)

        df['features'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        mlb = MultiLabelBinarizer()
        features_encoded = pd.DataFrame(mlb.fit_transform(df['features']),
                                        columns=mlb.classes_, index=df.index)
        df = df.join(features_encoded)
        df['featureCount'] = df['features'].apply(len)

        df['avgEmi'] = df[['emiStartingValue', 'emiEndingValue']].mean(axis=1)

        df.drop(columns=[
            'appointmentId', 'carName', 'modelGroup', 'features', 'displayText',
            'notAvailableText', 'tenure', 'registrationDate'], errors='ignore', inplace=True)

        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['ownership'] = df['ownership'].map({'Owned': 1, 'Leased': 0})

        bool_columns = [
            '360DegreeCamera', 'AlloyWheels', 'AppleCarplayAndroidAuto', 'Bluetooth',
            'CruiseControl', 'GpsNavigation', 'InfotainmentSystem', 'LeatherSeats',
            'ParkingAssist', 'PushButtonStart', 'RearAc', 'SpecialRegNo', 'Sunroof/Moonroof',
            'TopModel', 'Tpms', 'VentilatedSeats'
        ]
        for col in bool_columns:
            if col not in df.columns:
                df[col] = False
        df[bool_columns] = df[bool_columns].applymap(lambda x: 1 if x else 0)

        df.drop(columns=[
            'cityRto', 'cashDownPayment', 'emiStartingValue', 'emiEndingValue',
            'roiMinDiscounted', 'roiMaxDiscounted', 'roiMinOriginal',
            'roiMaxOriginal', 'emiOriginalStartingValue', 'emiOriginalEndingValue',
        ], errors='ignore', inplace=True)

        logging.info(f"Columns after cleaning: {df.columns.tolist()}")
        return df
    except Exception as e:
        logging.info("Error during data cleaning", exc_info=True)
        raise customexception(e, sys)


def perform_transformation(train_df, test_df, save_path="artifacts"):
    try:
        target_col = 'listingPrice'
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Detect column types
        numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

        # Pipelines
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

        # Transform
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # RFECV Feature Selection
        rfecv = RFECV(estimator=RandomForestRegressor(n_jobs=-1), step=1, cv=2, scoring='r2')
        rfecv.fit(X_train_transformed, y_train)

        # Apply selection
        X_train_selected = X_train_transformed[:, rfecv.support_]
        X_test_selected = X_test_transformed[:, rfecv.support_]

        # Save preprocessor and feature mask
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(rfecv.support_, os.path.join(save_path, "preprocessor/selected_features.pkl"))

        logging.info("Data transformation and feature selection completed.")
        return X_train_selected, y_train, X_test_selected, y_test, preprocessor

    except Exception as e:
        logging.info("Error during data transformation", exc_info=True)
        raise customexception(e, sys)


def save_transformed_data(X_train, y_train, X_test, y_test, train_data_dir, test_data_dir):
    try:
        os.makedirs(train_data_dir, exist_ok=True)
        os.makedirs(test_data_dir, exist_ok=True)

        train_df = pd.DataFrame(X_train)
        train_df['listingPrice'] = y_train.values
        test_df = pd.DataFrame(X_test)
        test_df['listingPrice'] = y_test.values

        train_df.to_csv(os.path.join(train_data_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(test_data_dir, "test.csv"), index=False)

        logging.info("Transformed datasets saved.")
    except Exception as e:
        logging.info("Error saving transformed data", exc_info=True)
        raise customexception(e, sys)


def save_preprocessor(preprocessor, preprocessor_dir="artifacts/preprocessor"):
    try:
        os.makedirs(preprocessor_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(preprocessor_dir, "preprocessor.pkl"))
        logging.info("Preprocessor saved.")
    except Exception as e:
        logging.info("Error saving preprocessor", exc_info=True)
        raise customexception(e, sys)


# MAIN ENTRY POINT
if __name__ == "__main__":
    train_path = r"artifacts/data_ingestion/train.csv"
    test_path = r"artifacts/data_ingestion/test.csv"

    train_df, test_df = read_data('path', train_path, test_path)
    train_df = data_cleaning(train_df)
    test_df = data_cleaning(test_df)
    
    train_df.to_csv(f"{preprocessor_dir}/train.csv", index=False)
    test_df.to_csv(f"{preprocessor_dir}/train.csv", index=False)

    X_train_selected, y_train, X_test_selected, y_test, preprocessor = perform_transformation(
        train_df, test_df,
        save_path="artifacts"
    )

    save_preprocessor(preprocessor)