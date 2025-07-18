import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    X_train_path: str = os.path.join("artifacts", "X_train.npy")
    X_test_path: str = os.path.join("artifacts", "X_test.npy")
    y_train_path: str = os.path.join("artifacts", "y_train.npy")
    y_test_path: str = os.path.join("artifacts", "y_test.npy")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def feature_engineering(self, df: pd.DataFrame):
        try:
            logging.info("Performing feature engineering...")
            df[['current_floor', 'total_floors']] = df['Floor'].str.extract(r'(\d+).*?(\d+)').astype(float)
            df.drop(['Posted On', 'Point of Contact', 'Floor', 'Area Locality'], axis=1, inplace=True)
            return df
        except Exception as e:
            logging.error("Error in feature engineering")
            raise customexception(e, sys)

    def get_preprocessor(self):
        try:
            categorical_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
            numerical_cols = ['BHK', 'Size', 'Bathroom', 'current_floor', 'total_floors']

            logging.info(f"Categorical Columns: {categorical_cols}")
            logging.info(f"Numerical Columns: {numerical_cols}")

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error in get_preprocessor")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            X_train = train_df.drop('Rent', axis=1)
            y_train = train_df['Rent']

            X_test = test_df.drop('Rent', axis=1)
            y_test = test_df['Rent']

            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info(f"Saving transformed data and preprocessor...")

            # Save preprocessor
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            # Save numpy arrays
            np.save(self.config.X_train_path, X_train_transformed)
            np.save(self.config.X_test_path, X_test_transformed)
            np.save(self.config.y_train_path, y_train.to_numpy())
            np.save(self.config.y_test_path, y_test.to_numpy())

            logging.info("Data transformation completed successfully.")

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        obj = DataTransformation()
        obj.initiate_data_transformation("artifacts/train.csv", "artifacts/test.csv")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")