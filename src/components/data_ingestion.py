import yaml
from datetime import datetime
import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from src.utils.utils import fetch_data
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self, source_type, file_path_or_query, db_url=None, s3_bucket=None, s3_key=None):
        self.ingestion_config = DataIngestionConfig()
        self.source_type = source_type
        self.file_path_or_query = file_path_or_query
        self.db_url = db_url
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        
        # Load params.yaml
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)
        
        self.test_size = self.params['data_ingestion']['test_size']
        

    def initial_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Data cleaning started")
        df.drop_duplicates(inplace=True)
        return df

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = fetch_data(
                source_type=self.source_type,
                file_path_or_query=self.file_path_or_query,
                db_url=self.db_url,
                s3_bucket=self.s3_bucket,
                s3_key=self.s3_key
            )
            logging.info(f"Data fetched successfully from {self.source_type}")

            data = self.initial_data_cleaning(data)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed and train-test split saved.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise customexception(e, sys)



if __name__ == "__main__":
    obj = DataIngestion(source_type='local',file_path_or_query='data/House_Rent_Dataset.csv')
    obj.initiate_data_ingestion()