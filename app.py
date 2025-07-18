from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import mlflow.pyfunc
import os
import boto3

app = Flask(__name__)

S3_BUCKET = 'prashant-mlops-bucket'
S3_PREPROCESSOR_KEY = 'artifacts/data_transformation/preprocessor.pkl'
LOCAL_PREPROCESSOR_PATH = 'preprocessor.pkl'

MLFLOW_TRACKING_URI = 'https://dagshub.com/iamprashantjain/Used-Car-Price-Predictor.mlflow'
MLFLOW_MODEL_NAME = 'used-car-price-regressor'
MLFLOW_STAGE = 'Production'


def load_preprocessor():
    preprocessor_path = 'artifacts/data_transformation/preprocessor.pkl'
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}. Please ensure it is downloaded via Docker or DVC before running the app.")
    print(f"Loading preprocessor from {preprocessor_path}")
    return joblib.load(preprocessor_path)


def load_model_from_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_STAGE}"
    print(f"Loading model from MLflow: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


preprocessor = load_preprocessor()
model = load_model_from_mlflow()

INPUT_COLUMNS = [
    "make", "model", "variant", "year", "transmissionType",
    "bodyType", "fuelType", "ownership", "color",
    "odometer", "fitnessAge", "featureCount"
]


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = {col: request.form[col] for col in INPUT_COLUMNS}

        # Convert types
        input_data['year'] = int(input_data['year'])
        input_data['ownership'] = int(input_data['ownership'])
        input_data['odometer'] = float(input_data['odometer'])
        input_data['fitnessAge'] = float(input_data['fitnessAge'])
        input_data['featureCount'] = int(input_data['featureCount'])

        input_df = pd.DataFrame([input_data])

        try:
            transformed_data = preprocessor.transform(input_df)
            prediction = model.predict(transformed_data)
            price = float(prediction[0])
            return render_template('index.html', prediction=price)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
