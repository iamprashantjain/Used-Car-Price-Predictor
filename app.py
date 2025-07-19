from flask import Flask, request, render_template
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


def download_preprocessor_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    if not os.path.exists(local_path):
        s3.download_file(bucket, key, local_path)
        print(f"✅ Preprocessor downloaded from S3 to {local_path}")
    else:
        print("✅ Preprocessor already exists locally.")


def load_preprocessor():
    download_preprocessor_from_s3(S3_BUCKET, S3_PREPROCESSOR_KEY, LOCAL_PREPROCESSOR_PATH)
    return joblib.load(LOCAL_PREPROCESSOR_PATH)


def load_model_from_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_STAGE}"
    print(f"✅ Loading model from MLflow: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


preprocessor = load_preprocessor()
model = load_model_from_mlflow()

INPUT_COLUMNS = [
    "make", "model", "variant", "year", "transmissionType", "bodyType", "fuelType",
    "ownership", "color", "odometer", "fitnessAge", "cashDownPayment",
    "emiStartingValue", "emiEndingValue", "roiMinDiscounted", "roiMaxDiscounted",
    "roiMinOriginal", "roiMaxOriginal", "emiOriginalStartingValue", "emiOriginalEndingValue",
    "360DegreeCamera", "AlloyWheels", "AppleCarplayAndroidAuto", "Bluetooth",
    "CruiseControl", "GpsNavigation", "InfotainmentSystem", "LeatherSeats",
    "ParkingAssist", "PushButtonStart", "RearAc", "SpecialRegNo", "Sunroof/Moonroof",
    "TopModel", "Tpms", "VentilatedSeats", "featureCount", "avgEmi"
]


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        input_data = {col: request.form.get(col) for col in INPUT_COLUMNS}

        try:
            input_df = pd.DataFrame([input_data])
            # Cast numerics where required (adjust based on your data)
            numeric_cols = ["year", "odometer", "fitnessAge", "cashDownPayment",
                            "emiStartingValue", "emiEndingValue", "roiMinDiscounted", "roiMaxDiscounted",
                            "roiMinOriginal", "roiMaxOriginal", "emiOriginalStartingValue", "emiOriginalEndingValue",
                            "360DegreeCamera", "AlloyWheels", "AppleCarplayAndroidAuto", "Bluetooth",
                            "CruiseControl", "GpsNavigation", "InfotainmentSystem", "LeatherSeats",
                            "ParkingAssist", "PushButtonStart", "RearAc", "SpecialRegNo", "Sunroof/Moonroof",
                            "TopModel", "Tpms", "VentilatedSeats", "featureCount", "avgEmi"]
            for col in numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            transformed_data = preprocessor.transform(input_df)
            prediction = model.predict(transformed_data)[0]

        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
