from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import time

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='iamprashantjain', repo_name='Used-Car-Price-Predictor', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/Used-Car-Price-Predictor.mlflow")

client = MlflowClient()

try:
    # Define experiment name and model name
    experiment_name = "BaseLine Model"
    model_name = "used-car-price-regressor"

    # Fetch experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # Get latest successful run (descending order of time)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs or len(runs) == 0:
        raise ValueError("No valid runs found to register model from.")

    run_id = runs[0].info.run_id
    print(f"Using Run ID: {run_id}")

    # Model artifact path logged during evaluation
    model_path = "model"
    model_uri = f"runs:/{run_id}/{model_path}"

    # Register model
    registration = mlflow.register_model(model_uri, model_name)
    print(f"Model registered successfully with version: {registration.version}")

    # Wait for backend to process registration
    time.sleep(5)

    # Optional: Add description and metadata
    client.update_model_version(
        name=model_name,
        version=registration.version,
        description="RandomForestRegressor model predicting used car price with MSE and R² metrics logged via MLflow."
    )

    client.set_model_version_tag(
        name=model_name,
        version=registration.version,
        key="author",
        value="prashantjain"
    )

    # Move model to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=registration.version,
        stage="Staging",
        archive_existing_versions=True
    )

    print(f"Model '{model_name}' version {registration.version} is now in 'Staging' stage.")

except Exception as e:
    print(f"Error during model registration: {e}")
