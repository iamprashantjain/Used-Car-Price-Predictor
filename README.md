**Highlevel ML Project WorkFlow with MLOps**

1. Create Github Repo of the Project using Template
2. Clone repo on local system
3. Initialize dvc
    + dvc init
    + git add .dvc .gitignore
    + git commit -m "Initialize DVC"
    + dvc remote add -d prashant-mlops-bucket s3://prashant-mlops-bucket
4. upload data on s3
5. perform experiments and find best algorithm & parameters - set everything on mlflow
6. create dvc pipeline, save code on github, data on s3 via dvc, model on model registry
7. Apply CI: run dvc pipeline, perform tests, promote model in mlflow
8. create flask app which fetches promoted model from mlflow & make predictions on local system
9. Apply CD: dockerize app (for consistency so that code that works on my system will also work on cloud) & push to ECR & deploy on EC2 from ECR