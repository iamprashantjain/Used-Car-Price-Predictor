FROM python:3.10-slim

WORKDIR /app

# Install system dependencies only if absolutely needed
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the Flask app and templates folder
COPY app.py /app/
COPY templates/ /app/templates/
COPY requirements_prod.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_prod.txt

# DVC pull handled in Dockerfile, preprocessor must be configured in DVC
RUN pip install --no-cache-dir dvc[s3] && \
    dvc pull artifacts/data_transformation/preprocessor.pkl && \
    pip uninstall -y dvc

EXPOSE 5000

CMD ["python", "app.py"]
