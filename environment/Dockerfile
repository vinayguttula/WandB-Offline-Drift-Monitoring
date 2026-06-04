FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install core production dependencies
RUN pip install --no-cache-dir \
    django==5.0.6 \
    djangorestframework==3.15.1 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    wandb==0.17.0 \
    joblib==1.4.2

# Pre-download test dependencies to a cache directory for offline installation
RUN mkdir -p /tmp/test-wheels && \
    pip download -d /tmp/test-wheels \
    pytest==8.2.1 \
    pytest-django==4.8.0

# Copy application files (to be mounted or copied in the runner)
COPY . /app/
