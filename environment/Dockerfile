FROM python:3.11-slim@sha256:67e6a6053f28db54c173ad84a4bf88fdd4e338793dc09672e87ee38e3b1b378c

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tmux \
    asciinema \
    && rm -rf /var/lib/apt/lists/*

# Install core production AND verifier dependencies
RUN pip install --no-cache-dir \
    django==5.0.6 \
    djangorestframework==3.15.1 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    wandb==0.17.0 \
    joblib==1.4.2 \
    pytest==8.2.1 \
    pytest-django==4.8.0

# Copy application environment files
COPY . /app/environment/
