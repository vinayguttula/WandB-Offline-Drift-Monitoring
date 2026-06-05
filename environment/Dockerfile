FROM python:3.11-slim@sha256:d8943ab07fbbdf4556a31f77d33d526e036e7880998a4a51e6dd2cd0f4a8618d

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

# Copy application files (to be mounted or copied in the runner)
COPY . /app/
