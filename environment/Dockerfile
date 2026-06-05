FROM python:3.11-slim@sha256:67e6a6053f28db54c173ad84a4bf88fdd4e338793dc09672e87ee38e3b1b378c

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    tmux \
    asciinema \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock

COPY cli/ /app/environment/cli/
COPY data/ /app/environment/data/
COPY django_project/ /app/environment/django_project/
