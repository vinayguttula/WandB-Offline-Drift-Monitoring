FROM python:3.11-slim@sha256:67e6a6053f28db54c173ad84a4bf88fdd4e338793dc09672e87ee38e3b1b378c

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    tmux \
    asciinema \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock

# Download test wheels to cache to follow no-internet testing rules
RUN pip download -d /tmp/test-wheels pytest==8.2.1 pytest-django==4.8.0 playwright==1.58.0 pluggy==1.6.0 iniconfig==2.3.0 packaging==26.2

COPY cli/ /app/environment/cli/
COPY data/ /app/environment/data/
COPY django_project/ /app/environment/django_project/
