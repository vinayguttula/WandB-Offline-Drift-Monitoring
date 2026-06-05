#!/usr/bin/env bash
set -uo pipefail

if [ -d "/app/environment" ]; then
    cd /app/environment
elif [ -d "/app" ]; then
    cd /app
fi

pip install --no-index --find-links=/tmp/test-wheels pytest pytest-django || true
if ! command -v pytest &> /dev/null; then
    pip install pytest pytest-django
fi

PYTHONPATH=$(pwd)/django_project pytest -v tests/test_outputs.py || \
PYTHONPATH=/app/environment/django_project pytest -v /app/tests/test_outputs.py || \
PYTHONPATH=$(pwd)/environment/django_project pytest -v tests/test_outputs.py
