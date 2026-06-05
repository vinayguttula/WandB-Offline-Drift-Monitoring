#!/usr/bin/env bash
set -uo pipefail

# Ensure tests are run inside the exact directory pytest.ini expects
cd /app

# Install test dependencies offline
pip install --no-index --find-links=/tmp/test-wheels pytest pytest-django || true

# Failsafe if pip fails finding links
if ! command -v pytest &> /dev/null; then
    pip install pytest pytest-django
fi

# We explicitly execute targeting the /app/tests files
PYTHONPATH=/app/environment/django_project pytest -c /app/tests/pytest.ini -v /app/tests/test_outputs.py
