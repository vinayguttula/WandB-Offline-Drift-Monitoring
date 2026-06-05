#!/usr/bin/env bash
set -uo pipefail

# Ensure we are in the correct directory for absolute paths
cd /app

# Install test dependencies offline
pip install --no-index --find-links=/tmp/test-wheels pytest pytest-django

# Run the test suite with pytest
# Explicitly add django_project to PYTHONPATH so settings are found
PYTHONPATH=/app/environment/django_project pytest -v tests/test_outputs.py
