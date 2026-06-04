#!/usr/bin/env bash
set -uo pipefail

# Install test dependencies offline
pip install --no-index --find-links=/tmp/test-wheels pytest pytest-django

# Run the test suite with pytest
pytest -v tests/test_outputs.py
