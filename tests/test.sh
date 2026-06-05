#!/usr/bin/env bash
set -uo pipefail

# Get the directory where test.sh is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"

# Navigate to the root directory
cd "$ROOT_DIR"

# Install test dependencies offline
pip install --no-index --find-links=/tmp/test-wheels pytest pytest-django || true

# Failsafe if pip fails finding links
if ! command -v pytest &> /dev/null; then
    pip install pytest pytest-django
fi

# We explicitly execute targeting the dynamically resolved tests folder
PYTHONPATH="$ROOT_DIR/environment/django_project" pytest -c "$DIR/pytest.ini" -v "$DIR/test_outputs.py"
