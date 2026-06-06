#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier || true

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"

cd "$ROOT_DIR"

pip install --quiet --no-index --find-links=/tmp/test-wheels pytest==8.2.1 pytest-django==4.8.0 playwright==1.58.0 pluggy==1.6.0 iniconfig==2.3.0 packaging==26.2

PYTHONPATH="$ROOT_DIR/environment/django_project" python -m pytest -c "$DIR/pytest.ini" -v "$DIR/test_outputs.py" -rA
rc=$?
if [ "$rc" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
