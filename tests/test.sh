#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"

cd "$ROOT_DIR"

PYTHONPATH="$ROOT_DIR/environment/django_project" python -m pytest -c "$DIR/pytest.ini" -v "$DIR/test_outputs.py" -rA
rc=$?

if [ "$rc" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
