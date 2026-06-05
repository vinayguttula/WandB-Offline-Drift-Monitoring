#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

# The Snorkel evaluation container strictly mounts the tests suite to /tests/
PYTHONPATH="/app/environment/django_project" python -m pytest -c /tests/pytest.ini -v /tests/test_outputs.py -rA
rc=$?

if [ "$rc" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
