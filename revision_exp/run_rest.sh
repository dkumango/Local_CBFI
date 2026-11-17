#!/bin/bash
# Runs the real-data experiments one after another once the synthetic benchmark (pid $1) has finished.
cd "$(dirname "$0")"
PY=~/CBFI_revision/.venv/bin/python
if [ -n "$1" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
for s in exp_cases exp_multidomain exp_stability exp_scalability; do
  echo "=== $s start $(date)" >> run_rest.log
  $PY -u $s.py > $s.log 2>&1
  echo "=== $s exit=$? $(date)" >> run_rest.log
done
echo "=== ALL DONE $(date)" >> run_rest.log
