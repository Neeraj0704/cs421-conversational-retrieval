#!/bin/bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
if [ -x "$project_root/.venv/bin/python" ]; then
    project_python="$project_root/.venv/bin/python"
elif [ -x "$project_root/../.venv/bin/python" ]; then
    project_python="$project_root/../.venv/bin/python"
else
    echo "Install the project environment described in README.md first."
    exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK=0
echo "Running three frozen GPU final comparisons. No indexing or rewriting is required."
echo "This is a follow-up benchmark on the public test set already used for CPU results."
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$project_python" scripts/run_gpu_final.py
else
    exec "$project_python" scripts/run_gpu_final.py
fi
