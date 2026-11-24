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
echo "Starting full GPU indexing and twelve validation comparisons."
echo "Estimated indexing from the sample: dense ~35 min; SPLADE ~2.6 hours."
echo "Full-run time can differ. Keep the Mac plugged in and its lid open."
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$project_python" scripts/run_gpu_study.py
else
    exec "$project_python" scripts/run_gpu_study.py
fi
