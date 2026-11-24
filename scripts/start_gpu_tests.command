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
"$project_python" scripts/gpu_probe.py --require-mps --output analysis/gpu/access_launcher.json
"$project_python" scripts/gpu_benchmark.py
echo "GPU feasibility checks completed. Full indexing has not been started."
