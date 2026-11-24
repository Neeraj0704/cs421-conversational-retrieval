"""Run one existing CLI stage only when this process can really use Apple MPS."""

import os
import runpy
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
from gpu_probe import probe

check = probe()
if not check["gpu_computation_passed"]:
    raise SystemExit(
        "GPU is unavailable in this process. No CPU fallback stage was started."
    )
print("Confirmed GPU computation on", check["computation_device"], flush=True)
sys.argv = ["convrag", *sys.argv[1:]]
runpy.run_module("convrag.cli", run_name="__main__")
