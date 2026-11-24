"""Diagnose Apple GPU visibility without downloading models or changing settings."""

from __future__ import annotations
import argparse
import ctypes
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def probe():
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mps_built": False,
        "mps_available": False,
        "gpu_computation_passed": False,
    }
    if sys.platform == "darwin":
        try:
            metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
            metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
            result["native_metal_device_visible"] = bool(
                metal.MTLCreateSystemDefaultDevice()
            )
        except Exception as error:
            result["native_metal_error"] = str(error)
    try:
        import torch

        result.update(
            torch_version=torch.__version__,
            mps_built=torch.backends.mps.is_built(),
            mps_available=torch.backends.mps.is_available(),
        )
        if result["mps_available"]:
            started = time.perf_counter()
            x = torch.ones((256, 256), device="mps")
            y = x @ x
            torch.mps.synchronize()
            value = float(y[0, 0].cpu())
            result.update(
                gpu_computation_passed=value == 256.0,
                computation_device=str(y.device),
                check_seconds=time.perf_counter() - started,
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["next_step"] = (
        "Run the MPS-only model benchmark"
        if result["gpu_computation_passed"]
        else "GPU access is not confirmed in this process. Check host/session access before indexing; no CPU indexing was started."
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="analysis/gpu/access_session.json")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    result = probe()
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_mps and not result["gpu_computation_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
