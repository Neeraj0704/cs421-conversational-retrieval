"""MPS-only feasibility and CPU/GPU numerical checks. No relevance labels are read."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["splade", "dense"],
        default=["splade", "dense"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--samples-per-domain", type=int, default=32)
    args = parser.parse_args()
    if any(n < 1 for n in args.batch_sizes) or args.samples_per_domain < 1:
        raise ValueError("Sample and batch sizes must be positive")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    from gpu_probe import probe

    diagnostic = probe()
    if not diagnostic["gpu_computation_passed"]:
        print(json.dumps(diagnostic, indent=2))
        raise SystemExit("MPS is unavailable; refusing to benchmark on CPU.")
    import numpy as np
    import torch
    from convrag.data import corpus, COLLECTIONS
    from convrag.neural import Encoder
    from convrag.io import write_json

    manifest = json.loads((ROOT / "analysis/data_manifest.json").read_text())
    texts = []
    for domain in COLLECTIONS:
        count = manifest["domains"][domain]["passages"]
        positions = set(
            np.linspace(
                0, count - 1, min(args.samples_per_domain, count), dtype=int
            ).tolist()
        )
        texts.extend(
            p["text"]
            for i, p in enumerate(corpus(ROOT / "data/upstream", domain))
            if i in positions
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "analysis/gpu" / stamp
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "access.json", diagnostic)
    for backend in args.backends:
        print("Loading", backend, "on MPS", flush=True)
        encoder = Encoder(ROOT, backend)
        if encoder.device != "mps":
            raise RuntimeError("Encoder did not select MPS")
        # Full query vectors avoid false pruning differences at the document top-128 boundary.
        gpu = encoder.encode(texts[:2], query=True)
        encoder.model.to("cpu")
        encoder.device = "cpu"
        cpu = encoder.encode(texts[:2], query=True)
        encoder.model.to("mps")
        encoder.device = "mps"
        if backend == "splade":
            gpu, cpu = gpu.toarray(), cpu.toarray()
        max_error = float(np.max(np.abs(cpu - gpu)))
        cosines = (cpu * gpu).sum(axis=1) / (
            np.linalg.norm(cpu, axis=1) * np.linalg.norm(gpu, axis=1) + 1e-12
        )
        parity = {
            "max_absolute_error": max_error,
            "minimum_cosine_similarity": float(cosines.min()),
            "passed": bool(max_error < 0.01 and cosines.min() > 0.999),
        }
        write_json(output / (backend + "_parity.json"), parity)
        if not parity["passed"]:
            raise RuntimeError("CPU/MPS numerical check failed; inspect " + str(output))
        rows = []
        for batch_size in args.batch_sizes:
            try:
                encoder.encode(texts[:batch_size])
                torch.mps.synchronize()
                started = time.perf_counter()
                for i in range(0, len(texts), batch_size):
                    encoder.encode(texts[i : i + batch_size])
                torch.mps.synchronize()
                seconds = time.perf_counter() - started
                row = {
                    "batch_size": batch_size,
                    "status": "passed",
                    "sample_passages": len(texts),
                    "seconds": seconds,
                    "passages_per_second": len(texts) / seconds,
                    "estimated_full_index_hours": manifest["total_passages"]
                    * seconds
                    / len(texts)
                    / 3600,
                    "current_allocated_bytes": torch.mps.current_allocated_memory(),
                }
                rows.append(row)
                print(backend, row, flush=True)
            except RuntimeError as error:
                rows.append(
                    {"batch_size": batch_size, "status": "failed", "error": str(error)}
                )
                torch.mps.empty_cache()
                break
        write_json(
            output / (backend + "_benchmark.json"),
            {
                "backend": backend,
                "model": encoder.model_id,
                "revision": encoder.revision,
                "device": "mps",
                "max_length": encoder.max_length,
                "fallback_enabled": False,
                "parity": parity,
                "results": rows,
                "scope": "Feasibility only; no qrels, no test scores, no full index",
            },
        )
        del encoder
        torch.mps.empty_cache()
    print("Saved GPU checks:", output, flush=True)


if __name__ == "__main__":
    main()
