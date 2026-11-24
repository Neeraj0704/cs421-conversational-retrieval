"""Reproduce results in an isolated directory without overwriting recorded experiments."""

from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="core_reproduction")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Quick independent BM25 reproduction",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Use letters, numbers, hyphens and underscores for the name")
    dest = ROOT / "reproductions" / args.name
    if dest.exists():
        raise ValueError("Reproduction directory already exists; use a new --name")
    dest.mkdir(parents=True)
    for name in ["src", "scripts", "configs", "splits", "tests"]:
        shutil.copytree(
            ROOT / name,
            dest / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ["pyproject.toml", "uv.lock"]:
        shutil.copy2(ROOT / name, dest / name)
    (dest / "report").mkdir()
    shutil.copy2(ROOT / "report/export_pdf.py", dest / "report/export_pdf.py")
    (dest / "analysis").mkdir()
    for name in [
        "model_inventory.json",
        "frozen_selection.json",
        "feasibility_dense.json",
        "feasibility_splade.json",
    ]:
        source = ROOT / "analysis" / name
        if source.exists():
            shutil.copy2(source, dest / "analysis" / name)
    for relative in ["data/upstream", "indexes", "cache/huggingface"]:
        source = ROOT / relative
        if source.exists():
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
    # Preserve the exact generated query text; reranker inference is deliberately recomputed.
    for source in (ROOT / "artifacts/queries").glob("*.jsonl"):
        target = dest / "cache/rewrites" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def run(*command):
        subprocess.run([sys.executable, *command], cwd=dest, check=True)

    if args.baseline_only:
        run("-m", "convrag.cli", "prepare")
        run("-m", "convrag.cli", "index", "--backend", "bm25")
        run("-m", "convrag.cli", "run", "--config", "configs/E00_bm25_latest.json")
    else:
        run("scripts/run_core.py", "--include-final")
        run("report/export_pdf.py")
    comparison = {}
    for p in sorted((dest / "runs").glob("*/metrics.json")):
        original = ROOT / "runs" / p.parent.name / "metrics.json"
        if not original.exists():
            continue
        before, after = json.loads(original.read_text()), json.loads(p.read_text())
        delta = {
            k: after["mean_over_scored_queries"][k] - v
            for k, v in before["mean_over_scored_queries"].items()
        }
        comparison[p.parent.name] = {
            "metric_deltas": delta,
            "maximum_absolute_delta": max(abs(v) for v in delta.values()),
        }
    (dest / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("Reproduction completed:", dest)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
