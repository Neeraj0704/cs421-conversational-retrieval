"""Freeze a bounded final comparison before looking at MTRAG-UN scores."""

from pathlib import Path
import argparse
import json
import subprocess
from convrag.io import write_json, fingerprint, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Development IDs selected for final testing",
    )
    args = parser.parse_args()
    root = Path.cwd()
    target = root / "analysis/frozen_selection.json"
    if target.exists():
        raise ValueError("Selection already frozen; preserve the original decision")
    final = {}
    dev = {}
    for rid in args.runs:
        cfg = json.loads((root / "configs" / (rid + ".json")).read_text())
        metrics = json.loads((root / "runs" / rid / "metrics.json").read_text())
        if metrics["dataset"] != "mtrag":
            raise ValueError("Select only from development runs")
        dev[rid] = {
            "ndcg_at_5": metrics["mean_over_scored_queries"]["ndcg_cut_5"],
            "config_sha256": fingerprint(cfg),
        }
        cfg.update(id="FINAL_" + rid, dataset="mtragun", split="all")
        if cfg.get("rewrites"):
            cfg["rewrites"] = "cache/rewrites/mtragun.jsonl"
        write_json(root / "configs" / (cfg["id"] + ".json"), cfg)
        final[cfg["id"]] = fingerprint(cfg)
    sources = {str(p): file_hash(p) for p in sorted(Path("src").rglob("*.py"))}
    write_json(
        target,
        {
            "selection_rule": "Prespecified baseline and validation-selected candidate comparisons",
            "development_results": dev,
            "final_configs": final,
            "source_sha256": sources,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "model_revisions": json.loads(
                Path("cache/model_revisions.json").read_text()
            )
            if Path("cache/model_revisions.json").exists()
            else {},
            "final_qrels_used_for_selection": False,
        },
    )
    print("Frozen:", ", ".join(final))


if __name__ == "__main__":
    main()
