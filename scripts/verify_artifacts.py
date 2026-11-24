"""Independently check saved predictions, metrics and the frozen protocol."""

from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path
from convrag.data import corpus, qrels, COLLECTIONS
from convrag.io import read_jsonl, write_json, fingerprint, file_hash
from convrag.evaluation import evaluate, validate_predictions

ROOT = Path(__file__).resolve().parents[1]


def main():
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    match = re.search(r"(\d+) passed", test.stdout)
    upstream = ROOT / "data/upstream"
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    )
    if dirty.strip():
        raise ValueError("Pinned upstream checkout is modified: " + dirty)
    ids = {d: {p["id"] for p in corpus(upstream, d)} for d in COLLECTIONS}
    frozen = json.loads((ROOT / "analysis/frozen_selection.json").read_text())
    missing = [
        name
        for name in frozen["final_configs"]
        if not (ROOT / "runs" / name / "metrics.json").exists()
    ]
    if missing:
        raise ValueError("Final comparison is incomplete: " + ", ".join(missing))
    for path, expected in frozen["source_sha256"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError("Frozen source differs: " + path)
    runs = {}
    final_audit = {}
    for path in sorted((ROOT / "runs").glob("*/metrics.json")):
        folder = path.parent
        if not (folder / "official_format_check.txt").exists():
            continue
        if "Format is valid" not in (folder / "official_format_check.txt").read_text():
            raise ValueError("Official format check failed: " + folder.name)
        recorded = json.loads(path.read_text())
        inputs = list(read_jsonl(folder / "inputs.jsonl"))
        preds = list(read_jsonl(folder / "predictions.jsonl"))
        candidates = list(read_jsonl(folder / "candidates.jsonl"))
        validate_predictions(preds, inputs, ids)
        fresh, per_query = evaluate(
            preds, inputs, upstream, recorded["dataset"], candidates
        )
        for group in [
            "mean_over_scored_queries",
            "official_style_submitted_count_weighted",
            "candidate_recall",
        ]:
            for metric, value in fresh[group].items():
                if abs(value - recorded[group][metric]) > 1e-12:
                    raise ValueError(f"{folder.name}: {group}/{metric} differs")
        if (
            fresh["scored"] != recorded["scored"]
            or fresh["submitted"] != recorded["submitted"]
        ):
            raise ValueError("Coverage differs: " + folder.name)
        cfg = json.loads((folder / "config.json").read_text())["config"]
        if recorded["dataset"] == "mtragun":
            if fingerprint(cfg) != frozen["final_configs"][folder.name]:
                raise ValueError("Final run not registered: " + folder.name)
            for d in COLLECTIONS:
                rel = qrels(upstream, "mtragun", d)
                required = {t["task_id"] for t in inputs if t["domain"] == d}
                outside = set(rel) - required
                missing = {pid for r in rel.values() for pid in r if pid not in ids[d]}
                if outside or missing:
                    raise ValueError(f"{d}: final qrel coverage mismatch")
                final_audit[d] = {
                    "judged_queries": len(rel),
                    "missing_input_ids": len(outside),
                    "missing_relevant_passage_ids": len(missing),
                }
        report = (ROOT / "report/PROJECT_REPORT.md").read_text()
        score = f"{recorded['mean_over_scored_queries']['ndcg_cut_5']:.4f}"
        if folder.name not in report or score not in report:
            raise ValueError("Report missing recorded result: " + folder.name)
        runs[folder.name] = {
            "status": "passed",
            "submitted": fresh["submitted"],
            "scored": fresh["scored"],
            "prediction_sha256": file_hash(folder / "predictions.jsonl"),
        }
    subprocess.run([sys.executable, "scripts/official_parity.py"], cwd=ROOT, check=True)
    write_json(
        ROOT / "analysis/final_protocol_audit.json",
        {
            "dataset": "mtragun",
            "performed_after_frozen_prediction_generation": True,
            "domains": final_audit,
            "leaderboard_equivalence": "Not independently established",
        },
    )
    rewrite_audit = {}
    for dataset in ["mtrag", "mtragun"]:
        rows = list(read_jsonl(ROOT / "cache/rewrites" / (dataset + ".jsonl")))
        by_id = {}
        for r in rows:
            by_id.setdefault(r["task_id"], []).append(r)
        conflicts = [
            q for q, rs in by_id.items() if len({r["rewrite"] for r in rs}) > 1
        ]
        if conflicts:
            raise ValueError("Conflicting cached rewrites: " + ", ".join(conflicts))
        rewrite_audit[dataset] = {
            "rows": len(rows),
            "unique_tasks": len(by_id),
            "identical_repeat_rows": len(rows) - len(by_id),
            "conflicting_tasks": len(conflicts),
            "summed_generation_seconds": sum(r["seconds"] for r in rows),
            "timing_note": "Sum of recorded inference time shares; excludes download/model load, includes resumed repeated rows",
        }
    result = {
        "status": "passed",
        "automated_tests": int(match.group(1)) if match else test.stdout,
        "frozen_source_hashes_match": True,
        "upstream_checkout_clean": True,
        "runs": runs,
        "rewrite_cache": rewrite_audit,
        "scope": "Recomputed metrics and coverage, original corpus IDs, format logs, source freeze, report values and cached rewrite consistency",
    }
    write_json(ROOT / "analysis/verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
