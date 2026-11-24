"""Freeze and evaluate the validation-selected GPU finalists without reindexing.

This is a disclosed follow-up benchmark: earlier CPU results on the same public
test set are already known. Selection here uses only the completed GPU validation runs.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_gpu_study import prepare as prepare_study, save

ROOT = Path(__file__).resolve().parents[1]


def select_finalists(configs, scores):
    groups = {"sparse": [], "dense": [], "hybrid": []}
    if set(configs) != set(scores):
        raise ValueError("Every declared validation config needs a score")
    for name, cfg in configs.items():
        if cfg["dataset"] != "mtrag" or cfg["split"] != "validation":
            raise ValueError("Final selection must use MTRAG validation only")
        backends = set(cfg["backends"])
        family = (
            "hybrid"
            if backends == {"splade", "dense"}
            else "sparse"
            if backends == {"splade"}
            else "dense"
            if backends == {"dense"}
            else None
        )
        if family is None:
            raise ValueError("Unexpected retrieval family")
        groups[family].append(name)
    if any(not names for names in groups.values()):
        raise ValueError("Each retrieval family needs a completed validation run")
    winners = {
        family: max(sorted(names), key=lambda n: scores[n])
        for family, names in groups.items()
    }
    primary = max(sorted(scores), key=lambda n: scores[n])
    return primary, winners


def prepare_final(root=ROOT):
    from convrag.io import fingerprint, file_hash

    workspace, study = prepare_study(root)
    verified = json.loads(
        (workspace / "analysis/full_study_verification.json").read_text()
    )
    if (
        verified["status"] != "passed"
        or verified["completed_validation_configurations"] != 12
    ):
        raise ValueError("Complete and independently verify GPU validation first")
    configs = {}
    scores = {}
    for name, expected in study["config_sha256"].items():
        cfg = json.loads((workspace / "configs" / (name + ".json")).read_text())
        if fingerprint(cfg) != expected:
            raise ValueError("Validation configuration changed: " + name)
        metrics = json.loads((workspace / "runs" / name / "metrics.json").read_text())
        if metrics["dataset"] != "mtrag":
            raise ValueError("Selection cannot use final scores")
        score = metrics["mean_over_scored_queries"]["ndcg_cut_5"]
        if abs(score - verified["runs"][name]["ndcg_at_5"]) > 1e-12:
            raise ValueError("Verified validation score differs")
        configs[name] = cfg
        scores[name] = score
    primary, winners = select_finalists(configs, scores)
    snapshot = root / "artifacts/queries/mtragun.jsonl"
    target = workspace / "cache/rewrites/mtragun.jsonl"
    if not target.exists():
        shutil.copy2(snapshot, target)
    if file_hash(target) != file_hash(snapshot):
        raise ValueError("Final query snapshot differs")
    final_configs = {}
    development = {}
    for name in dict.fromkeys([primary, winners["sparse"], winners["dense"]]):
        cfg = dict(configs[name])
        cfg.update(id="FINAL_" + name, dataset="mtragun", split="all")
        if cfg.get("rewrites"):
            cfg["rewrites"] = "cache/rewrites/mtragun.jsonl"
        path = workspace / "configs" / (cfg["id"] + ".json")
        if path.exists() and json.loads(path.read_text()) != cfg:
            raise ValueError("Final configuration changed")
        if not path.exists():
            save(path, cfg)
        final_configs[cfg["id"]] = fingerprint(cfg)
        development[name] = {
            "ndcg_at_5": scores[name],
            "config_sha256": study["config_sha256"][name],
        }
    protocol = {
        "study": "gpu_final_followup_v1",
        "selection_rule": "Highest GPU validation mean nDCG@5 overall, plus highest within sparse and dense families; lexical tie break",
        "primary_final_run": "FINAL_" + primary,
        "family_winners": winners,
        "development_results": development,
        "final_configs": final_configs,
        "source_sha256": study["original_core_source_sha256"],
        "model_inventory": study["model_inventory"],
        "query_snapshot_sha256": file_hash(target),
        "original_cpu_test_results_already_known": True,
        "disclosure": "Follow-up benchmark on the previously used public final set; not a fresh unseen holdout",
        "new_gpu_test_scores_used_for_selection": False,
    }
    path = workspace / "analysis/frozen_selection.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if any(existing.get(k) != v for k, v in protocol.items()):
            raise ValueError("Final follow-up protocol changed after freeze")
        protocol = existing
    else:
        protocol["created_utc"] = datetime.now(timezone.utc).isoformat()
        save(path, protocol)
    return workspace, protocol


def verify_and_report(workspace, protocol, root=ROOT):
    from convrag.data import corpus, COLLECTIONS
    from convrag.io import read_jsonl, file_hash
    from convrag.evaluation import evaluate, validate_predictions, bootstrap_difference

    ids = {
        d: {r["id"] for r in corpus(workspace / "data/upstream", d)}
        for d in COLLECTIONS
    }
    baseline = json.loads(
        (root / "runs/FINAL_E00_bm25_latest/per_query.json").read_text()
    )
    rows = []
    checked = {}
    for name in protocol["final_configs"]:
        folder = workspace / "runs" / name
        tasks = list(read_jsonl(folder / "inputs.jsonl"))
        pred = list(read_jsonl(folder / "predictions.jsonl"))
        cand = list(read_jsonl(folder / "candidates.jsonl"))
        validate_predictions(pred, tasks, ids)
        actual, per = evaluate(
            pred, tasks, workspace / "data/upstream", "mtragun", cand
        )
        saved = json.loads((folder / "metrics.json").read_text())
        for group in [
            "mean_over_scored_queries",
            "official_style_submitted_count_weighted",
            "candidate_recall",
        ]:
            if any(abs(v - saved[group][k]) > 1e-12 for k, v in actual[group].items()):
                raise ValueError("Saved metrics differ: " + name)
        if set(per) != set(baseline):
            raise ValueError("Final scored query sets differ from baseline")
        if "Format is valid" not in (folder / "official_format_check.txt").read_text():
            raise ValueError("Official format check failed")
        checked[name] = {
            "status": "passed",
            "submitted": actual["submitted"],
            "scored": actual["scored"],
            "prediction_sha256": file_hash(folder / "predictions.jsonl"),
            "paired_vs_bm25": bootstrap_difference(baseline, per),
        }
        rows.append((name, actual))
    lines = [
        "# GPU final follow-up benchmark",
        "",
        protocol["disclosure"] + ".",
        "All three configurations were frozen from validation results before their GPU final runs. No parameters are changed using these new test scores.",
        "",
        "Prespecified primary: **" + protocol["primary_final_run"] + "**.",
        "",
        "| Run | nDCG@5 mean | nDCG@5 weighted | Recall@5 | Recall@10 | Scored / inputs |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['mean_over_scored_queries']['ndcg_cut_5']:.4f} | {m['official_style_submitted_count_weighted']['ndcg_cut_5']:.4f} | {m['mean_over_scored_queries']['recall_5']:.4f} | {m['mean_over_scored_queries']['recall_10']:.4f} | {m['scored']} / {m['submitted']} |"
        )
    lines += [
        "",
        "Earlier CPU references on this release: BM25 latest = 0.2896; BM25 latest + reranker = 0.4161 mean nDCG@5.",
        "Historical leaderboard equivalence remains unverified. The strongest final score is not used to replace the prespecified primary after testing.",
        "",
    ]
    (workspace / "GPU_FINAL_RESULTS.md").write_text("\n".join(lines))
    save(
        workspace / "analysis/final_followup_verification.json",
        {
            "status": "passed",
            "runs": checked,
            "primary_final_run": protocol["primary_final_run"],
            "original_cpu_artifacts_preserved": True,
            "original_test_results_already_known": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        workspace, protocol = prepare_final()
        print("Frozen primary:", protocol["primary_final_run"])
        print("Final runs:", ", ".join(protocol["final_configs"]))
        return
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    from gpu_probe import probe

    check = probe()
    if not check["gpu_computation_passed"]:
        raise SystemExit(
            "GPU unavailable. No CPU fallback final evaluation was started."
        )
    workspace, protocol = prepare_final()
    lock = (workspace / ".runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A GPU workflow is already running.")
    status = {
        "study": "gpu_final_followup_v1",
        "state": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": [],
    }
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace / "src")
    env["HF_HOME"] = str(workspace / "cache/huggingface")
    try:
        for name, expected in protocol["final_configs"].items():
            from convrag.io import fingerprint

            folder = workspace / "runs" / name
            if (folder / "metrics.json").exists():
                cfg = json.loads((folder / "config.json").read_text())["config"]
                if fingerprint(cfg) != expected:
                    raise ValueError("Existing final config differs")
                print("REUSE", name, flush=True)
            else:
                status.update(current_run=name, current_log=str(logs / (name + ".log")))
                save(workspace / "analysis/final_status.json", status)
                print("START", name, flush=True)
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_gpu_stage.py"),
                    "run",
                    "--config",
                    str(workspace / "configs" / (name + ".json")),
                    "--final",
                ]
                with (logs / (name + ".log")).open("a") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=workspace,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)
            status["completed_runs"].append(name)
            save(workspace / "analysis/final_status.json", status)
        verify_and_report(workspace, protocol)
        status.update(
            state="completed",
            current_run=None,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(workspace / "analysis/final_status.json", status)
        print("Verified final results:", workspace / "GPU_FINAL_RESULTS.md", flush=True)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save(workspace / "analysis/final_status.json", status)
        raise


if __name__ == "__main__":
    main()
