"""Resumable GPU extension: full indexes and predeclared validation experiments.

Run from a GPU-enabled process. The original CPU study and final-test artifacts
are preserved. This workflow never evaluates the final test set.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(root=ROOT):
    from convrag.io import file_hash, fingerprint

    workspace = root / "experiments/gpu_extension_v1"
    inventory = json.loads((root / "analysis/model_inventory.json").read_text())
    original_freeze = json.loads((root / "analysis/frozen_selection.json").read_text())
    for relative, expected in original_freeze["source_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError("Original study source has changed: " + relative)
    base_names = [
        "D00_dense_latest",
        "D01_dense_fusion",
        "E03_splade_latest",
        "E04_splade_rewrite",
        "E05_splade_fusion",
        "E06_splade_rerank",
        "E07_splade_blend",
        "E08_hybrid_fusion",
        "E10_keyword_fusion",
    ]
    configs = {
        name: json.loads((root / "configs" / (name + ".json")).read_text())
        for name in base_names
    }
    for name, backends in [
        ("G01_splade_latest_rerank", ["splade"]),
        ("G02_dense_latest_rerank", ["dense"]),
        ("G03_hybrid_latest_rerank", ["splade", "dense"]),
    ]:
        configs[name] = dict(
            id=name,
            dataset="mtrag",
            split="validation",
            backends=backends,
            query_mode="latest",
            retrieve_k=100,
            pool_size=100,
            rrf_k=60,
            reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
            retrieval_rank_weight=0,
        )
    manifest = {
        "study": "gpu_extension_v1",
        "dataset": "mtrag",
        "split": "validation",
        "original_core_source_sha256": original_freeze["source_sha256"],
        "model_inventory": inventory,
        "config_sha256": {k: fingerprint(v) for k, v in configs.items()},
        "index_settings": {
            "batch_size": 16,
            "max_length": 256,
            "splade_document_terms": 128,
        },
        "selection_rule": "Highest validation mean-over-scored nDCG@5 among these twelve fixed configurations",
        "original_cpu_test_scores_already_known": True,
        "final_test_evaluation_enabled": False,
        "scope": "Separate development extension; original CPU final results are preserved",
    }
    path = workspace / "analysis/study_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Prepared study specification changed. Use a separately versioned study."
        )
    (workspace / "analysis").mkdir(parents=True, exist_ok=True)
    if not (workspace / "src").exists():
        shutil.copytree(
            root / "src",
            workspace / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
    for relative, expected in manifest["original_core_source_sha256"].items():
        if file_hash(workspace / relative) != expected:
            raise ValueError("Extension source changed: " + relative)
    for name in ["data_manifest.json", "model_inventory.json"]:
        target = workspace / "analysis" / name
        if not target.exists():
            shutil.copy2(root / "analysis" / name, target)
        if file_hash(target) != file_hash(root / "analysis" / name):
            raise ValueError("Extension data/model manifest changed: " + name)
    for relative in ["indexes", "cache/huggingface", "data/processed"]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    links = {
        "data/upstream": root / "data/upstream",
        "data/processed": root / "data/processed",
        "indexes": root / "indexes",
        "splits": root / "splits",
        "cache/huggingface": root / "cache/huggingface",
    }
    for relative, source in links.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError("Runtime link changed: " + relative)
        elif target.exists():
            raise ValueError("Expected a shared runtime link: " + relative)
        else:
            target.symlink_to(source, target_is_directory=True)
    rewrites = workspace / "cache/rewrites/mtrag.jsonl"
    rewrites.parent.mkdir(parents=True, exist_ok=True)
    reference = root / "artifacts/queries/mtrag.jsonl"
    if not rewrites.exists():
        shutil.copy2(reference, rewrites)
    if file_hash(rewrites) != file_hash(reference):
        raise ValueError("Extension query snapshot changed")
    for name, cfg in configs.items():
        target = workspace / "configs" / (name + ".json")
        if not target.exists():
            save(target, cfg)
        if (
            fingerprint(json.loads(target.read_text()))
            != manifest["config_sha256"][name]
        ):
            raise ValueError("Extension configuration changed: " + name)
    if not path.exists():
        save(path, manifest)
    return workspace, manifest


def inspect_index(workspace, backend):
    import numpy as np
    from scipy import sparse
    from convrag.data import COLLECTIONS, corpus
    from convrag.io import fingerprint

    manifest = json.loads((workspace / "analysis/data_manifest.json").read_text())
    output = {}
    for domain in COLLECTIONS:
        path = workspace / "indexes" / backend / domain
        meta = json.loads((path / "metadata.json").read_text())
        ids = json.loads((path / "ids.json").read_text())
        expected = [p["id"] for p in corpus(workspace / "data/upstream", domain)]
        if ids != expected or len(ids) != manifest["domains"][domain]["passages"]:
            raise ValueError(
                "Full original corpus IDs do not match: " + backend + "/" + domain
            )
        if meta["limited_corpus"]:
            raise ValueError("Partial pilot index cannot be evaluated")
        if meta["device"] != "mps":
            raise ValueError("Expected a GPU-built index")
        if backend == "dense":
            vectors = np.load(path / "vectors.npy", mmap_mode="r")
            assert vectors.shape == (len(ids), 384)
            for i in range(0, len(ids), 4096):
                if not np.isfinite(vectors[i : i + 4096]).all():
                    raise ValueError("Nonfinite dense vectors")
            zero = int(np.count_nonzero(np.linalg.norm(vectors[::100], axis=1) == 0))
        else:
            vectors = sparse.load_npz(path / "vectors.npz")
            assert vectors.shape == (len(ids), 30522)
            if not np.isfinite(vectors.data).all() or (vectors.data < 0).any():
                raise ValueError("Invalid sparse values")
            if (np.diff(vectors.indptr) > 128).any():
                raise ValueError("Unexpected sparse pruning")
            zero = int(np.count_nonzero(np.diff(vectors.indptr) == 0))
        output[domain] = {
            "status": "passed",
            "documents": len(ids),
            "shape": list(vectors.shape),
            "ids_sha256": fingerprint(ids),
            "zero_vector_count_or_sample_count": zero,
            "metadata": meta,
        }
        del vectors
    save(workspace / "analysis" / ("index_audit_" + backend + ".json"), output)


def summarize(workspace, manifest):
    from convrag.evaluation import bootstrap_difference

    original = json.loads(
        (ROOT / "runs/R00_bm25_latest_rerank/per_query.json").read_text()
    )
    rows = []
    details = {}
    for name in manifest["config_sha256"]:
        path = workspace / "runs" / name
        if not (path / "metrics.json").exists():
            continue
        if "Format is valid" not in (path / "official_format_check.txt").read_text():
            raise ValueError("Format check failed")
        m = json.loads((path / "metrics.json").read_text())
        per = json.loads((path / "per_query.json").read_text())
        if set(per) != set(original):
            raise ValueError("Validation query sets differ")
        score = m["mean_over_scored_queries"]["ndcg_cut_5"]
        rows.append(
            (
                name,
                score,
                m["mean_over_scored_queries"]["recall_10"],
                m["candidate_recall"]["recall_100"],
            )
        )
        details[name] = bootstrap_difference(original, per)
    rows.sort(key=lambda r: -r[1])
    text = [
        "# GPU extension: development results",
        "",
        "These are MTRAG validation results from a separately declared extension. The original CPU final test has already been evaluated. No new final-test runs are included here.",
        "",
        "Original CPU validation reference: BM25 latest + reranker = 0.3898 nDCG@5.",
        "",
        "| Configuration | nDCG@5 | Recall@10 | Candidate R@100 |",
        "|---|---|---|---|",
    ]
    text.extend(
        f"| {name} | {score:.4f} | {recall:.4f} | {candidate:.4f} |"
        for name, score, recall, candidate in rows
    )
    text += [
        "",
        "Only completed, format-checked runs are listed. Query sets must match the original 161 judged validation tasks. GPU encoding is combined with CPU sparse/dense index search. Cached reranking and resumed indexing timings require care when interpreting speed.",
        "",
    ]
    (workspace / "GPU_EXTENSION_RESULTS.md").write_text("\n".join(text))
    save(workspace / "analysis/paired_comparisons.json", details)
    if rows:
        save(
            workspace / "analysis/validation_selection.json",
            {
                "selected": rows[0][0],
                "score": rows[0][1],
                "selection_uses": "MTRAG validation only",
                "completed_configurations": len(rows),
                "final_test_evaluated_in_this_extension": False,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        workspace, manifest = prepare()
        print("Prepared:", workspace)
        return
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    from gpu_probe import probe

    check = probe()
    if not check["gpu_computation_passed"]:
        print(json.dumps(check, indent=2))
        raise SystemExit("GPU unavailable. No indexing or CPU fallback was started.")
    workspace, manifest = prepare()
    lock = (workspace / ".runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A GPU study runner is already active.")
    save(workspace / "analysis/gpu_access.json", check)
    save(
        workspace / "analysis/runtime_versions.json",
        {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "numpy",
                "scipy",
                "sentence-transformers",
            ]
        },
    )
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    status = {
        "study": "gpu_extension_v1",
        "started_utc": now(),
        "pid": os.getpid(),
        "state": "running",
        "completed_stages": [],
    }

    def stage(name, *cli):
        status.update(
            state="running",
            current_stage=name,
            updated_utc=now(),
            current_log=str(logs / (name + ".log")),
        )
        save(workspace / "analysis/status.json", status)
        print("START", name, "log:", status["current_log"], flush=True)
        start = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace / "src")
        env["HF_HOME"] = str(workspace / "cache/huggingface")
        with (logs / (name + ".log")).open("a") as output:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/run_gpu_stage.py"), *cli],
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, process.args)
        status["completed_stages"].append(
            {
                "name": name,
                "seconds_this_invocation": time.perf_counter() - start,
                "finished_utc": now(),
            }
        )
        save(workspace / "analysis/status.json", status)
        print("DONE", name, flush=True)

    def experiment(name):
        folder = workspace / "runs" / name
        if (folder / "metrics.json").exists():
            recorded = json.loads((folder / "config.json").read_text())["config"]
            from convrag.io import fingerprint

            if fingerprint(recorded) != manifest["config_sha256"][name]:
                raise ValueError("Existing run differs: " + name)
            if (
                "Format is valid"
                not in (folder / "official_format_check.txt").read_text()
            ):
                raise ValueError("Existing run failed format check")
            print("REUSE completed", name, flush=True)
        else:
            stage(
                name, "run", "--config", str(workspace / "configs" / (name + ".json"))
            )
        summarize(workspace, manifest)

    try:
        stage(
            "index_dense",
            "index",
            "--backend",
            "dense",
            "--batch-size",
            "16",
            "--max-length",
            "256",
        )
        inspect_index(workspace, "dense")
        for name in ["D00_dense_latest", "D01_dense_fusion", "G02_dense_latest_rerank"]:
            experiment(name)
        stage(
            "index_splade",
            "index",
            "--backend",
            "splade",
            "--batch-size",
            "16",
            "--max-length",
            "256",
        )
        inspect_index(workspace, "splade")
        for name in manifest["config_sha256"]:
            if name not in [
                "D00_dense_latest",
                "D01_dense_fusion",
                "G02_dense_latest_rerank",
            ]:
                experiment(name)
        status.update(state="completed", finished_utc=now(), current_stage=None)
        save(workspace / "analysis/status.json", status)
        print(
            "GPU extension completed. Results:",
            workspace / "GPU_EXTENSION_RESULTS.md",
            flush=True,
        )
    except BaseException as error:
        status.update(
            state="failed", error=f"{type(error).__name__}: {error}", updated_utc=now()
        )
        save(workspace / "analysis/status.json", status)
        raise


if __name__ == "__main__":
    main()
