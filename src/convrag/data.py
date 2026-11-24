from __future__ import annotations
import csv
import io
import json
import random
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from .io import read_jsonl, write_json, write_jsonl, file_hash

UPSTREAM_URL = "https://github.com/IBM/mt-rag-benchmark.git"
UPSTREAM_COMMIT = "e0970de1b1207c2a9506adf238e6e1a22037e38a"
COLLECTIONS = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611",
}
DOMAIN = {v: k for k, v in COLLECTIONS.items()}
DOMAIN.update({k:k for k in COLLECTIONS})
DOMAIN["ibmcloud"] = "cloud"

def corpus(upstream, domain):
    archive = Path(upstream) / "corpora/passage_level" / (domain + ".jsonl.zip")
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX"))
        with z.open(name) as f:
            for line in f:
                row = json.loads(line)
                yield {"id": str(row["_id"]), "text": row["text"], "title": row.get("title", "")}

def qrels(upstream, dataset, domain):
    path = (Path(upstream) / "mtrag-human/retrieval_tasks" / domain / "qrels/dev.tsv"
            if dataset == "mtrag" else
            Path(upstream) / "mtragun-human/retrieval_tasks/qrels" / (domain + ".tsv"))
    rels = {}
    with path.open() as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for qid, pid, grade, *_ in rows:
            rels.setdefault(qid, {})[pid] = int(grade)
    return rels

def tasks(upstream, dataset):
    folder = "mtrag-human" if dataset == "mtrag" else "mtragun-human"
    path = Path(upstream) / folder / "generation_tasks/reference.jsonl"
    out = []
    for row in read_jsonl(path):
        col = row["Collection"]
        if col not in DOMAIN:
            raise ValueError("Unknown collection: " + col)
        # Deliberately whitelist fields: no targets, contexts, labels or supplied rewrites.
        out.append({
            "task_id": row["task_id"], "conversation_id": row["conversation_id"],
            "Collection": col, "domain": DOMAIN[col],
            "input": [{"speaker": x["speaker"], "text": x["text"]} for x in row["input"]],
        })
    if len({t["task_id"] for t in out}) != len(out): raise ValueError("Duplicate task IDs")
    return out

def prepare(root, upstream):
    root, upstream = Path(root), Path(upstream)
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(upstream)], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", UPSTREAM_COMMIT], check=True)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT: raise ValueError(f"Expected pinned commit {UPSTREAM_COMMIT}, got {commit}")
    out = root / "data/processed"
    dev, final = tasks(upstream, "mtrag"), tasks(upstream, "mtragun")
    write_jsonl(out / "mtrag.inputs.jsonl", dev)
    write_jsonl(out / "mtragun.inputs.jsonl", final)
    # Development metadata remains separate from retrieval inputs.
    raw = list(read_jsonl(upstream / "mtrag-human/generation_tasks/reference.jsonl"))
    write_json(out / "mtrag.dimensions.json", {
        r["task_id"]: {k: r.get(k) for k in ("Answerability", "Multi-Turn", "Question Type")}
        for r in raw
    })
    rng = random.Random(42); train, valid = [], []
    for domain in COLLECTIONS:
        ids = sorted({t["conversation_id"] for t in dev if t["domain"] == domain})
        rng.shuffle(ids); nv = max(1, round(0.2 * len(ids)))
        valid.extend(ids[:nv]); train.extend(ids[nv:])
    split = {"seed": 42, "unit": "conversation_id", "train": sorted(train), "validation": sorted(valid)}
    assert not set(train).intersection(valid)
    write_json(root / "splits/conversations.json", split)
    manifest = {"repository": UPSTREAM_URL, "commit": commit, "datasets": {
        "mtrag": {"tasks": len(dev), "conversations": len({t["conversation_id"] for t in dev})},
        "mtragun": {"tasks": len(final), "conversations": len({t["conversation_id"] for t in final}),
                    "qrels_inspected": False}},
        "domains": {}, "split": {k: len(v) for k, v in split.items() if isinstance(v, list)}}
    # Do NOT read final-test qrels here. Audit development IDs only.
    for domain in COLLECTIONS:
        archive = upstream / "corpora/passage_level" / (domain + ".jsonl.zip")
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad: raise ValueError("Corrupt archive member: " + bad)
        docs = list(corpus(upstream, domain)); ids = {d["id"] for d in docs}
        if len(ids) != len(docs): raise ValueError("Duplicate corpus ID")
        rels = qrels(upstream, "mtrag", domain)
        missing = sorted({p for rel in rels.values() for p in rel if p not in ids})
        if missing: raise ValueError(f"{domain}: {len(missing)} unknown development qrel IDs")
        manifest["domains"][domain] = {
            "passages": len(docs), "judged_development_queries": len(rels),
            "archive_sha256": file_hash(archive), "missing_development_passage_ids": len(missing),
        }
        print(domain, manifest["domains"][domain], flush=True)
    manifest["total_passages"] = sum(x["passages"] for x in manifest["domains"].values())
    write_json(root / "analysis/data_manifest.json", manifest)
    return manifest

def select_tasks(root, dataset="mtrag", split="validation", limit=None):
    root = Path(root)
    rows = list(read_jsonl(root / "data/processed" / (dataset + ".inputs.jsonl")))
    if dataset == "mtrag" and split != "all":
        ids = set(json.loads((root / "splits/conversations.json").read_text())[split])
        rows = [r for r in rows if r["conversation_id"] in ids]
    if limit: rows = rows[:limit]
    return rows

