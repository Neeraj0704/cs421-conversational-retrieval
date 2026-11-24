from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
import pytrec_eval
from .data import qrels

def validate_predictions(predictions, tasks, passage_ids=None, topk=10):
    required = {t["task_id"]: t for t in tasks}
    ids = [p["task_id"] for p in predictions]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate query IDs")
    if set(ids) != set(required): raise ValueError("Prediction/input task sets differ")
    for p in predictions:
        t = required[p["task_id"]]
        if p["Collection"] != t["Collection"]: raise ValueError("Wrong collection")
        contexts = p.get("contexts", [])
        if len(contexts) != topk: raise ValueError(f"Expected {topk} contexts")
        pids = [c["document_id"] for c in contexts]
        if len(set(pids)) != len(pids): raise ValueError("Duplicate passage IDs")
        scores = [c["score"] for c in contexts]
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
            raise ValueError("Scores must be finite numbers")
        if any(a <= b for a,b in zip(scores, scores[1:])): raise ValueError("Scores must strictly decrease")
        if passage_ids and any(pid not in passage_ids[t["domain"]] for pid in pids):
            raise ValueError("Passage absent from assigned corpus")

def evaluate(predictions, tasks, upstream, dataset="mtrag", candidates=None):
    if set(p["task_id"] for p in predictions) != set(t["task_id"] for t in tasks):
        raise ValueError("Missing or extra predictions")
    pred = {p["task_id"]: p for p in predictions}
    by_domain = defaultdict(list)
    for t in tasks: by_domain[t["domain"]].append(t)
    metrics = {"ndcg_cut.1,3,5,10", "recall.5,10,50,100"}
    per_query, domains = {}, {}
    for domain, group in by_domain.items():
        rel = qrels(upstream, dataset, domain)
        # Include every required judged query, never only successful returned queries.
        judged = {t["task_id"]: rel[t["task_id"]] for t in group if t["task_id"] in rel}
        results = {qid: {c["document_id"]: c["score"] for c in pred[qid]["contexts"]} for qid in judged}
        if not judged:
            domains[domain] = {"submitted": len(group), "scored": 0, "metrics": {}}
            continue
        scores = pytrec_eval.RelevanceEvaluator(judged, metrics).evaluate(results)
        keys = next(iter(scores.values())).keys() if scores else []
        # Empty candidate dictionaries must still count as zero.
        for t in group:
            qid = t["task_id"]
            if qid in judged:
                s = scores.get(qid, {k: 0.0 for k in keys})
                per_query[qid] = {"domain": domain, "conversation_id": t["conversation_id"], **s}
        mean = {k: float(np.mean([per_query[q][k] for q in judged])) for k in keys}
        domains[domain] = {"submitted": len(group), "scored": len(judged), "metrics": mean}
    if not per_query: raise ValueError("No queries scored: verify dataset, task IDs and qrels routing")
    keys = [k for k in next(iter(per_query.values())) if k not in {"domain","conversation_id"}]
    micro = {k: float(np.mean([r[k] for r in per_query.values()])) for k in keys}
    active = {d:s for d,s in domains.items() if s["scored"]}
    total = sum(s["submitted"] for s in active.values())
    official = {k: sum(round(s["metrics"][k],5)*s["submitted"] for s in active.values())/total for k in keys}
    summary = {"dataset": dataset, "submitted": len(tasks), "scored": len(per_query),
        "unscored": len(tasks)-len(per_query), "domains": domains,
        "mean_over_scored_queries": micro, "official_style_submitted_count_weighted": official,
        "historical_leaderboard_equivalence": "unverified"}
    if candidates:
        crecord = {p["task_id"]: p for p in candidates}
        cscores = {}
        for domain, group in by_domain.items():
            rel = qrels(upstream,dataset,domain)
            judged = {t["task_id"]:rel[t["task_id"]] for t in group if t["task_id"] in rel}
            results = {q:{c["document_id"]:c["score"] for c in crecord[q]["contexts"]} for q in judged}
            if judged: cscores.update(pytrec_eval.RelevanceEvaluator(judged,{"recall.50,100"}).evaluate(results))
        summary["candidate_recall"] = {k:float(np.mean([s[k] for s in cscores.values()]))
                                       for k in ("recall_50","recall_100")}
    return summary, per_query

def bootstrap_difference(a,b,metric="ndcg_cut_5",samples=3000,seed=42):
    if set(a)!=set(b): raise ValueError("Paired comparisons require identical query sets")
    groups=defaultdict(list)
    for q in a: groups[a[q]["conversation_id"]].append(b[q][metric]-a[q][metric])
    sums=np.array([sum(v) for v in groups.values()])
    counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(sums),size=(samples,len(sums)))
    draws=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
    return {"metric":metric,"delta_b_minus_a":float(sums.sum()/counts.sum()),
            "cluster_unit":"conversation_id","bootstrap_samples":samples,
            "ci95":[float(x) for x in np.quantile(draws,[.025,.975])]}

