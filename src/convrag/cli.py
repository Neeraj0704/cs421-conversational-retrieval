from __future__ import annotations
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from .data import prepare, select_tasks, COLLECTIONS, corpus, UPSTREAM_COMMIT
from .io import read_jsonl, write_json, write_jsonl, fingerprint
from .queries import query_views
from .retrieval import BM25Index, build_bm25, rrf
from .evaluation import evaluate, validate_predictions, bootstrap_difference

def run(args):
    root=Path(args.root); upstream=root/args.upstream
    cfg=json.loads(Path(args.config).read_text())
    out=root/"runs"/cfg["id"]
    if (out/"metrics.json").exists() and not args.overwrite:
        raise ValueError("Run already exists; use a new run ID")
    dataset=cfg.get("dataset","mtrag")
    if dataset=="mtragun" and not args.final:
        raise ValueError("Final dataset requires explicit --final after a frozen development decision")
    tasks=select_tasks(root,dataset,cfg.get("split","validation"),cfg.get("limit"))
    rewrite_file=cfg.get("rewrites")
    rewrites=None
    if rewrite_file:
        rewrites={r["task_id"]:r["rewrite"] for r in read_jsonl(root/rewrite_file)}
    predictions=[]; candidates=[]; traces=[]; latencies=[]
    by_domain=defaultdict(list)
    for t in tasks: by_domain[t["domain"]].append(t)
    runstart=time.perf_counter()
    reranker=None
    if cfg.get("reranker"):
        from .neural import Reranker
        reranker=Reranker(root,cfg["reranker"])
    for domain,dtasks in by_domain.items():
        backends=cfg.get("backends",["bm25"])
        indices=[]
        for backend in backends:
            if backend=="bm25": indices.append(BM25Index(root/"indexes/bm25"/domain))
            else:
                from .neural import NeuralIndex
                indices.append(NeuralIndex(root/"indexes"/backend/domain))
        texts={d["id"]:d["text"] for d in corpus(upstream,domain)} if reranker else None
        for n,t in enumerate(dtasks):
            start=time.perf_counter()
            views=query_views(t,cfg.get("query_mode","latest"),rewrites)
            rankings=[]
            for index in indices:
                for v in views: rankings.append(index.search(v,cfg.get("retrieve_k",100)))
            fused=rankings[0] if len(rankings)==1 else rrf(rankings,k=cfg.get("rrf_k",60))
            fused=fused[:cfg.get("pool_size",100)]
            search_time=time.perf_counter()-start
            ranked=fused
            rerank_time=0
            if reranker:
                rs=time.perf_counter()
                q=rewrites[t["task_id"]] if rewrites else views[-1]
                ordered=reranker.rank(q,fused,texts)
                weight=cfg.get("retrieval_rank_weight",0)
                ranked=rrf([fused,ordered],weights=[weight,1-weight],k=cfg.get("rrf_k",60)) if weight else ordered
                rerank_time=time.perf_counter()-rs
            top=ranked[:10]
            # Encode final ordering explicitly; never trust tied neural scores.
            pred={"task_id":t["task_id"],"Collection":t["Collection"],
                  "contexts":[{"document_id":p,"score":float(len(top)-i)} for i,(p,_) in enumerate(top)]}
            cand={"task_id":t["task_id"],"Collection":t["Collection"],
                  "contexts":[{"document_id":p,"score":float(len(fused)-i)} for i,(p,_) in enumerate(fused)]}
            predictions.append(pred); candidates.append(cand)
            latencies.append(time.perf_counter()-start)
            traces.append({"task_id":t["task_id"],"queries":views,"search_calls":len(rankings),
                           "candidate_count":len(fused),"search_seconds":search_time,
                           "rerank_seconds":rerank_time})
            if n%25==0: print(cfg["id"],domain,n,"/",len(dtasks),flush=True)
    passage_ids={d:{r["id"] for r in corpus(upstream,d)} for d in by_domain}
    validate_predictions(predictions,tasks,passage_ids)
    summary,scores=evaluate(predictions,tasks,upstream,dataset,candidates)
    summary["runtime"]={"total_seconds":time.perf_counter()-runstart,
                        "median_query_seconds":float(np.median(latencies)),
                        "p95_query_seconds":float(np.quantile(latencies,.95)),
                        "rewrite_generation_seconds":"excluded; cached, recorded separately",
                        "includes_first_query_warmup":True}
    manifest={"config":cfg,"config_sha256":fingerprint(cfg),"upstream_commit":UPSTREAM_COMMIT,
              "python":platform.python_version(),"platform":platform.system()+" "+platform.machine()}
    write_json(out/"config.json",manifest);write_json(out/"metrics.json",summary)
    write_json(out/"per_query.json",scores)
    write_jsonl(out/"predictions.jsonl",predictions);write_jsonl(out/"candidates.jsonl",candidates)
    write_jsonl(out/"inputs.jsonl",tasks);write_jsonl(out/"traces.jsonl",traces)
    checker=upstream/"scripts/evaluation/format_checker.py"
    result=subprocess.run(["python3",str(checker),"--input_file",str(out/"inputs.jsonl"),
        "--prediction_file",str(out/"predictions.jsonl"),"--mode","retrieval_taska"],capture_output=True,text=True)
    (out/"official_format_check.txt").write_text(result.stdout+result.stderr)
    if result.returncode or "Format is valid" not in result.stdout:
        raise RuntimeError("Official format checker rejected predictions: "+result.stdout+result.stderr)
    print(json.dumps(summary,indent=2))

def main():
    parser=argparse.ArgumentParser(description="Conversational passage retrieval")
    parser.add_argument("--root",default=".")
    parser.add_argument("--upstream",default="data/upstream")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare")
    idx=sub.add_parser("index");idx.add_argument("--backend",choices=["bm25","splade","dense"],default="bm25")
    idx.add_argument("--domains",nargs="+",default=list(COLLECTIONS))
    idx.add_argument("--batch-size",type=int,default=16)
    idx.add_argument("--max-length",type=int,default=256)
    idx.add_argument("--limit",type=int)
    rew=sub.add_parser("rewrite");rew.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    rew.add_argument("--split",default="all");rew.add_argument("--dataset",default="mtrag")
    rew.add_argument("--limit",type=int);rew.add_argument("--history-mode",default="full",choices=["full","selected"])
    rew.add_argument("--output",default="cache/rewrites/mtrag.jsonl")
    runp=sub.add_parser("run");runp.add_argument("--config",required=True)
    runp.add_argument("--overwrite",action="store_true");runp.add_argument("--final",action="store_true")
    comp=sub.add_parser("compare");comp.add_argument("a");comp.add_argument("b")
    sub.add_parser("report")
    args=parser.parse_args()
    root=Path(args.root).resolve();upstream=root/args.upstream
    if args.command=="prepare": print(json.dumps(prepare(root,upstream),indent=2))
    elif args.command=="index":
        if args.backend=="bm25": build_bm25(root,upstream,args.domains)
        else:
            from .neural import build_neural
            build_neural(root,upstream,args)
    elif args.command=="rewrite":
        from .neural import generate_rewrites
        generate_rewrites(root,args)
    elif args.command=="run": run(args)
    elif args.command=="compare":
        a=json.loads((root/"runs"/args.a/"per_query.json").read_text())
        b=json.loads((root/"runs"/args.b/"per_query.json").read_text())
        result=bootstrap_difference(a,b);write_json(root/"analysis"/(args.a+"__"+args.b+".json"),result);print(result)
    elif args.command=="report":
        from .report import build_report
        build_report(root)
if __name__=="__main__": main()

