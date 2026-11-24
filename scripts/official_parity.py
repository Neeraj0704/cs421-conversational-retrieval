"""Execute the unmodified upstream metric function, avoiding unrelated judge imports."""
import ast
import json
from pathlib import Path
from typing import Dict,List,Tuple
import pytrec_eval
from convrag.data import qrels,UPSTREAM_COMMIT
from convrag.io import read_jsonl,write_json
from convrag.evaluation import evaluate

def main():
    upstream=Path('data/upstream');run=Path('runs/E00_bm25_latest')
    tree=ast.parse((upstream/'scripts/evaluation/run_retrieval_eval.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    scope=dict(Dict=Dict,List=List,Tuple=Tuple,pytrec_eval=pytrec_eval)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream_evaluate','exec'),scope)
    preds=list(read_jsonl(run/'predictions.jsonl'));tasks=list(read_jsonl(run/'inputs.jsonl'))
    summary,ours=evaluate(preds,tasks,upstream);domains={}
    for d in summary['domains']:
        ids={t['task_id'] for t in tasks if t['domain']==d}
        results={p['task_id']:{c['document_id']:c['score'] for c in p['contexts']} for p in preds if p['task_id'] in ids}
        scores,ndcg,_,recall,_=scope['evaluate'](qrels(upstream,'mtrag',d),results,[1,3,5])
        for q in scores:
            for k in [1,3,5]:
                assert abs(scores[q][f'ndcg_cut_{k}']-ours[q][f'ndcg_cut_{k}'])<1e-12
            assert abs(scores[q]['recall_5']-ours[q]['recall_5'])<1e-12
        domains[d]={'NDCG@5':ndcg['NDCG@5'],'queries':len(scores)}
    write_json('analysis/official_metric_parity.json',dict(status='passed',
        source='Unmodified upstream evaluate function extracted via AST; paths routed explicitly',
        metrics=['nDCG@1,3,5','Recall@5'],domains=domains,upstream_commit=UPSTREAM_COMMIT))
    print('Official metric parity passed')
if __name__=='__main__':main()
