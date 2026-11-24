"""Run the resource-feasible, preregistered core from data to a report.

By default this runs development only. --include-final freezes a bounded comparison
before scoring final data; it does not select or tune on final results.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from convrag.io import read_jsonl,write_json
from convrag.data import select_tasks

ROOT=Path(__file__).resolve().parents[1]
CORE=['E00_bm25_latest','E01_bm25_history','R00_bm25_latest_rerank','E02_bm25_rewrite',
      'B02_bm25_fusion','B03_bm25_rerank','B04_bm25_blend']

def call(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def rewrite(dataset,wait):
    split='validation' if dataset=='mtrag' else 'all'
    if not wait:
        call('-m','convrag.cli','rewrite','--dataset',dataset,'--split',split,
             '--output',f'cache/rewrites/{dataset}.jsonl')
        return
    required={t['task_id'] for t in select_tasks(ROOT,dataset,split)}
    path=ROOT/'cache/rewrites'/f'{dataset}.jsonl';start=time.monotonic()
    while True:
        # A concurrent writer flushes complete JSONL rows; tolerate an incomplete final row.
        present=set()
        if path.exists():
            for line in path.read_text().splitlines():
                try:present.add(json.loads(line)['task_id'])
                except json.JSONDecodeError:pass
        if required<=present:return
        if time.monotonic()-start>7200:raise TimeoutError('Rewrite producer did not complete within two hours')
        print('Waiting for',dataset,'rewrites:',len(required&present),'/',len(required),flush=True)
        time.sleep(30)

def completed(rid):
    folder=ROOT/'runs'/rid
    return ((folder/'metrics.json').exists() and (folder/'official_format_check.txt').exists()
            and 'Format is valid' in (folder/'official_format_check.txt').read_text())

def run(rid,final=False):
    if completed(rid):return
    call('-m','convrag.cli','run','--config',f'configs/{rid}.json',*(['--final'] if final else []))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-final',action='store_true')
    parser.add_argument('--wait-existing-rewrites',action='store_true',help='Only when a separate rewrite producer is already running')
    args=parser.parse_args()
    if not (ROOT/'analysis/data_manifest.json').exists():call('-m','convrag.cli','prepare')
    call('scripts/experiments.py')
    call('-m','convrag.cli','index','--backend','bm25')
    rewrite('mtrag',args.wait_existing_rewrites)
    for rid in CORE:run(rid)
    call('scripts/official_parity.py')
    best=max(CORE,key=lambda rid:json.loads((ROOT/'runs'/rid/'metrics.json').read_text())['mean_over_scored_queries']['ndcg_cut_5'])
    write_json(ROOT/'analysis/development_selection.json',{
        'selection_rule':'Maximum validation mean-over-scored nDCG@5 among seven fixed core ablations; ties use listed order',
        'eligible_runs':CORE,'selected':best,'test_scores_used':False})
    call('scripts/error_analysis.py','--run',best)
    call('scripts/demo.py','--run',best,'--export')
    if args.include_final:
        frozen=ROOT/'analysis/frozen_selection.json'
        # Keep an interpretable baseline, fusion without reranking, and the selected best.
        selected=list(dict.fromkeys(['E00_bm25_latest','B02_bm25_fusion',best]))
        if not frozen.exists():call('scripts/freeze.py','--runs',*selected)
        rewrite('mtragun',args.wait_existing_rewrites)
        final=json.loads(frozen.read_text())['final_configs']
        for rid in final:run(rid,final=True)
    call('-m','convrag.cli','report')
    print('Completed core pipeline. Selected:',best,flush=True)
if __name__=='__main__':main()
