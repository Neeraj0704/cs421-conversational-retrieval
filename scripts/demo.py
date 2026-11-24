"""Display genuine saved retrieval output and the exact conversation prefix."""
import argparse
import json
from pathlib import Path
from convrag.io import read_jsonl,write_json
from convrag.data import corpus

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='E00_bm25_latest');p.add_argument('--task-id')
    p.add_argument('--export',action='store_true');args=p.parse_args();root=Path.cwd();folder=root/'runs'/args.run
    tasks={t['task_id']:t for t in read_jsonl(folder/'inputs.jsonl')}
    predictions={t['task_id']:t for t in read_jsonl(folder/'predictions.jsonl')}
    traces={t['task_id']:t for t in read_jsonl(folder/'traces.jsonl')}
    qid=args.task_id or next(q for q,t in tasks.items() if len(t['input'])>2)
    task=tasks[qid];ids={c['document_id'] for c in predictions[qid]['contexts']}
    docs={d['id']:d for d in corpus(root/'data/upstream',task['domain']) if d['id'] in ids}
    result={'run':args.run,'task_id':qid,'domain':task['domain'],'conversation':task['input'],
            'query_views':traces[qid]['queries'],'results':[{'rank':i+1,**docs[c['document_id']]} for i,c in enumerate(predictions[qid]['contexts'])]}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.export:write_json(root/'walkthrough/demo_output.json',result)
if __name__=='__main__':main()
