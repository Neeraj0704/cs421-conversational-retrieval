"""Save domain-balanced validation examples and objective failure diagnostics."""
import argparse
import json
from pathlib import Path
from convrag.io import read_jsonl,write_json
from convrag.data import qrels,COLLECTIONS,corpus

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',default='E00_bm25_latest')
    args=p.parse_args();root=Path.cwd();folder=root/'runs'/args.run
    m=json.loads((folder/'metrics.json').read_text())
    if m['dataset']!='mtrag':raise ValueError('Error inspection is limited to development')
    scores=json.loads((folder/'per_query.json').read_text());baseline=json.loads((root/'runs'/args.baseline/'per_query.json').read_text())
    tasks={r['task_id']:r for r in read_jsonl(folder/'inputs.jsonl')}
    preds={r['task_id']:r for r in read_jsonl(folder/'predictions.jsonl')}
    candidates={r['task_id']:r for r in read_jsonl(folder/'candidates.jsonl')}
    traces={r['task_id']:r for r in read_jsonl(folder/'traces.jsonl')}
    diagnoses={'no_relevant_candidate':0,'relevant_candidate_but_none_in_top5':0,'relevant_in_top5':0};examples=[]
    for domain in COLLECTIONS:
        rel=qrels(root/'data/upstream','mtrag',domain);group=[]
        for q,s in scores.items():
            if s['domain']!=domain:continue
            positive={pid for pid,grade in rel[q].items() if grade>0}
            pool={c['document_id'] for c in candidates[q]['contexts']};top={c['document_id'] for c in preds[q]['contexts'][:5]}
            category='relevant_in_top5' if positive&top else 'relevant_candidate_but_none_in_top5' if positive&pool else 'no_relevant_candidate'
            diagnoses[category]+=1
            group.append({'task_id':q,'domain':domain,'delta_ndcg5':s['ndcg_cut_5']-baseline[q]['ndcg_cut_5'],
                          'ndcg5':s['ndcg_cut_5'],'diagnosis':category,'conversation':tasks[q]['input'],
                          'queries':traces[q]['queries'],'relevant_ids':sorted(positive),
                          'top5':[c['document_id'] for c in preds[q]['contexts'][:5]]})
        group.sort(key=lambda r:r['delta_ndcg5'])
        # Three largest regressions and three gains per domain; illustrative, not random.
        picked={r['task_id']:r for r in [*group[:3],*group[-3:]]}
        need={pid for r in picked.values() for pid in [*r['relevant_ids'][:1],*r['top5'][:1]]}
        snippets={d['id']:d['text'][:1200] for d in corpus(root/'data/upstream',domain) if d['id'] in need}
        for r in picked.values():
            r['first_relevant_excerpt']=snippets.get(next(iter(r['relevant_ids']),''),'')
            r['top_ranked_excerpt']=snippets.get(r['top5'][0],'')
        examples.extend(picked.values())
    write_json(root/'analysis'/('errors_'+args.run+'.json'),{'run':args.run,'baseline':args.baseline,
        'counts':diagnoses,'sampling':'Three largest gains and regressions per domain; not representative',
        'interpretation':'Categories describe judged retrieval/ranking outcomes; no automatic claim about rewrite correctness',
        'examples':examples})
    print(diagnoses)
if __name__=='__main__':main()
