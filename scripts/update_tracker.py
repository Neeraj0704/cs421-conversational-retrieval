"""Update the human-readable tracker from completed artifacts, preserving open tasks."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / 'PROJECT_PLAN.md'
    text = path.read_text()
    results = {}
    for p in (ROOT / 'runs').glob('*/metrics.json'):
        check = p.parent / 'official_format_check.txt'
        if check.exists() and 'Format is valid' in check.read_text():
            results[p.parent.name] = json.loads(p.read_text())
    dev = {k:v for k,v in results.items() if v['dataset']=='mtrag'}
    final = {k:v for k,v in results.items() if v['dataset']=='mtragun'}
    complete = [
        'Record decisions and unresolved', 'Inspect several real examples',
        'Use a known-score official prediction artifact', 'Generate and cache conservative rewrites',
        'Run BM25 using those rewrites', 'Fuse original and rewrite rankings',
        'Rerank a fixed candidate pool', 'Compare pure reranking',
        'Check retrieval and ranking failures', 'Choose one experiment based',
        'Keep or reject the component', 'Select the primary method',
        'Freeze code/configuration', 'Preserve our simple baseline',
        'Explain which components', 'Include results, ablations',
    ]
    if len(final)==3:
        complete += ['Generate final predictions', 'Validate task coverage',
                     'Score the prespecified runs', 'Report official-style',
                     'Compare with published results']
    for prefix in complete:
        text = text.replace('- [ ] '+prefix, '- [x] '+prefix)
    text = re.sub(r'\*\*Status:\*\*[^\n]*', '**Status:** Core implementation and seven development ablations complete. '+
        ('Three frozen final comparisons complete; course-format review remains.' if len(final)==3 else 'Frozen final evaluation in progress.'),text,count=1)
    aliases = {'E00':'E00_bm25_latest','E01':'E01_bm25_history','E02':'E02_bm25_rewrite'}
    lines = text.splitlines()
    for i,line in enumerate(lines):
        match = re.match(r'\| (E\d\d) \|',line)
        if not match:continue
        code = match.group(1)
        if code in aliases and aliases[code] in results:
            score=results[aliases[code]]['mean_over_scored_queries']['ndcg_cut_5']
            parts=line.split('|');parts[-2]=f' Complete: validation nDCG@5 {score:.5f} ';lines[i]='|'.join(parts)
        elif code in ['E03','E04','E05','E06','E07','E08','E09']:
            parts=line.split('|');parts[-2]=' Deferred full-index experiment after CPU feasibility gate ';lines[i]='|'.join(parts)
    text='\n'.join(lines)+'\n'
    start=text.index('## 12. Current status and next action')
    end=text.index('### Decision log',start)
    status='''## 12. Current status and next action

| Milestone | Status | Evidence |
|---|---|---|
| Research synthesis and plan | Complete for implementation scope | Research map and source links; 50-primary-paper audit not verified |
| Final-project brief | Pending | Proposal instructions only; exact final rubric unavailable |
| Hardware/model feasibility | Complete | `analysis/feasibility_*.json`, `analysis/model_inventory.json` |
| Data audit | Complete | 366,479 passages; pinned source and checksums in `analysis/data_manifest.json` |
| Evaluation | Complete | Official format checks, metric parity, automated tests |
| Baselines and core pipeline | Complete | Seven validation runs, rewrites, fusion, reranking and rank blending |
| Learned sparse/dense extension | Implemented and smoke-tested; full experiments deferred | 16-passage index tests; estimated 10.8/3.2 CPU indexing hours |
| Final evaluation | FINAL_STATUS | Frozen baseline, fusion and selected reranker configurations |
| Report and walkthrough | REPORT_STATUS | `report/`, `walkthrough/`; automated review aid is separate from student presentation |
| Course submission | Pending | No course upload has been performed |

**Next action:** NEXT_ACTION

'''.replace('FINAL_STATUS','Complete' if len(final)==3 else 'In progress').replace('REPORT_STATUS','Generated; course-format review pending' if len(final)==3 else 'Development draft; final regeneration pending').replace('NEXT_ACTION',
        'Review the final course brief, report and walkthrough, and run larger neural models only as a separately documented extension.' if len(final)==3 else 'Finish the already frozen final comparisons, regenerate deliverables, validate the artifacts and push the completed checkpoint.')
    text=text[:start]+status+text[end:]
    marker='## Latest implementation results'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n'
    rows=['| Run | Dataset | Scored / input | nDCG@5 |','|---|---|---|---|']
    for rid,m in sorted(results.items()):
        rows.append(f"| {rid} | {m['dataset']} | {m['scored']} / {m['submitted']} | {m['mean_over_scored_queries']['ndcg_cut_5']:.5f} |")
    text+='\n'+marker+'\n\n'+'\n'.join(rows)+'\n\n'
    text+='The source/configuration hashes in the freeze manifest are authoritative; its Git SHA identifies the preceding checkpoint. The completed commit also includes the hashed working-tree changes. Final test outcomes are not used to change the selected method.\n'
    path.write_text(text)
    print('Updated project tracker:',len(dev),'development and',len(final),'final runs')

if __name__=='__main__':main()
