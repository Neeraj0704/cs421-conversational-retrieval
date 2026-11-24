from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from html import escape
from collections import defaultdict
from .io import write_json, read_jsonl
from .evaluation import bootstrap_difference

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])

def build_report(root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    runs=[]
    for p in sorted((root/'runs').glob('*/metrics.json')):
        if not (p.parent/'official_format_check.txt').exists():continue
        m=json.loads(p.read_text());cfg=json.loads((p.parent/'config.json').read_text())['config']
        if 'Format is valid' not in (p.parent/'official_format_check.txt').read_text():continue
        runs.append((p.parent.name,cfg,m))
    dev=[r for r in runs if r[2]['dataset']=='mtrag'];final=[r for r in runs if r[2]['dataset']=='mtragun']
    if not dev:raise ValueError('No verified development runs')
    best=max(dev,key=lambda r:r[2]['mean_over_scored_queries']['ndcg_cut_5'])
    manifest=json.loads((root/'analysis/data_manifest.json').read_text())
    columns=['Run','nDCG@5','Recall@10','Candidate R@100','Scored / input','Median search + rank (s)']
    def rows(group):
        return [[rid,f"{m['mean_over_scored_queries']['ndcg_cut_5']:.4f}",
                 f"{m['mean_over_scored_queries']['recall_10']:.4f}",f"{m['candidate_recall']['recall_100']:.4f}",
                 f"{m['scored']} / {m['submitted']}",f"{m['runtime']['median_query_seconds']:.4f}"] for rid,cfg,m in group]
    aggregate_rows=[[rid,f"{m['mean_over_scored_queries']['ndcg_cut_5']:.4f}",
                     f"{m['official_style_submitted_count_weighted']['ndcg_cut_5']:.4f}",
                     f"{m['mean_over_scored_queries']['recall_5']:.4f}",
                     f"{m['official_style_submitted_count_weighted']['recall_5']:.4f}"] for rid,cfg,m in final]
    comparisons=[]
    base=root/'runs/E00_bm25_latest/per_query.json'
    if base.exists():
        a=json.loads(base.read_text())
        for rid,cfg,m in dev:
            b=json.loads((root/'runs'/rid/'per_query.json').read_text())
            if rid=='E00_bm25_latest' or set(a)!=set(b):continue
            result=bootstrap_difference(a,b);comparisons.append((rid,result))
    write_json(root/'analysis/paired_comparisons.json',dict(comparisons))
    sections=[
      ('CS421: Improving multi-turn passage retrieval',[
        'Implementation and experiment report. Prepared from saved, format-checked runs; this is a draft pending the separate final-project formatting instructions.',
        'The system receives a conversation up to the latest user question and retrieves ten passages from the assigned collection. It implements SemEval-2026 Task 8 / MTRAGEval Subtask A. It does not generate an answer.',
        f"The strongest completed validation run is {best[0]}, with nDCG@5 = {best[2]['mean_over_scored_queries']['ndcg_cut_5']:.4f}. This is our local development result, not an official competition submission or a verified win over the historical leaderboard.",
      ]),
      ('Research question and method',[
        'Can conversation context improve retrieval, and when do rewriting, query fusion and cross-encoder reranking help? We test each addition against fixed baselines, rather than assume a larger pipeline always improves quality.',
        'BM25 searches existing passage boundaries using English stopwords, Snowball stemming, Lucene-style scoring, k1=1.2 and b=0.75. Each query searches only its assigned domain.',
        'A local Qwen2.5-0.5B-Instruct model rewrites the current question. A conservative prompt asks it to resolve references and preserve constraints without answering or adding facts. All available past user and assistant turns can be used to resolve references; assistant content is not treated as verified evidence. The generated rewrite is cached with the prompt, model revision and conversation prefix.',
        'Fusion searches the original question and rewrite separately, then combines ranks using reciprocal rank fusion with k=60. This retains an original wording branch when a rewrite omits a useful exact term. The fixed candidate pool contains at most 100 passages.',
        'A pretrained MS MARCO MiniLM-L6 cross-encoder scores query-passage pairs. We compare pure reranking and a 50/50 fusion of retrieval and reranker ranks on the same candidates. Model weights are pretrained; this project does not fine-tune them.',
        'SPLADE++ and BGE-small indexing and search are implemented as optional backends. SPLADE uses maximum log(1+ReLU(logit)) pooling and at most 128 document terms; BGE uses normalized CLS vectors, the documented query instruction, and exact inner-product search. These local choices and 256-token document limits differ from larger unrestricted model configurations.',
      ]),
      ('Data and separation of development from test',[
        f"Data are pinned to IBM commit {manifest['commit']}. The audited corpus contains {manifest['total_passages']:,} passages across CLAPNQ, IBM Cloud, FiQA and government collections. Original passage and task IDs are preserved, including suffixes and delimiters.",
        'MTRAG contains 842 task prefixes from 110 conversations. Seed 42 assigns 88 conversations to the development/training partition and 22 to validation, stratified by domain. No model fitting is performed on these partitions. The validation partition contains 174 prefixes; only tasks with released relevance judgments contribute to retrieval metrics.',
        'The current pinned MTRAG-UN release contains 507 input tasks. Test relevance files are not consulted for model or prompt selection. A frozen manifest records final configurations before test evaluation. Released test questions can be rewritten without reading their answers or relevance labels.',
        'A field whitelist removes targets, relevant contexts, answerability labels, question labels and provided benchmark rewrites before retrieval or rewriting. The system uses the task prefix supplied by the benchmark; it does not look at later turns.',
      ]),
      ('Evaluation and validity checks',[
        'The primary development metric is mean nDCG@5 over scored validation questions. Recall@10 measures retrieved relevant passages in the final ten; candidate Recall@100 diagnoses whether reranking had a chance to succeed. Required judged questions are not removed when retrieval fails.',
        'For each run, we separately record submitted, scored and unscored counts, per-domain metrics and the upstream-style domain aggregation weighted by submitted task counts. That weighted result can differ from a direct mean over scored questions.',
        'The unmodified official evaluate function is loaded through Python AST to avoid unrelated generation-judge imports and obsolete hard-coded data paths. Per-query nDCG@1,3,5 and Recall@5 match our wrapper on the baseline. Every completed run also passes the official JSONL format checker and stricter checks for task coverage, original corpus IDs, unique top ten, finite scores and decreasing rank scores.',
        'Prediction scores encode final rank order. Evaluation uses these numeric scores; it does not rely only on list order. Paired uncertainty uses a fixed-seed bootstrap resampling conversations, which avoids treating turns from the same conversation as independent.',
        'Local latency includes search and ranking with possible first-query warmup. Rewrite generation and initial model/index loading are recorded separately or excluded. Cached reranking can make a later blend run appear much faster; those times are not an independent live inference benchmark. CPU jobs also ran concurrently during this session, so timings are practical observations rather than controlled hardware comparisons.',
      ]),
      ('Development results',[table(columns,rows(dev))]),
      ('Uncertainty and interpretation',[
        table(['Compared with BM25 latest','Delta nDCG@5','95% conversation-bootstrap interval'],
              [[rid,f"{r['delta_b_minus_a']:+.4f}",f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"] for rid,r in comparisons]),
        'Intervals covering zero do not establish a clear directional effect on this small validation sample. Selecting the best of several validation runs also introduces selection optimism; the final test is reserved for the frozen comparisons.',
        'History concatenation can bury the latest intent under earlier topics. Rewriting can resolve references, but the small rewriter may leave pronouns unresolved or change meaning. Fusion protects some exact wording but can add irrelevant candidates. A cross-encoder can only reorder retrieved passages, so candidate recall bounds what reranking can repair.',
        *([table(['Validation diagnostic','Questions'],[[k,v] for k,v in json.loads((root/'analysis'/('errors_'+best[0]+'.json')).read_text())['counts'].items()])]
          if (root/'analysis'/('errors_'+best[0]+'.json')).exists() else []),
        *([x['observation']+f" Observed delta nDCG@5: {x['delta_ndcg5']:+.4f}." for x in json.loads((root/'analysis/manual_error_notes.json').read_text())['examples']]
          if (root/'analysis/manual_error_notes.json').exists() else []),
      ]),
      ('Final evaluation',[table(columns,rows(final)) if final else 'No final test scores have been reported yet. Final testing requires a frozen configuration manifest and explicit final-run mode.',
        *([table(['Run','nDCG@5 mean','nDCG@5 weighted','Recall@5 mean','Recall@5 weighted'],aggregate_rows)] if final else []),
        'These are results on the currently pinned release. Historical competition equivalence remains unverified unless the exact original question subset, relevance judgments, filtering and aggregation are independently matched. The organizer overview (Table 4) reports AILS-NTUA at 0.578 nDCG@5 and 0.622 Recall@5; that is contextual until exact protocol equivalence is established. It is not a comparison with our development split.',
        'An archived IBM RAG retrieval artifact is separately checked in analysis/archived_rag_artifact_check.json. Its current per-scored-query nDCG@5 is approximately 0.4866, versus rounded ELSER values of 0.45/0.48 in the development README. Query formulation, data revision and aggregation have not been matched closely enough to claim exact historical reproduction. These are supplied predictions, not our retrieval output.']),
      ('Resources and feasibility',[
        'The implementation runs on an Apple Silicon Mac, Python 3.11. The execution session does not expose Metal/MPS, so neural feasibility measurements use CPU. No paid model API or remote GPU has been provisioned.',
        *[f"{json.loads(p.read_text())['backend']}: sampled throughput {json.loads(p.read_text())['passages_per_second']:.2f} passages/s; estimated full indexing {json.loads(p.read_text())['estimated_full_index_hours']:.1f} hours. These are estimates from small samples, not completed full-index measurements." for p in sorted((root/'analysis').glob('feasibility_*.json'))],
        'The plan explicitly permits BM25 + rewriting + fusion + a compact reranker when learned sparse indexing fails the resource gate. Full SPLADE/dense quality, ColBERT, retriever fine-tuning, RL and adaptive agent loops are not claimed as completed experiments.',
      ]),
      ('Limitations and next steps',[
        'A compact general rewriter and an MS MARCO reranker are not tuned for these domains. BM25 loses some semantic matches; document token truncation and sparse pruning would limit the optional neural branches. Released relevance judgments can be incomplete.',
        'The broader research reports informed the candidate design, but their paper counts and every quantitative claim have not been independently audited. This implementation is not evidence that at least 50 primary papers were individually verified.',
        'Next work is an accelerator-backed full sparse index, then a controlled dense-complementarity test at the same final candidate budget. Only adopt a component after a measured development benefit. Do not tune after seeing final test scores.',
        'The final course brief is still needed to check page limits, video requirements, assistance disclosure and submission format. This report and the walkthrough materials must be reviewed by the student before course submission. AI assistance was used to implement and check this pipeline; the student should understand, verify and disclose it as required by the course.',
      ]),
      ('References',[
        'IBM MTRAG benchmark and evaluation: https://github.com/IBM/mt-rag-benchmark',
        'Rosenthal et al. (2026), SemEval-2026 Task 8: MTRAGEval: Evaluating Multi-Turn RAG Conversations: https://aclanthology.org/2026.semeval-1.447/',
        'Formal et al. (2022), From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective: https://arxiv.org/abs/2205.04733',
        'SPLADE code and model: https://github.com/naver/splade ; https://huggingface.co/naver/splade-cocondenser-ensembledistil (CC-BY-NC-SA-4.0 model)',
        'BGE-small-en-v1.5 model and usage: https://huggingface.co/BAAI/bge-small-en-v1.5 (MIT)',
        'MiniLM cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0)',
        'Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0)',
        'Cormack, Clarke and Buettcher (2009), Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods: https://doi.org/10.1145/1571941.1572114',
      ])]
    markdown='\n\n'.join(('## ' if i else '# ')+h+'\n\n'+'\n\n'.join(paras) for i,(h,paras) in enumerate(sections))+'\n'
    (out/'PROJECT_REPORT.md').write_text(markdown)
    (root/'RESULTS.md').write_text('# Recorded results\n\n'+
        'Only completed runs passing the official format checker are listed.\n\n'+
        '## Held-out development conversations\n\n'+table(columns,rows(dev))+'\n\n'+
        f"Validation-selected best: **{best[0]}**. Selection uses development data only.\n\n"+
        '## Frozen final comparisons\n\n'+(table(columns,rows(final)) if final else 'Not scored yet.')+'\n\n'+
        'These are local benchmark results, not an official competition submission. Historical leaderboard equivalence is unverified. '+
        'The full report includes aggregation differences, uncertainty, negative results and resource limits.\n')
    _pdf(out/'PROJECT_REPORT.pdf',sections)
    _walkthrough(root,best,dev,final)
    _plot(out,dev)
    print('Wrote',out/'PROJECT_REPORT.pdf')

def _pdf(path,sections):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=14
    styles['BodyText'].spaceAfter=9;styles['Heading2'].spaceBefore=16
    story=[]
    for i,(heading,paras) in enumerate(sections):
        story.append(Paragraph(escape(heading),styles['Title' if i==0 else 'Heading2']))
        for para in paras:
            if para.startswith('|'):
                lines=para.splitlines();rows=[[c.strip() for c in l.strip('|').split('|')] for l in [lines[0],*lines[2:]]]
                width=495/len(rows[0])
                cells=[[Paragraph(escape(c),styles['BodyText']) for c in row] for row in rows]
                tab=Table(cells,colWidths=[width]*len(rows[0]),repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),
                   ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8295a9')),
                   ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]))
                story.extend([tab,Spacer(1,10)])
            else:story.append(Paragraph(escape(para),styles['BodyText']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
        canvas.drawString(50,28,'CS421 | Conversational passage retrieval | Draft from recorded experiments')
        canvas.drawRightString(545,28,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)

def _plot(out,dev):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/nlp-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,max(3,.45*len(dev))))
    ax.barh([r[0] for r in dev],[r[2]['mean_over_scored_queries']['ndcg_cut_5'] for r in dev],color='#266b99')
    ax.set_xlabel('Validation nDCG@5 (mean over scored questions)');ax.invert_yaxis();ax.set_xlim(left=0)
    fig.tight_layout();fig.savefig(out/'validation_results.png',dpi=180);plt.close(fig)

def _walkthrough(root,best,dev,final):
    out=root/'walkthrough';out.mkdir(exist_ok=True)
    (out/'WALKTHROUGH.md').write_text(f'''# Code walkthrough and recording script

This is a recording guide, not a claim that a student-narrated video has been recorded.
Adapt its duration after checking the final course instructions.

1. Explain the task: given a conversation and its current question, return ten existing passages.
   Show the original question, a contextual rewrite and the assigned domain in saved traces.
2. Open `src/convrag/data.py`. Explain the pinned corpus version, ID preservation,
   field whitelist, conversation split and separate final data.
3. Open `src/convrag/queries.py`. Show the rewrite prompt and original-plus-rewrite views.
4. Open `src/convrag/retrieval.py`. Explain BM25 and reciprocal rank fusion.
5. Open `src/convrag/neural.py`. Explain local rewriting and query-passage reranking.
   Distinguish implemented neural index backends from completed full-corpus experiments.
6. Open `src/convrag/evaluation.py`. Explain nDCG@5, recall, scored coverage and two aggregates.
7. Run `python -m pytest -q` and `python scripts/official_parity.py`.
8. Show `runs/{best[0]}/metrics.json` and `report/validation_results.png`.
   State the best completed validation result: {best[2]['mean_over_scored_queries']['ndcg_cut_5']:.4f} nDCG@5.
   Explain negative results and uncertainty rather than presenting only the best row.
9. Show `analysis/frozen_selection.json` if present, then the final result table if scored.
   There are currently {len(final)} completed final runs; do not imply historical leaderboard equivalence.
10. End with limitations: CPU-only session, small rewriter, retrieval candidate ceiling,
    incomplete judgments, deferred accelerator experiments and unconfirmed final brief.

Reproduce a run using a NEW config ID, to preserve original experiment artifacts.
Do not show credentials, environment secrets, or private account pages in a recording.
''')
