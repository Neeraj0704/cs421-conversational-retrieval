# Conversational passage retrieval — CS421

Research-driven implementation of **SemEval-2026 Task 8 / MTRAGEval Subtask A**.
Given a conversation up to the current question, retrieve the ten best existing passages
from its assigned collection. This project evaluates retrieval; it does not generate final answers.

## Run locally

Python 3.11 or 3.12 is required. No paid API is required.

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[neural,report,test]'
source .venv/bin/activate
convrag prepare
python scripts/experiments.py
pytest -q
convrag index --backend bm25
convrag run --config configs/E00_bm25_latest.json
convrag run --config configs/E01_bm25_history.json
convrag rewrite --split validation
convrag run --config configs/E02_bm25_rewrite.json
convrag run --config configs/B02_bm25_fusion.json
convrag run --config configs/B03_bm25_rerank.json
convrag index --backend splade --batch-size 8
convrag run --config configs/E03_splade_latest.json
convrag run --config configs/E04_splade_rewrite.json
convrag run --config configs/E05_splade_fusion.json
convrag run --config configs/E06_splade_rerank.json
convrag run --config configs/E07_splade_blend.json
convrag index --backend dense --batch-size 32
convrag run --config configs/D00_dense_latest.json
convrag run --config configs/D01_dense_fusion.json
convrag run --config configs/E08_hybrid_fusion.json
convrag report
```

Corpus preparation downloads the IBM benchmark at a pinned Git commit. Models download on first use;
their resolved revisions are recorded. Neural indexing covers the entire corpus, so it can take
hours on a CPU. Partial pilot indexes are rejected by evaluation. Index construction is resumable.
MLX is optional (`pip install -e '.[apple]'`), requires accessible Apple Metal, and is selected by
passing an `mlx-community/...` rewrite model. The default CPU-compatible rewriter is Qwen2.5-0.5B-Instruct.

## Research implemented

| Research idea | Implementation / ablation |
|---|---|
| Resolve conversational references | Conservative standalone query rewriting; compare latest question and raw user history |
| Retain the original question | Original + rewrite reciprocal rank fusion, preventing complete dependence on a rewrite |
| Learned sparse retrieval | SPLADE++ masked-LM max log-ReLU pooling; document vectors pruned to 128 terms for local resources |
| Semantic retrieval | BGE-small-en-v1.5 normalized dense vectors, exact inner-product search |
| Improve top results | MiniLM cross-encoder; compare pure reranking and a 50/50 rank blend |
| Test complementary signals | Sparse + dense fusion and a heuristic keyword view |
| Separate promising ideas from evidence | Every experiment has a versioned config; only completed runs receive scores |

SPLADE++, a small BGE encoder and a compact reranker are practical local choices, not a claim that
these are the strongest available models. ColBERT and model fine-tuning remain optional extensions.

## Evaluation rules

- Full official passage collections, original IDs, ten unique passages per question.
- Fixed seed-42, domain-stratified 80/20 split **by conversation**. No turns from the same
  conversation occur in both development partitions.
- Retrieval sees only the supplied conversation prefix. Gold targets, relevant passages, labels
  and provided benchmark rewrites are excluded from model inputs.
- Primary development measure: nDCG@5 over judged validation questions. Recall@10 and candidate
  Recall@50/100 are also recorded, together with domain scores and local latency.
- We also report the upstream scorer's submitted-count-weighted domain aggregation separately.
  This is not automatically the same as the simple average over scored questions.
- Official format checking runs on every prediction file. Metric parity is checked separately.
- MTRAG-UN is reserved for final testing after a frozen development decision. A final run requires
  `--final`; no performance claim is made before a completed run.
- Historical winner comparisons require matching released data, qrels and aggregation. Until
  that is established, scores are contextual comparisons rather than official competition results.

## Files

`src/convrag/` contains preparation, retrieval, rewriting, reranking and evaluation.
`configs/` defines experiments; `splits/` fixes development partitions.
`analysis/` holds dataset audits and comparisons; `runs/<id>/` holds metrics, per-query scores,
config provenance, format-check logs and local predictions. Large corpus, model and index files
are intentionally regenerated, rather than committed. `PROJECT_PLAN.md` is the original project plan.

## Sources

- [IBM MTRAG benchmark, data and evaluation](https://github.com/IBM/mt-rag-benchmark)
- [SPLADE implementation and publications](https://github.com/naver/splade)
- [SPLADE++ model](https://huggingface.co/naver/splade-cocondenser-ensembledistil)
- [BGE-small model and retrieval instructions](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Upstream data and pretrained models retain their respective licenses. This repository contains
original pipeline code; it does not redistribute benchmark corpora or model weights.
