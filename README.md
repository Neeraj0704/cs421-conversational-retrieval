# Conversational passage retrieval — CS421

Research-driven implementation of **SemEval-2026 Task 8 / MTRAGEval Subtask A**.
Given a conversation up to the current question, retrieve the ten best existing passages
from its assigned collection. This project evaluates retrieval; it does not generate final answers.

For a plain-language explanation, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Run locally

Python 3.11 or 3.12 is required. No paid API is required. Clone this private repository
using your authenticated GitHub account, then install the locked dependencies:

```sh
gh repo clone Neeraj0704/cs421-conversational-retrieval
cd cs421-conversational-retrieval
uv sync --locked --python 3.11 --extra neural --extra report --extra test
source .venv/bin/activate
pytest -q
```

The saved results are in [RESULTS.md](RESULTS.md). To reproduce them without overwriting
recorded experiments, use an isolated workspace:

```sh
python scripts/reproduce.py --name my_reproduction
```

This runs seven development comparisons and the three already frozen final configurations.
It reuses full indexes and model downloads if available, preserves the saved generated queries,
and recomputes reranker predictions. On a fresh checkout it downloads the pinned corpus and
models. Outputs go to `reproductions/my_reproduction/`, with metric differences in `comparison.json`.
A quick BM25-only check is:

```sh
python scripts/reproduce.py --baseline-only --name my_baseline
```

For an unfinished workspace with its local runtime artifacts present, resume using
`python scripts/run_core.py --include-final`. The original run IDs are immutable checkpoints;
use the isolated reproduction command on a fresh clone containing saved results.

Ask your own question after corpus preparation and BM25 indexing:

```sh
convrag prepare
convrag index --backend bm25
convrag search --domain cloud --question "How can I deploy a Python application?" --rerank
```

For a follow-up, supply `--history-file history.json` (an array of `speaker`/`text` messages)
or `--standalone-query "your resolved question"`. This interactive command returns passages;
benchmark experiments use the separately cached automatic rewriting pipeline. Inspect a genuine
saved example with `python scripts/demo.py --run R00_bm25_latest_rerank` after running experiments.

## Optional neural corpus indexes

The current completed core uses BM25, rewriting, fusion and a compact cross-encoder.
SPLADE and BGE-small are implemented and smoke-tested; their full-corpus quality is not reported.
Sample CPU timings estimated approximately 10.8 and 3.2 indexing hours respectively. Run these
as a separate development study if accelerator resources or extra time are available:

```sh
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
```

Keep these exploratory results separate from the frozen core study. Full index construction is
resumable; partial pilot indexes cannot be used for benchmark evaluation. Model revisions are
pinned in `analysis/model_inventory.json`. MLX is optional (`pip install -e '.[apple]'`), requires
accessible Apple Metal, and is selected with an `mlx-community/...` rewrite model. The default
CPU-compatible rewriter is Qwen2.5-0.5B-Instruct.

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
are intentionally regenerated, rather than committed. `PROJECT_PLAN.md` is the updated milestone tracker. Generated query snapshots in `artifacts/queries/` preserve the exact rewrites used for reproducibility; they contain no gold targets or relevant-passage labels.

## Deliverables and checks

- [Recorded results](RESULTS.md) and [updated milestone tracker](PROJECT_PLAN.md).
- [Report source](report/PROJECT_REPORT.md) and [PDF](report/PROJECT_REPORT.pdf).
- [Walkthrough script](walkthrough/WALKTHROUGH.md), [automated narrated review video](walkthrough/AUTOMATED_WALKTHROUGH.mp4), and [transcript](walkthrough/NARRATION.md).
- `analysis/verification.json` records independent metric/coverage checks once all final runs finish.
- `analysis/clean_environment_reproduction.json` records the exact BM25 match in a separate Python environment.

The video uses a synthetic stock voice and actual saved results. It is a review aid; check the
course instructions before using it as a submission. The separate final brief, exact cutoff,
report format and student-presentation requirements remain unconfirmed. Nothing has been
uploaded to a course submission portal.

After all runs and the report are generated, run `python scripts/verify_artifacts.py` to
recompute metrics and check the frozen protocol. To rebuild the optional video, install
`walkthrough/requirements.txt`, download the checksum-verified Kokoro assets from the release
linked in `walkthrough/narration_metadata.json` into `cache/tts/`, and run
`python walkthrough/build_video.py` on macOS with FFmpeg installed. Retrieval itself does not
require these video tools or TTS models.

## Sources

- [IBM MTRAG benchmark, data and evaluation](https://github.com/IBM/mt-rag-benchmark)
- [SPLADE implementation and publications](https://github.com/naver/splade)
- [SPLADE++ model](https://huggingface.co/naver/splade-cocondenser-ensembledistil)
- [BGE-small model and retrieval instructions](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Upstream data and pretrained models retain their respective licenses. The IBM benchmark Apache-2.0 license is included in `third_party/IBM_BENCHMARK_LICENSE` for the derived query snapshots. This repository contains
original pipeline code; it does not redistribute benchmark corpora or model weights.
