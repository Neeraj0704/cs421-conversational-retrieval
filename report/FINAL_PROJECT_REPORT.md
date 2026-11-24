# CS421: Improving multi-turn passage retrieval

Combined CPU and GPU experiment report, November 24, 2026. Prepared from verified saved runs. Course-format review remains pending. The earlier CPU-stage report is preserved separately as PROJECT_REPORT.md and PROJECT_REPORT.pdf.

## Task and main finding

The system implements SemEval-2026 Task 8 / MTRAGEval Subtask A: given a conversation ending in a user question, return ten ranked original passages from its assigned collection. It retrieves evidence; answer generation is outside this subtask.

The prespecified GPU primary, E08 hybrid fusion, combines SPLADE learned sparse retrieval and BGE dense retrieval across the original question and a generated standalone rewrite. It achieved validation nDCG@5 0.4479 and final follow-up nDCG@5 0.4334 (submitted-domain-weighted: 0.4379). Final Recall@10 is 0.6211. The earlier BM25 baseline scored 0.2896 and BM25 plus reranker scored 0.4161 on the same released final set.

These are local benchmark measurements. Historical leaderboard equivalence is unverified. Earlier CPU results on this public final set were already known before the GPU study; the GPU final comparisons are disclosed follow-up benchmarks, not a fresh unseen holdout. GPU model selection used validation scores only, and all three configurations were frozen before their new final runs.

## Research question and implementation

We test whether conversation context, complementary retrieval signals and reranking improve passage finding on a local machine. Research supplies hypotheses; controlled comparisons determine which tested components help. No claim is made that this implementation is the best available system or that 50 primary papers were independently verified.

BM25 uses English stopwords, Snowball stemming and Lucene-style k1=1.2, b=0.75. The local Qwen2.5-0.5B-Instruct rewriter receives the available speaker-labelled conversation prefix and conservatively resolves references while preserving entities and constraints. It is instructed not to answer. Original queries remain a separate search branch, because rewrites can omit useful terms or retain unresolved references. Exact generated queries are saved for replay.

SPLADE uses naver/splade-cocondenser-ensembledistil, masked maximum log(1+ReLU) pooling, length 256 and 128 retained terms per document. BAAI/bge-small-en-v1.5 uses normalized 384-dimensional CLS vectors, length 256, its documented query instruction and exact inner-product search. Model revisions are pinned in analysis/model_inventory.json. Corpus text, passage boundaries and original offset-bearing identifiers are preserved.

Each search branch retrieves up to 100 passages. Reciprocal rank fusion uses k=60 and retains a final candidate pool of 100; the top ten are returned. E08 fuses sparse and dense searches for both original and rewritten questions and does not apply the cross-encoder. Other comparisons use cross-encoder/ms-marco-MiniLM-L6-v2 (length 384), either pure reranking or a 50/50 retrieval/reranker rank blend. Reranking is tested, not assumed to improve every retriever.

## Data and evaluation protocol

The IBM benchmark is pinned at commit e0970de1b1207c2a9506adf238e6e1a22037e38a. Full collections contain 366,479 passages: CLAPNQ 183,408; Cloud 72,442; FiQA 61,022; Government 49,607. Each task searches only its assigned collection. Source archives, model revisions, configuration hashes and source hashes are recorded.

MTRAG contains 842 development prefixes across 110 conversations. A seed-42, domain-stratified split holds out 22 complete conversations: 174 validation inputs, of which 161 have relevance judgments. No conversation crosses the partition. A field whitelist excludes gold targets, relevant-context lists, labels and supplied benchmark rewrites from model inputs.

MTRAG-UN has 507 final inputs, of which 332 are judged and 175 are unscored by this release. All configurations return ten unique original IDs for every input. The direct nDCG@5 mean over judged questions is reported separately from the upstream-style aggregate, which rounds domain metrics to five decimals then weights by submitted domain counts. These two aggregates must not be mixed. Recall@10 measures returned coverage; candidate Recall@100 diagnoses retrieval coverage before final truncation.

Official format checks passed for every recorded run. The unmodified upstream retrieval metric function matched the local wrapper on the baseline. Independent checks recomputed stored metrics, checked corpus membership, exact task coverage and unchanged frozen source/configurations. Per-query output and saved query snapshots support auditing. Missing judged tasks are not silently dropped.

## CPU-stage validation

All rows use the same 161 judged validation questions. Identifiers E00/E01/E02 denote latest/history/rewrite BM25; B02 denotes original-plus-rewrite BM25 fusion; B03 adds reranking; B04 blends retrieval and reranker ranks; R00 reranks latest-question BM25.

| Run | nDCG@5 | Recall@10 | Candidate R@100 |
|---|---|---|---|
| B02_bm25_fusion | 0.2472 | 0.3516 | 0.6271 |
| B03_bm25_rerank | 0.3464 | 0.4388 | 0.6271 |
| B04_bm25_blend | 0.3138 | 0.4246 | 0.6271 |
| E00_bm25_latest | 0.2249 | 0.3265 | 0.5688 |
| E01_bm25_history | 0.1842 | 0.2816 | 0.5404 |
| E02_bm25_rewrite | 0.2114 | 0.3156 | 0.5896 |
| R00_bm25_latest_rerank | 0.3898 | 0.4679 | 0.5688 |

Raw history and rewrite-only BM25 underperformed latest-question BM25. Original-plus-rewrite fusion helped, and the strongest CPU configuration was R00. This supports preserving the original wording and testing context use rather than blindly appending every turn. Individual failures in the CPU-stage report include topic dilution from earlier questions and a fragment whose topic is recoverable only from history.

## GPU-stage validation

The completed twelve-configuration validation study uses full neural indexes. D00/D01 are dense latest/fusion. E03/E04/E05 are SPLADE latest/rewrite/fusion; E06/E07 add pure/blended reranking to SPLADE fusion. E08 is sparse+dense original/rewrite fusion; E10 adds a keyword view to SPLADE fusion. G01/G02 rerank latest-question sparse/dense retrieval; G03 reranks latest-question hybrid retrieval. Full configurations are saved in experiments/gpu_extension_v1/configs/.

| Run | nDCG@5 | Recall@10 | Candidate R@100 |
|---|---|---|---|
| D00_dense_latest | 0.3850 | 0.4771 | 0.7122 |
| D01_dense_fusion | 0.3966 | 0.5204 | 0.7481 |
| E03_splade_latest | 0.3990 | 0.5330 | 0.7243 |
| E04_splade_rewrite | 0.3611 | 0.5093 | 0.7510 |
| E05_splade_fusion | 0.4107 | 0.5581 | 0.7925 |
| E06_splade_rerank | 0.3795 | 0.4976 | 0.7925 |
| E07_splade_blend | 0.4121 | 0.5514 | 0.7925 |
| E08_hybrid_fusion | 0.4479 | 0.5860 | 0.7917 |
| E10_keyword_fusion | 0.3909 | 0.5511 | 0.7878 |
| G01_splade_latest_rerank | 0.4281 | 0.5525 | 0.7243 |
| G02_dense_latest_rerank | 0.4366 | 0.5494 | 0.7122 |
| G03_hybrid_latest_rerank | 0.4291 | 0.5551 | 0.7304 |

E08 was selected by validation nDCG@5, before any new GPU final scores were available. Its paired difference against the earlier CPU R00 is +0.0581, with a 95% conversation-bootstrap interval [+0.0138, +0.1033]. Selection among twelve configurations can still create validation optimism. Pure reranking did not consistently help: SPLADE fusion outperformed its pure-reranked variant, while latest-query sparse and dense reranking were stronger family comparators.

## Final comparisons and uncertainty

The following CPU configurations were frozen and evaluated before the GPU extension:

| Run | nDCG@5 mean | nDCG@5 weighted | Recall@10 | Candidate R@100 |
|---|---|---|---|---|
| FINAL_B02_bm25_fusion | 0.3306 | 0.3332 | 0.4872 | 0.8187 |
| FINAL_E00_bm25_latest | 0.2896 | 0.2918 | 0.4261 | 0.6989 |
| FINAL_R00_bm25_latest_rerank | 0.4161 | 0.4188 | 0.5347 | 0.6989 |

The GPU follow-up retained the validation-selected primary FINAL_E08_hybrid_fusion and the strongest sparse and dense family comparators. Primary status was not reassigned after test scores were read. All rows below score the same 332 judged questions out of 507 inputs:

| Run | nDCG@5 mean | nDCG@5 weighted | Recall@10 | Candidate R@100 |
|---|---|---|---|---|
| FINAL_E08_hybrid_fusion | 0.4334 | 0.4379 | 0.6211 | 0.8610 |
| FINAL_G01_splade_latest_rerank | 0.4199 | 0.4239 | 0.5470 | 0.7570 |
| FINAL_G02_dense_latest_rerank | 0.4040 | 0.4082 | 0.5309 | 0.6810 |

The primary improves mean nDCG@5 by 0.1438 over plain BM25, with a 95% conversation-bootstrap interval [+0.1017, +0.1879]. Compared with the stronger earlier BM25 reranker, the improvement is 0.0173, with interval [-0.0178, +0.0525]. This interval includes zero, so the small ranking improvement over the earlier reranker is not statistically resolved by this analysis. Bootstrap uses 3,000 whole-conversation resamples with seed 42; intervals are exploratory and do not correct for multiple comparisons.

The final hybrid candidate Recall@100 is 0.8610, compared with 0.6989 for latest-question BM25. Candidate coverage does not guarantee correct ordering: the final top-five score remains substantially below perfect. Domain-level nDCG@5 varies: fiqa 0.3086; cloud 0.4608; clapnq 0.4885; govt 0.4364.

## Resources, runtime and reproducibility

Normal macOS Terminal exposed Apple Metal/MPS and passed actual tensor computation and CPU/GPU agreement checks. The agent's separate execution process could not access Metal. GPU work was launched by the user through the provided Terminal scripts, with CPU fallback disabled. Dense indexing took about 36 minutes and SPLADE indexing about 156 minutes. Both cover all 366,479 passages; the full validation workflow took about 3 hours 19 minutes including indexing.

The final primary took 44.0 seconds for 507 tasks, with median recorded query time 0.0629 seconds and p95 0.1098 seconds. These are local measured search timings with cached rewrites, including first-query warmup; they exclude rewriting and index construction. Model inference can use MPS, while sparse matrix and dense vector search run on CPU. Cached reranker timings and fresh reranker timings should not be treated as controlled hardware comparisons.

Frozen inference/evaluation source was preserved. The root CPU run artifacts and freeze manifest remain intact; the GPU study has a separate workspace and protocol. scripts/reproduce.py provides an isolated CPU reproduction; the GPU study/final launchers restore runtime assets and reuse completed checkpoints. A separate clean Python environment reproduced BM25 exactly. Large indexes, model weights and corpora are regenerated rather than committed.

## Limitations and remaining work

The compact rewriter, sparse model, dense encoder and MS MARCO reranker are pretrained and not adapted to these domains. Truncation, SPLADE pruning, incomplete judgments and fixed candidate budgets limit performance. The GPU follow-up reuses a previously inspected public final set. Historical winner comparisons still require matching the exact original question subset, qrels, filters and aggregation; these results do not establish an official rank or a competition win.

ColBERT, fine-tuning, stronger rewriters and adaptive retrieval remain future experiments. They should use a new prespecified development protocol and a genuinely untouched test set if available. The present outcome is a reproducible system with measured improvements and documented negative results, not proof of a universally best architecture.

The student must review the implementation and adapt the report, authorship details and presentation to the final course brief. No course submission has been made. AI assistance was used for implementation and checks; disclose it as required by the course. The accompanying synthetic-voice video is explicitly an automated review aid.

## References

IBM MTRAG benchmark, data and evaluation: https://github.com/IBM/mt-rag-benchmark

Rosenthal et al. (2026), SemEval-2026 Task 8: MTRAGEval: Evaluating Multi-Turn RAG Conversations: https://aclanthology.org/2026.semeval-1.447/

Formal et al. (2022), From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective: https://arxiv.org/abs/2205.04733

SPLADE implementation and model: https://github.com/naver/splade ; https://huggingface.co/naver/splade-cocondenser-ensembledistil

BGE model and retrieval instructions: https://huggingface.co/BAAI/bge-small-en-v1.5

Cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

Qwen rewriter: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Cormack, Clarke and Buettcher (2009), Reciprocal Rank Fusion: https://doi.org/10.1145/1571941.1572114
