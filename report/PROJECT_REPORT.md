# CS421: Improving multi-turn passage retrieval

Implementation and experiment report. Prepared from saved, format-checked runs; this is a draft pending the separate final-project formatting instructions.

The system receives a conversation up to the latest user question and retrieves ten passages from the assigned collection. It implements SemEval-2026 Task 8 / MTRAGEval Subtask A. It does not generate an answer.

The strongest completed validation run is E00_bm25_latest, with nDCG@5 = 0.2249. This is our local development result, not an official competition submission or a verified win over the historical leaderboard.

## Research question and method

Can conversation context improve retrieval, and when do rewriting, query fusion and cross-encoder reranking help? We test each addition against fixed baselines, rather than assume a larger pipeline always improves quality.

BM25 searches existing passage boundaries using English stopwords, Snowball stemming, Lucene-style scoring, k1=1.2 and b=0.75. Each query searches only its assigned domain.

A local Qwen2.5-0.5B-Instruct model rewrites the current question. A conservative prompt asks it to resolve references and preserve constraints without answering or adding facts. All available past user and assistant turns can be used to resolve references; assistant content is not treated as verified evidence. The generated rewrite is cached with the prompt, model revision and conversation prefix.

Fusion searches the original question and rewrite separately, then combines ranks using reciprocal rank fusion with k=60. This retains an original wording branch when a rewrite omits a useful exact term. The fixed candidate pool contains at most 100 passages.

A pretrained MS MARCO MiniLM-L6 cross-encoder scores query-passage pairs. We compare pure reranking and a 50/50 fusion of retrieval and reranker ranks on the same candidates. Model weights are pretrained; this project does not fine-tune them.

SPLADE++ and BGE-small indexing and search are implemented as optional backends. SPLADE uses maximum log(1+ReLU(logit)) pooling and at most 128 document terms; BGE uses normalized CLS vectors, the documented query instruction, and exact inner-product search. These local choices and 256-token document limits differ from larger unrestricted model configurations.

## Data and separation of development from test

Data are pinned to IBM commit e0970de1b1207c2a9506adf238e6e1a22037e38a. The audited corpus contains 366,479 passages across CLAPNQ, IBM Cloud, FiQA and government collections. Original passage and task IDs are preserved, including suffixes and delimiters.

MTRAG contains 842 task prefixes from 110 conversations. Seed 42 assigns 88 conversations to the development/training partition and 22 to validation, stratified by domain. No model fitting is performed on these partitions. The validation partition contains 174 prefixes; only tasks with released relevance judgments contribute to retrieval metrics.

The current pinned MTRAG-UN release contains 507 input tasks. Test relevance files are not consulted for model or prompt selection. A frozen manifest records final configurations before test evaluation. Released test questions can be rewritten without reading their answers or relevance labels.

A field whitelist removes targets, relevant contexts, answerability labels, question labels and provided benchmark rewrites before retrieval or rewriting. The system uses the task prefix supplied by the benchmark; it does not look at later turns.

## Evaluation and validity checks

The primary development metric is mean nDCG@5 over scored validation questions. Recall@10 measures retrieved relevant passages in the final ten; candidate Recall@100 diagnoses whether reranking had a chance to succeed. Required judged questions are not removed when retrieval fails.

For each run, we separately record submitted, scored and unscored counts, per-domain metrics and the upstream-style domain aggregation weighted by submitted task counts. That weighted result can differ from a direct mean over scored questions.

The unmodified official evaluate function is loaded through Python AST to avoid unrelated generation-judge imports and obsolete hard-coded data paths. Per-query nDCG@1,3,5 and Recall@5 match our wrapper on the baseline. Every completed run also passes the official JSONL format checker and stricter checks for task coverage, original corpus IDs, unique top ten, finite scores and decreasing rank scores.

Prediction scores encode final rank order. Evaluation uses these numeric scores; it does not rely only on list order. Paired uncertainty uses a fixed-seed bootstrap resampling conversations, which avoids treating turns from the same conversation as independent.

Local latency includes search and ranking with possible first-query warmup. Rewrite generation and initial model/index loading are recorded separately or excluded. Cached reranking can make a later blend run appear much faster; those times are not an independent live inference benchmark.

## Development results

| Run | nDCG@5 | Recall@10 | Candidate R@100 | Scored / input | Median search + rank (s) |
|---|---|---|---|---|---|
| E00_bm25_latest | 0.2249 | 0.3265 | 0.5688 | 161 / 174 | 0.0021 |
| E01_bm25_history | 0.1842 | 0.2816 | 0.5404 | 161 / 174 | 0.0008 |

## Uncertainty and interpretation

| Compared with BM25 latest | Delta nDCG@5 | 95% conversation-bootstrap interval |
|---|---|---|
| E01_bm25_history | -0.0408 | [-0.0909, +0.0097] |

Intervals covering zero do not establish a clear directional effect on this small validation sample. Selecting the best of several validation runs also introduces selection optimism; the final test is reserved for the frozen comparisons.

History concatenation can bury the latest intent under earlier topics. Rewriting can resolve references, but the small rewriter may leave pronouns unresolved or change meaning. Fusion protects some exact wording but can add irrelevant candidates. A cross-encoder can only reorder retrieved passages, so candidate recall bounds what reranking can repair.

## Final evaluation

No final test scores have been reported yet. Final testing requires a frozen configuration manifest and explicit final-run mode.

These are results on the currently pinned release. Historical competition equivalence remains unverified unless the exact original question subset, relevance judgments, filtering and aggregation are independently matched. A historical winning score is context, not proof of a fair direct comparison.

## Resources and feasibility

The implementation runs on an Apple Silicon Mac, Python 3.11. The execution session does not expose Metal/MPS, so neural feasibility measurements use CPU. No paid model API or remote GPU has been provisioned.

dense: sampled throughput 31.88 passages/s; estimated full indexing 3.2 hours. These are estimates from small samples, not completed full-index measurements.

splade: sampled throughput 9.42 passages/s; estimated full indexing 10.8 hours. These are estimates from small samples, not completed full-index measurements.

The plan explicitly permits BM25 + rewriting + fusion + a compact reranker when learned sparse indexing fails the resource gate. Full SPLADE/dense quality, ColBERT, retriever fine-tuning, RL and adaptive agent loops are not claimed as completed experiments.

## Limitations and next steps

A compact general rewriter and an MS MARCO reranker are not tuned for these domains. BM25 loses some semantic matches; document token truncation and sparse pruning would limit the optional neural branches. Released relevance judgments can be incomplete.

The broader research reports informed the candidate design, but their paper counts and every quantitative claim have not been independently audited. This implementation is not evidence that at least 50 primary papers were individually verified.

Next work is an accelerator-backed full sparse index, then a controlled dense-complementarity test at the same final candidate budget. Only adopt a component after a measured development benefit. Do not tune after seeing final test scores.

The final course brief is still needed to check page limits, video requirements, assistance disclosure and submission format. This report and the walkthrough materials must be reviewed by the student before course submission. AI assistance was used to implement and check this pipeline; the student should understand, verify and disclose it as required by the course.

## References

IBM MTRAG benchmark and evaluation: https://github.com/IBM/mt-rag-benchmark

Formal et al. (2022), From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective: https://arxiv.org/abs/2205.04733

SPLADE code and model: https://github.com/naver/splade ; https://huggingface.co/naver/splade-cocondenser-ensembledistil (CC-BY-NC-SA-4.0 model)

BGE-small-en-v1.5 model and usage: https://huggingface.co/BAAI/bge-small-en-v1.5 (MIT)

MiniLM cross-encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0)

Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0)

Cormack, Clarke and Buettcher (2009), Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods: https://doi.org/10.1145/1571941.1572114
