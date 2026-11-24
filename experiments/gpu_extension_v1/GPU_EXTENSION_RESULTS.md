# GPU extension: development results

These are MTRAG validation results from a separately declared extension. The original CPU final test has already been evaluated. No new final-test runs are included here.

Original CPU validation reference: BM25 latest + reranker = 0.3898 nDCG@5.

| Configuration | nDCG@5 | Recall@10 | Candidate R@100 |
|---|---|---|---|
| G02_dense_latest_rerank | 0.4366 | 0.5494 | 0.7122 |
| D01_dense_fusion | 0.3966 | 0.5204 | 0.7481 |
| D00_dense_latest | 0.3850 | 0.4771 | 0.7122 |

Only completed, format-checked runs are listed. Query sets must match the original 161 judged validation tasks. GPU encoding is combined with CPU sparse/dense index search. Cached reranking and resumed indexing timings require care when interpreting speed.
