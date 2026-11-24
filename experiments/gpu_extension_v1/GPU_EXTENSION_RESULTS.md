# GPU extension: development results

These are MTRAG validation results from a separately declared extension. The original CPU final test has already been evaluated. No new final-test runs are included here.

Original CPU validation reference: BM25 latest + reranker = 0.3898 nDCG@5.

| Configuration | nDCG@5 | Recall@10 | Candidate R@100 |
|---|---|---|---|
| E08_hybrid_fusion | 0.4479 | 0.5860 | 0.7917 |
| G02_dense_latest_rerank | 0.4366 | 0.5494 | 0.7122 |
| G03_hybrid_latest_rerank | 0.4291 | 0.5551 | 0.7304 |
| G01_splade_latest_rerank | 0.4281 | 0.5525 | 0.7243 |
| E07_splade_blend | 0.4121 | 0.5514 | 0.7925 |
| E05_splade_fusion | 0.4107 | 0.5581 | 0.7925 |
| E03_splade_latest | 0.3990 | 0.5330 | 0.7243 |
| D01_dense_fusion | 0.3966 | 0.5204 | 0.7481 |
| E10_keyword_fusion | 0.3909 | 0.5511 | 0.7878 |
| D00_dense_latest | 0.3850 | 0.4771 | 0.7122 |
| E06_splade_rerank | 0.3795 | 0.4976 | 0.7925 |
| E04_splade_rewrite | 0.3611 | 0.5093 | 0.7510 |

Only completed, format-checked runs are listed. Query sets must match the original 161 judged validation tasks. GPU encoding is combined with CPU sparse/dense index search. Cached reranking and resumed indexing timings require care when interpreting speed.
