# Recorded results

Primary: **E08 hybrid fusion** (SPLADE + dense, original + rewritten query; no cross-encoder).
Selected by validation nDCG@5 before the GPU final runs.

## CPU-stage validation

| Run | nDCG@5 | Recall@10 | Candidate R@100 |
|---|---|---|---|
| B02_bm25_fusion | 0.2472 | 0.3516 | 0.6271 |
| B03_bm25_rerank | 0.3464 | 0.4388 | 0.6271 |
| B04_bm25_blend | 0.3138 | 0.4246 | 0.6271 |
| E00_bm25_latest | 0.2249 | 0.3265 | 0.5688 |
| E01_bm25_history | 0.1842 | 0.2816 | 0.5404 |
| E02_bm25_rewrite | 0.2114 | 0.3156 | 0.5896 |
| R00_bm25_latest_rerank | 0.3898 | 0.4679 | 0.5688 |

## GPU-stage validation

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

All validation rows: 161 judged / 174 inputs, 22 held-out conversations.

## Earlier frozen CPU final comparisons

| Run | nDCG@5 mean | nDCG@5 weighted | Recall@10 | Candidate R@100 |
|---|---|---|---|---|
| FINAL_B02_bm25_fusion | 0.3306 | 0.3332 | 0.4872 | 0.8187 |
| FINAL_E00_bm25_latest | 0.2896 | 0.2918 | 0.4261 | 0.6989 |
| FINAL_R00_bm25_latest_rerank | 0.4161 | 0.4188 | 0.5347 | 0.6989 |

## Frozen GPU final follow-up

| Run | nDCG@5 mean | nDCG@5 weighted | Recall@10 | Candidate R@100 |
|---|---|---|---|---|
| FINAL_E08_hybrid_fusion | 0.4334 | 0.4379 | 0.6211 | 0.8610 |
| FINAL_G01_splade_latest_rerank | 0.4199 | 0.4239 | 0.5470 | 0.7570 |
| FINAL_G02_dense_latest_rerank | 0.4040 | 0.4082 | 0.5309 | 0.6810 |

All final rows: 332 judged / 507 inputs. Direct means and submitted-domain-weighted scores are different aggregates.
Primary versus earlier CPU reranker: delta +0.0173, 95% conversation-bootstrap interval [-0.0178, +0.0525]. This interval includes zero, so the small ranking improvement over the earlier reranker is not statistically resolved by this analysis.

The final set was already used for CPU evaluation. GPU scores are disclosed follow-up results, not a fresh unseen holdout. Historical leaderboard equivalence remains unverified.

See [combined report](report/FINAL_PROJECT_REPORT.md), [GPU final verification](experiments/gpu_extension_v1/analysis/final_followup_verification.json), and the preserved [CPU-stage report](report/PROJECT_REPORT.md).
