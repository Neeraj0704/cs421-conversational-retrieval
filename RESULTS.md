# Recorded results

Only completed runs passing the official format checker are listed.

## Held-out development conversations

| Run | nDCG@5 | Recall@10 | Candidate R@100 | Scored / input | Median search + rank (s) |
|---|---|---|---|---|---|
| B02_bm25_fusion | 0.2472 | 0.3516 | 0.6271 | 161 / 174 | 0.0013 |
| B03_bm25_rerank | 0.3464 | 0.4388 | 0.6271 | 161 / 174 | 0.8993 |
| B04_bm25_blend | 0.3138 | 0.4246 | 0.6271 | 161 / 174 | 0.0026 |
| E00_bm25_latest | 0.2249 | 0.3265 | 0.5688 | 161 / 174 | 0.0021 |
| E01_bm25_history | 0.1842 | 0.2816 | 0.5404 | 161 / 174 | 0.0008 |
| E02_bm25_rewrite | 0.2114 | 0.3156 | 0.5896 | 161 / 174 | 0.0022 |
| R00_bm25_latest_rerank | 0.3898 | 0.4679 | 0.5688 | 161 / 174 | 1.8340 |

Validation-selected best: **R00_bm25_latest_rerank**. Selection uses development data only.

## Frozen final comparisons

| Run | nDCG@5 | Recall@10 | Candidate R@100 | Scored / input | Median search + rank (s) |
|---|---|---|---|---|---|
| FINAL_B02_bm25_fusion | 0.3306 | 0.4872 | 0.8187 | 332 / 507 | 0.0018 |
| FINAL_E00_bm25_latest | 0.2896 | 0.4261 | 0.6989 | 332 / 507 | 0.0014 |
| FINAL_R00_bm25_latest_rerank | 0.4161 | 0.5347 | 0.6989 | 332 / 507 | 1.8638 |

These are local benchmark results, not an official competition submission. Historical leaderboard equivalence is unverified. The full report includes aggregation differences, uncertainty, negative results and resource limits.
