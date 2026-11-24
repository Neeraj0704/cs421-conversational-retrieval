# GPU final follow-up benchmark

Follow-up benchmark on the previously used public final set; not a fresh unseen holdout.
All three configurations were frozen from validation results before their GPU final runs. No parameters are changed using these new test scores.

Prespecified primary: **FINAL_E08_hybrid_fusion**.

| Run | nDCG@5 mean | nDCG@5 weighted | Recall@5 | Recall@10 | Scored / inputs |
|---|---|---|---|---|---|
| FINAL_E08_hybrid_fusion | 0.4334 | 0.4379 | 0.4756 | 0.6211 | 332 / 507 |
| FINAL_G01_splade_latest_rerank | 0.4199 | 0.4239 | 0.4491 | 0.5470 | 332 / 507 |
| FINAL_G02_dense_latest_rerank | 0.4040 | 0.4082 | 0.4281 | 0.5309 | 332 / 507 |

Earlier CPU references on this release: BM25 latest = 0.2896; BM25 latest + reranker = 0.4161 mean nDCG@5.
Historical leaderboard equivalence remains unverified. The strongest final score is not used to replace the prespecified primary after testing.
