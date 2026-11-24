# GPU final follow-up protocol

Next step: evaluate the configurations selected exclusively from GPU validation results.

| Role | Frozen configuration | Validation nDCG@5 |
|---|---|---:|
| Prespecified primary | E08: SPLADE + dense original/rewrite fusion | 0.4479 |
| Sparse-only comparator | G01: SPLADE latest + reranker | 0.4281 |
| Dense-only comparator | G02: dense latest + reranker | 0.4366 |

All settings, source hashes, model versions and final query snapshots are recorded in
`analysis/frozen_selection.json` before the new GPU final runs. No reindexing, rewriting,
model training or parameter tuning is performed. The primary remains E08 even if a
comparator later obtains a higher final score.

The earlier CPU systems already used the same public final set. These GPU runs are
therefore a disclosed follow-up benchmark, not a fresh unseen holdout. Selection does
not use the new GPU test scores. Original CPU artifacts remain intact.

Run from the normal macOS Terminal that has GPU access:

```sh
bash "/Users/neerajvp/Documents/CS421 NLP/NLP/implementation/scripts/start_gpu_final.command"
```

Allow approximately 10–15 minutes based on the validation runtimes. The launcher
refuses CPU fallback and reuses the full indexes. It saves progress in
`analysis/final_status.json`, verifies all predictions/metrics, and writes
`GPU_FINAL_RESULTS.md`. It never changes the original CPU final-test freeze.
