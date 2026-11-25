# Walkthrough code and artifact verification

Checked against the selected final E08 run on 2026-11-25. These are presentation checks;
no models, full indexing or benchmark runs were started.

## Files used in the recording

| File | Relevant implementation or artifact |
|---|---|
| ARCHITECTURE.png | The selected original/rewrite → SPLADE and BGE → RRF → ten original IDs flow |
| experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json | Actual selected configuration; sparse+dense, query fusion, 100 per search/pool, RRF k=60, no reranker |
| src/convrag/data.py, lines 45–60 | Input whitelist and provided-collection validation |
| src/convrag/queries.py, lines 4–14 and 33–40 | Rewrite prompt, last-user-turn guard and original/rewrite deduplication |
| src/convrag/neural.py, lines 160–166 | Actual Qwen generation call used before retrieval |
| src/convrag/neural.py, lines 51–69 and 111–122 | Sparse and dense representations, document pruning, stored-vector search and original-ID lookup |
| src/convrag/retrieval.py, lines 11–20 | Reciprocal rank fusion, separate from the BM25 comparison class below |
| src/convrag/cli.py, lines 58–77 | Query/retriever loops, fusion, candidate pool and top-ten prediction serialization |
| walkthrough/RECORDING_EXAMPLE.json | Concise display of a genuine saved E08 validation result, not executable inference |
| experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json, lines 64–72 | Final mean-over-judged-query metrics, separate from the weighted aggregate |

## Checks passed

- All nine source files match their frozen SHA-256 hashes. The demo shows the evaluated implementation.
- The final configuration matches its frozen fingerprint and the saved final-run manifest.
- The configured path uses SPLADE and BGE, original/rewrite fusion, and no reranker.
- The worker-node question, collection, two query strings and first predicted passage ID match the original validation inputs, traces and predictions.
- The excerpt matches the source corpus passage after trimming leading whitespace.
- Final mean nDCG@5 rounds to 0.4334; 332 questions are judged out of 507 inputs.
- There are seven saved CPU and twelve saved GPU validation configurations, nineteen in total.
- The expanded function-by-function revision has sixteen code/result navigation destinations;
  all exist and their line numbers are in range. Named functions were checked against their definitions.
- Expanded narration has 1297 words. It is an editing master rather than a five-minute cut;
  the speaker will shorten it and check duration in rehearsal.

The saved example is illustrative; it does not establish the benefit of rewriting because
its rewrite changes capitalization only. The GPU final study is a disclosed follow-up on
a previously inspected public set. It is not a fresh unseen holdout or verified official
leaderboard comparison. BM25/reranker comparisons are discussed as results, not shown as
components of the selected system.
