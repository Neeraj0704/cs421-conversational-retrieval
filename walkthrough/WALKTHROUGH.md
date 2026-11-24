# Code walkthrough and recording script

This is a recording guide, not a claim that a student-narrated video has been recorded.
Adapt its duration after checking the final course instructions.

1. Explain the task: given a conversation and its current question, return ten existing passages.
   Show the original question, a contextual rewrite and the assigned domain in saved traces.
2. Open `src/convrag/data.py`. Explain the pinned corpus version, ID preservation,
   field whitelist, conversation split and separate final data.
3. Open `src/convrag/queries.py`. Show the rewrite prompt and original-plus-rewrite views.
4. Open `src/convrag/retrieval.py`. Explain BM25 and reciprocal rank fusion.
5. Open `src/convrag/neural.py`. Explain local rewriting and query-passage reranking.
   Distinguish implemented neural index backends from completed full-corpus experiments.
6. Open `src/convrag/evaluation.py`. Explain nDCG@5, recall, scored coverage and two aggregates.
7. Run `python -m pytest -q` and `python scripts/official_parity.py`.
8. Show `runs/E00_bm25_latest/metrics.json` and `report/validation_results.png`.
   State the best completed validation result: 0.2249 nDCG@5.
   Explain negative results and uncertainty rather than presenting only the best row.
9. Show `analysis/frozen_selection.json` if present, then the final result table if scored.
   There are currently 0 completed final runs; do not imply historical leaderboard equivalence.
10. End with limitations: CPU-only session, small rewriter, retrieval candidate ceiling,
    incomplete judgments, deferred accelerator experiments and unconfirmed final brief.

Reproduce a run using a NEW config ID, to preserve original experiment artifacts.
Do not show credentials, environment secrets, or private account pages in a recording.
