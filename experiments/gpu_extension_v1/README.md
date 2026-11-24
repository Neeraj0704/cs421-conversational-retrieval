# GPU development extension

Status: completed successfully at 2:02 AM Chicago time on November 24, 2026. Both full indexes and all twelve validation configurations passed independent verification.

Launch from normal macOS Terminal:

```sh
bash "/Users/neerajvp/Documents/CS421 NLP/NLP/implementation/scripts/start_gpu_study.command"
```

- Requires a successful actual MPS computation; CPU fallback is disabled.
- Builds full dense and SPLADE indexes with batch size 16 and maximum length 256.
- Preserves original passage IDs, corpus boundaries and the original validation split.
- Runs twelve configurations listed in `analysis/study_manifest.json`.
- Keeps original CPU reports, configurations, predictions and final-test results separate.
- Declares that earlier CPU test scores are already known; all new selection uses MTRAG validation only.
- Writes progress to `analysis/status.json`, detailed logs to `logs/`, and scores to `GPU_EXTENSION_RESULTS.md`.
- Resumes completed indexes/runs; a lock prevents two study runners from running together.

The copied source, shared corpus/model/index paths and query cache are local runtime
files excluded from Git. Recreate them with `python scripts/run_gpu_study.py --prepare-only`.

Best observed validation result: SPLADE + dense fusion, nDCG@5 0.4479 (original CPU validation reference 0.3898). No new final-test evaluation has been performed. See `analysis/full_study_verification.json`.
