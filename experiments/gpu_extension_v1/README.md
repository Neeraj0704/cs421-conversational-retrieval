# GPU development extension

Status: prepared; full indexing has not been started by the agent.

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
