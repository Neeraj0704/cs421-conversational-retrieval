# Apple GPU access and first benchmarks

The installed PyTorch build includes MPS. The current restricted execution session
reports no native Metal device and `mps_available: false`. A successful matrix
multiplication on `device='mps'` is required before any GPU benchmark.

## Confirmed on this Mac

The user's normal Terminal passed the GPU computation and both model parity checks.
The saved run is `analysis/gpu/20261124T042807Z/`. At batch size 16, measured sample
throughput was about 39.4 SPLADE passages/s and 175.8 dense passages/s. Estimated
full indexing is 2.6 hours and 35 minutes, before experiment and file-processing time.
The agent's separate execution process still reports no native Metal device, despite
the UI setting reported by the user. No further permission changes are requested.

## Start the full GPU development study

Run this yourself in the same normal macOS Terminal that passed the tests:

```sh
bash "/Users/neerajvp/Documents/CS421 NLP/NLP/implementation/scripts/start_gpu_study.command"
```

It builds the dense index first and runs its validation experiments, then builds
SPLADE and runs the remaining comparisons. There are twelve fixed configurations,
including learned sparse retrieval, dense retrieval, hybrid fusion and reranking.
It keeps the Mac awake while running; keep the Mac plugged in with the lid open.
Allow roughly 3–4 hours based on the sample, with actual time dependent on full data.

All new configurations, status and results live in `experiments/gpu_extension_v1/`.
The original CPU results and final-test freeze remain intact. This workflow reads
development relevance judgments only; it does not run another final-test evaluation.
You can rerun the same command to reuse completed indexes and validated runs.
`analysis/status.json` in that study folder records the active stage and log location.

## Confirm access

The desktop app has a permissions control beneath the message composer. Where the
mode is available, Full access removes the local sandbox restrictions and grants broad
filesystem/network access. Changing that setting is a user action; it is not a dedicated
GPU permission switch and is not guaranteed to fix every device-visibility problem.
After a change, rerun:

```sh
../.venv/bin/python scripts/gpu_probe.py --require-mps
```

If a session still cannot see Metal, run the prepared bounded test yourself in normal
macOS Terminal, using the project environment already installed on this machine:

```sh
cd "/Users/neerajvp/Documents/CS421 NLP/NLP/implementation"
./scripts/start_gpu_tests.command
```

This alternative does not require broadening the agent's permissions. The launcher
supports both a `.venv` inside the repository and the existing environment one level above.

## What the launcher does

1. Confirms native Metal/PyTorch GPU visibility and performs a real GPU calculation.
2. Stops immediately if MPS is unavailable; it does not start a long CPU fallback.
3. Samples passages from all four corpora without reading relevance labels.
4. Compares small CPU/GPU query-vector outputs for numerical agreement.
5. Times SPLADE and BGE-small on MPS with batch sizes 8 and 16, with fallback disabled.
6. Writes timestamped results under `analysis/gpu/` for review before full indexing.

The original CPU study, its source/configuration freeze and final scores are preserved.
GPU retrieval experiments are a separate, previously planned extension. Full corpus
indexing and new final-test evaluation are not started by this launcher.

## References

- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Apple verification example: https://developer.apple.com/metal/pytorch/
- Desktop permission modes: https://learn.chatgpt.com/docs/permission-modes
- Sandbox controls: https://learn.chatgpt.com/docs/sandboxing
