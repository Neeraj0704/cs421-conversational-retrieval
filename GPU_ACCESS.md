# Apple GPU access and first benchmarks

The installed PyTorch build includes MPS. The current restricted execution session
reports no native Metal device and `mps_available: false`. A successful matrix
multiplication on `device='mps'` is required before any GPU benchmark.

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
