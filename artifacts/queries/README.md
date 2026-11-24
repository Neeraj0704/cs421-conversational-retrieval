# Exact generated queries

These JSONL snapshots preserve the generated search questions used in the recorded
experiments. Each row includes a task ID, rewrite, prompt/input cache fingerprint,
model revision and inference timing. Gold answers, reference-passage lists and task
labels are excluded from rewrite inputs.

`mtrag.jsonl` contains 174 unique validation tasks, with three identical repeat rows
from an interrupted/resumed run. The repeats have the same text and do not change
the query lookup. `mtragun.jsonl` contains 507 unique final tasks. The verification
report checks that there are no conflicting rewrites.

The reproduction script copies these files into its isolated cache to preserve the
exact query formulation. Pretrained model weights are not included. The underlying
benchmark is attributed to IBM and its Apache-2.0 license is included in
`third_party/IBM_BENCHMARK_LICENSE`.
