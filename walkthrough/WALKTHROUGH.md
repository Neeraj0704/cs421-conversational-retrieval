# Student code walkthrough recording guide

For exact spoken narration and navigation cues, use [RECORDING_SCRIPT.md](RECORDING_SCRIPT.md).

Record your own explanation of the implemented system in VS Code. Aim for 4 minutes
45 seconds; the assignment maximum is five minutes. A camera is not required.
The existing synthetic-voice video is a rehearsal aid, not a recording of your explanation.

## Prepare before recording

Open the implementation folder in VS Code. Enlarge code and terminal text to a readable
size, approximately 18 to 20 points. Keep only the tabs below open. Use a selected-window
or selected-area screen recording with microphone audio enabled. Check a short audio sample.

The final selected system is E08: original/rewrite queries through SPLADE and dense search,
combined with RRF. It does not use a cross-encoder reranker or BM25 in its selected path.
BM25 and rerankers were comparison experiments. All full neural indexes are complete.

## Recording sequence

| Time | What to show | Explain in your own words |
|---|---|---|
| 0:00 to 0:25 | Optional diagram, then selected configuration | Given the conversation and assigned collection, return ten existing passages. Original and rewritten questions feed sparse and dense search; rankings are fused. |
| 0:25 to 1:05 | `experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json`; `src/convrag/cli.py`, lines 48 to 78 | The configuration selects both retrievers and query fusion. Trace query views through search calls, RRF, the 100-candidate pool and final ten IDs. This selected configuration has no reranker. |
| 1:05 to 1:45 | `src/convrag/queries.py`, prompt and `query_views` | The small model resolves follow-up references. The original query is retained because a rewrite can lose terms or invent context. Saved rewrites make comparisons repeatable. |
| 1:45 to 2:55 | `src/convrag/neural.py`, `Encoder.encode` and `NeuralIndex.search`; `src/convrag/retrieval.py`, `rrf` | Explain representations: SPLADE weighted vocabulary terms versus BGE's normalized 384-number vectors. Show sparse/dense similarity search and how reciprocal ranks combine signals on different score scales. |
| 2:55 to 3:40 | `src/convrag/data.py`, `COLLECTIONS` and `tasks`; `src/convrag/queries.py`, `latest`; `Encoder.encode` truncation | Explicitly identify at least two assumptions and point to their code. Suggested assumptions appear below. |
| 3:40 to 4:25 | One saved query/output; `experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json` | Show an actual retrieved passage and explain the measured result: 0.4334 mean nDCG@5, 0.4379 under the separate weighted aggregate, and Recall@10 0.6211. Predictions cover 507 inputs; 332 are judged. |
| 4:25 to 4:45 | Code or result view | End with a tradeoff: rewrites can invent context, and truncation/pruning can discard useful evidence. Mention that the small gain over BM25 plus reranking remains uncertain. |

## Assumptions to state explicitly

Choose at least two, name them as assumptions, and show the implementation:

- The input identifies the correct one of four collections. `data.py` maps collections;
  `cli.py` groups tasks by domain and searches that collection. This is not unrestricted
  search across every collection.
- The supplied prefix ends with the current user question and contains only available
  past turns. `queries.latest` checks the final speaker; `data.tasks` whitelists the inputs
  and excludes gold fields. The system has no future conversation turns.
- Neural encoders read at most 256 tokens per input, and SPLADE documents retain 128
  terms. Show token truncation and top-k pruning in `Encoder.encode`. This makes local
  indexing practical but may lose useful content later in a long passage.
- The chosen retrievers and tokenizer setup assume English text. English model choices
  and BM25 English stopword/stemming settings reflect that assumption; multilingual
  performance has not been established.

## Showing an example

A live inference run is optional; the assignment is a code explanation. It is acceptable
to inspect genuine saved predictions and clearly call them saved results. Do not spend
recording time rebuilding indexes or rerunning the full benchmark.

For a saved validation example, run this before recording from the implementation folder:

```sh
../.venv/bin/python scripts/demo.py \
  --run ../experiments/gpu_extension_v1/runs/E08_hybrid_fusion \
  --task-id '747a810abfd6da4a9c37cdb74feec95e<::>2'
```

This displays saved output; it does not run the retrievers again. The example asks
"what are worker nodes?" and the hybrid retrieves the three judged passages at ranks
1 to 3. Its rewrite only changes capitalization, so do not claim that the improvement
came from resolving a pronoun. The full JSON is lengthy: prepare the relevant input,
query views and first passage on screen before recording.

For a failure, the report's scoring-system example shows the rewriter inventing an exam
setting. Use one example in the video; all four need not be narrated.

## Before submission

Watch the recording at normal playback size: can you read the code and hear the narration?
Confirm it is at most five minutes and explicitly covers architecture, representations
and at least two assumptions. Upload a shareable video and check viewer access, then
submit the link in the separate Canvas walkthrough assignment. The final brief states
November 25, 2026 at noon CST; late cards do not apply to the video. No upload or course
submission has been performed by these scripts.
