# Student code walkthrough recording guide

For expanded spoken narration, function explanations and navigation cues, use
[RECORDING_SCRIPT.md](RECORDING_SCRIPT.md). It is an editing master, longer than five minutes;
shorten it after rehearsing. The table below remains a suggested five-minute allocation.

The architecture diagram is available as [PNG](ARCHITECTURE.png), [PDF](ARCHITECTURE.pdf),
and [SVG](ARCHITECTURE.svg). Show it during the opening 35 seconds, then switch to code.
The checked file roles and example provenance are recorded in [DEMO_VERIFICATION.md](DEMO_VERIFICATION.md).

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
| 0:00 to 0:35 | `walkthrough/ARCHITECTURE.png` | Trace input, two query views, sparse and dense searches, fusion and ten passage IDs. |
| 0:35 to 1:10 | Selected configuration; `src/convrag/data.py:45` | Identify both retrievers, explain the input whitelist and state the assigned-collection assumption. |
| 1:10 to 2:00 | `src/convrag/queries.py:4`; `src/convrag/neural.py:160`; `src/convrag/queries.py:33` | Explain the rewrite prompt, final-user-turn assumption, actual generation call, saved rewrites and original query retention. |
| 2:00 to 2:50 | `src/convrag/neural.py:51` and `:118` | Explain sparse/dense representations, truncation and document pruning, then similarity search. |
| 2:50 to 3:40 | `src/convrag/retrieval.py:11`; `src/convrag/cli.py:58` | Explain RRF, the nested searches, the 100-candidate pool and final ten original IDs. |
| 3:40 to 4:05 | `walkthrough/RECORDING_EXAMPLE.json` | Show the saved validation question and passage, clearly identifying this as saved output. |
| 4:05 to 4:45 | Final `metrics.json:64` | Explain 0.4334 mean nDCG@5, the validation comparisons, uncertainty over the stronger baseline and rewrite limitations. |

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
