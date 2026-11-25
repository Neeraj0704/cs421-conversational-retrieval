# Full student recording script

Target: approximately 4 minutes 40 seconds, including navigation. Maximum: five minutes.
Read only the quoted speech. File names, times and actions are directions, not narration.
Use your own voice and pause briefly between sections. Timings are rehearsal targets;
check an actual timed practice before recording the submission.

## Set up before recording

1. In VS Code, choose File > Open Folder and open:
   `/Users/neerajvp/Documents/CS421 NLP/NLP/implementation`.
2. Increase the editor font to about 18 to 20 points. Option+Z toggles word wrap.
   Command+B hides the Explorer sidebar when you need more room.
3. To jump to code, press Command+P, paste the exact path and line shown below, then Return.
   The number after the colon is a line number; it is not part of the filename.
4. Open each destination once before recording so navigation is familiar. Keep the script
   on a second device or printed beside you. Record the VS Code window with microphone audio.
5. No indexing, model download or benchmark rerun is needed during the recording.
   The example file is already prepared from a genuine saved validation run.

Optional terminal preparation, before recording:

```sh
cd "/Users/neerajvp/Documents/CS421 NLP/NLP/implementation"
../.venv/bin/python -m json.tool walkthrough/RECORDING_EXAMPLE.json
```

This only displays saved output; it does not perform inference. The narrated route below
opens the JSON in the editor, so using the terminal is optional.

## 0:00 to 0:30 — Task and selected configuration

Open with Command+P:
`experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json`

Keep the complete short configuration visible. Point to `backends`, `query_mode`,
`pool_size` and `retrieve_k` as you explain the components.

> Hi, I'm Neeraj. My project finds relevant passages for questions in a conversation. It takes the conversation so far and returns ten passages for the latest question.
>
> This configuration shows the selected system. It searches with both SPLADE and a dense retriever, using the original question and a rewrite. It combines their rankings before returning ten results. This version doesn't use a reranker.

## 0:30 to 1:05 — Inputs and first assumption

Command+P: `src/convrag/data.py:45`

Show `tasks`, especially the collection check and the dictionary created at lines 50 to 57.
Move the pointer over `Collection`, `domain` and the speaker/text input fields.

> Before searching, I prepare the input here. I keep the conversation and its assigned collection, while leaving out gold answers and relevance labels. Those labels are used later for evaluation.
>
> The first assumption is that the correct collection is already provided. A Cloud question searches Cloud passages. The system isn't choosing a collection or searching all four at once.

## 1:05 to 1:45 — Rewriting and second assumption

Command+P: `src/convrag/queries.py:4`

Show the prompt and `latest` check. At “I also keep,” jump with Command+P to
`src/convrag/queries.py:33`; show the `fusion` branch returning original and rewrite.

> With the input ready, the next issue is understanding follow-up questions. I use a small Qwen model to rewrite the latest question so it makes sense on its own. This prompt asks it to resolve references without adding facts.
>
> A second assumption is that the supplied conversation ends with a user question. This check enforces that.
>
> I also keep the original wording here, because a rewrite can misunderstand the conversation. Both query versions go into retrieval.

## 1:45 to 2:40 — Representations and search

Command+P: `src/convrag/neural.py:51`

Show `Encoder.encode`. Point first to the SPLADE branch, then to the normalized dense
vector on line 68. Point to truncation on line 54 when discussing the token limit.
At “The search method,” jump to `src/convrag/neural.py:118` and show similarity scores.

> Those queries are converted into numerical representations in this encoder.
>
> SPLADE produces weighted vocabulary terms. That retains useful word-level matching while allowing learned expansion. BGE represents meaning as a normalized vector of 384 numbers. The idea is that the two searches can find useful passages that the other misses.
>
> For local resources, the encoder reads at most 256 tokens, and SPLADE documents keep 128 terms. That's another assumption: the retained content carries enough useful information. Longer passages can lose details.
>
> The search method compares a query representation with the stored passage representations and returns the strongest candidates.

## 2:40 to 3:30 — Fusion and the full prediction path

Command+P: `src/convrag/retrieval.py:11`

Show the short `rrf` function. At “Back in,” jump to `src/convrag/cli.py:58`.
Point to the nested searches, fusion and pool limit at lines 58 to 63; then scroll to
`top=ranked[:10]` and the prediction dictionary at lines 74 to 77.

> Once I have the rankings, I combine them with reciprocal rank fusion. It rewards passages that rank highly across searches. Using positions avoids directly mixing model scores that have different scales.
>
> Back in the main loop, you can follow the whole prediction path: create the query views, search each index, combine the lists, keep a hundred candidates, and return the first ten original passage IDs.
>
> This configuration was selected using validation results. Reranking was tested separately, but isn't part of this selected path.

## 3:30 to 4:00 — One actual saved example

Command+P: `walkthrough/RECORDING_EXAMPLE.json`

Show the question, two query views and the first passage excerpt. Point to the explanatory
sentence in the passage. Keep the source label visible so this is clearly a saved example.

> Here's an actual saved validation example: “What are worker nodes?” The first retrieved passage explains that worker nodes carry the deployments and services of an application.
>
> The rewrite only changes capitalization here, so I wouldn't credit the improvement to resolving a reference. This file displays an existing result; I'm not rerunning the full benchmark during the recording.

## 4:00 to 4:40 — Results and an honest closing

Command+P:
`experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json:64`

Show `mean_over_scored_queries`; point to `ndcg_cut_5` on line 71. If space permits,
show the separate weighted aggregate below it, but do not read every number aloud.

> Across the final judged questions, the selected system scored point four three three four on nDCG at five. That measures ranking quality; it isn't a percentage of correct answers.
>
> I evaluated nineteen validation configurations. The hybrid improved clearly over my plain BM25 baseline, while its smaller gain over BM25 with reranking remained uncertain.
>
> The main tradeoff is that extra context can help, but rewrites can also invent details. More reliable rewriting would be my next step. Thank you.

## Rehearsal and submission check

- Speak at a comfortable pace. This is about 513 spoken words, plus short navigation pauses.
  If a practice exceeds five minutes, first omit the optional final sentence about reranking
  in the fusion section and shorten the example commentary; keep both named assumptions
  and the sparse/dense representation explanation.
- Distinguish representations, search, fusion and final output. The selected path is not
  BM25 plus a reranker, and it does not generate an answer.
- The worker-node example is from validation; 0.4334 is the separate final benchmark mean.
  The final dataset has 507 inputs and 332 judged questions. Historical leaderboard
  equivalence remains unverified; do not claim an official competition rank.
- Watch the finished video at normal size, checking code legibility, audio and duration.
  Submit an accessible shareable link in the separate Canvas assignment. The brief gives
  a noon CST deadline on November 25, 2026, and says late cards do not apply to the video.
