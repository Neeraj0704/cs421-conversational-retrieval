# Recording script — verified code route and natural narration

Aim for 4 minutes 45 seconds; the maximum is five minutes. Timings below are practice
targets, not a measured recording. Speak only the blockquotes. All other text is direction.
There are about 540 spoken words. Leave short pauses while changing tabs.

## Prepare the screen

Open /Users/neerajvp/Documents/CS421 NLP/NLP/implementation in VS Code.
Use an 18–20 point font, line numbers and Option+Z word wrap. Hide the sidebar with
Command+B if needed. Keep this script on another device or printed beside you.
Use Command+P, paste a path:line destination, then Return.
Open all destinations once before recording. Fit ARCHITECTURE.png in the image tab.
Record the editor window and microphone. Use your own voice; a camera is optional.

The timed route follows the actual selected E08 system. BM25 and the reranker are
comparison methods; neither is in the selected prediction path. This demonstration
explains code and a saved result, without rerunning models or indexing.

## 0:00–0:35 — Start with the diagram

SHOW: walkthrough/ARCHITECTURE.png

POINT: Input → Prepare two queries → the SPLADE and Dense branches → Combine rankings →
Return 10 passages. Move to the input box again during the last sentence.

> Hi, I'm Neeraj. My project is about finding the right passages for a question when that question is part of a conversation.
>
> This diagram shows the approach. We keep the original question and create a standalone rewrite using the earlier conversation. Both versions go through SPLADE and BGE, and we combine their rankings to return ten passages.
>
> Let me follow that same path through the code, starting with what the system receives.

## 0:35–1:10 — Establish the selected configuration and input

OPEN: experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json:5

POINT: backends on lines 8–11, then query_mode on line 12.
The file is short; keep the whole configuration visible.

> First, this configuration tells us which version we're looking at: two retrievers, with original-and-rewritten queries.

OPEN: src/convrag/data.py:45

POINT: Collection and the check at lines 50–52, then the whitelist at lines 53–57.
Move from Collection/domain to input as you explain the two retained inputs.

> The input preparation keeps the conversation and its assigned collection. Gold answers and relevance labels are left out of the retrieval input.
>
> That gives us our first assumption: the correct collection is already provided. So a Cloud question searches Cloud passages. We don't have to predict its collection.

## 1:10–2:00 — Explain why and how the question is rewritten

OPEN: src/convrag/queries.py:4

POINT: prompt lines 4–9, then the final-user-turn check on lines 12–14.

> Once we have that input, the challenge is that a follow-up question may not make sense by itself. This prompt asks Qwen to resolve references from the conversation without adding facts.
>
> Our second assumption is that the final turn is the current user question. This check makes that explicit.

OPEN: src/convrag/neural.py:160

POINT: model loading on line 160 and model.generate on line 165.
Do not describe this as running during the recording. It is the implementation that
previously generated the saved rewrites.

> This is the actual model call that generates the rewrite. We save those rewrites beforehand, so each comparison uses the same queries.

OPEN: src/convrag/queries.py:33

POINT: original=latest(task), line 34; rewrite lookup, line 38; fusion branch, line 40.

> But a rewrite can get things wrong. That's why this branch keeps the original question too, and removes exact duplicates. Now we have the queries we'll search with.

## 2:00–2:50 — Explain the representations, then search

OPEN: src/convrag/neural.py:51

POINT IN SPEECH ORDER:
- Lines 57–59: SPLADE's log1p(relu(...)) and vocabulary-wise max pooling.
- Line 68: the BGE CLS vector and normalize(...).
- Line 54: truncation=True and max_length; the default 256 is visible at line 40 if needed.
- Lines 61–64: document-only topk(...,k=128).

> The next step is turning those queries into something we can compare with passages.
>
> SPLADE produces weighted vocabulary terms, including learned expansions. BGE produces a normalized vector of 384 numbers to capture meaning. Using both gives us word-level and meaning-based matching.
>
> To fit local resources, we limit encoder inputs to 256 tokens and retain 128 SPLADE document terms. That can lose information.

OPEN: src/convrag/neural.py:118

POINT: stored matrix at line 117, query encoding at line 119, similarity at line 120,
then strongest-candidate selection and original-ID lookup at lines 121–122.

> The passage representations are already stored. Here, we encode the query, calculate similarity, and return up to a hundred candidates. That leaves us with several ranked lists to combine.

## 2:50–3:40 — Show how lists become the final prediction

OPEN: src/convrag/retrieval.py:11

POINT: w/(k+rank) on line 19, then sorting on line 20. Stay above the BM25 class.

> To combine them, we use reciprocal rank fusion. This formula rewards passages that appear near the top of the lists. Using positions avoids mixing the retrievers' different score scales.

OPEN: src/convrag/cli.py:58

POINT IN SPEECH ORDER:
1. Line 58, query_views: the original and rewritten queries.
2. Lines 60–61, nested loops: each query through each index.
3. Lines 62–63, rrf and pool_size: fusion, followed by the 100-candidate pool.
4. Scroll to lines 74–77, top=ranked[:10] and document_id: the returned prediction.

> You can see those steps come together in this loop: query views, searches, fusion, and a hundred-candidate pool. With two distinct queries and two retrievers, there are four lists.
>
> Finally, these lines return the first ten original passage IDs. The optional reranker branch is skipped in this configuration. So that's the path from conversation to prediction.

## 3:40–4:05 — Connect the code to a real saved output

OPEN: walkthrough/RECORDING_EXAMPLE.json:1

POINT: saved-output label, line 2 → latest_question, line 4 → query_views, lines 6–7 →
passage_excerpt, line 12. Keep the source label visible and wrap the passage text.

> To make that concrete, here's a saved validation example asking, “What are worker nodes?” The first passage explains their role in carrying application deployments and services.
>
> The rewrite only changes capitalization here. This shows the retrieved output, rather than a successful reference resolution. It's an existing result, not a new search.

## 4:05–4:45 — Move from one example to overall results

OPEN: experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json:64

POINT: mean_over_scored_queries on line 64, then ndcg_cut_5 on line 71.
Use this mean, not the separate weighted aggregate below it.

> Of course, one example doesn't tell us how well the system works overall. Across the final judged questions, mean nDCG at five was point four three three four. That measures whether relevant passages appear near the top.
>
> We compared nineteen configurations on validation data. The hybrid improved clearly over plain BM25, although its smaller gain over BM25 with reranking remained uncertain.
>
> Rewrites can still invent context, so improving their reliability would be my next step. Thank you.

## Practice and accuracy notes

- Rehearse once with the actual tab changes. If you exceed five minutes, omit the two
  sentences on truncation/pruning and shorten the closing comparison. Keep both named
  assumptions, the representation explanation and final-output code.
- At the Qwen stop, line 165 shows greedy generation with do_sample=False. The input prompt
  is assembled in queries.py lines 24–27, passed into generation in neural.py line 176,
  and the output is saved in neural.py lines 179–184. These are backup references,
  not additional stops in the timed route.
- The 256-token maximum is the configured sequence length, including tokenizer special
  tokens and the dense query prefix. Do not promise 256 tokens of raw passage content.
  Only SPLADE document vectors are pruned to 128 terms; sparse queries are not.
- Identical original/rewrite strings are deduplicated, so the system produces up to four
  lists. The worker-node example has two distinct strings differing in capitalization.
- RECORDING_EXAMPLE.json is a concise display of a genuine validation run, checked against
  the saved input, trace, first predicted ID and source corpus excerpt (leading whitespace trimmed).
  Its validation score is not the final benchmark mean.
- The final mean is over 332 judged questions out of 507 inputs. Historical leaderboard
  equivalence remains unverified. The GPU study is a disclosed follow-up on a previously
  inspected public final set, not a fresh unseen holdout.
- The assignment does not require live commands. This route explains the code and
  clearly labels saved output. If you display JSON in Terminal, that is not live inference.
- See DEMO_VERIFICATION.md for the file-role map and checks. No retrieval source code
  was changed for this presentation revision.
- Watch the complete recording for readable code, clear sound and duration, then submit
  a shareable link with working access in the separate Canvas walkthrough assignment.
