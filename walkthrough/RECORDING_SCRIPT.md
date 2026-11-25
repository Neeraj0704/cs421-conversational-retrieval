# Full recording script: architecture and exact code pointers

Target: about 4 minutes 45 seconds; maximum five minutes.
Read only the blockquoted speech. Directions and code locations are not narration.
Timings are rehearsal targets. Use your own voice and check one timed practice.

## Before recording

Open /Users/neerajvp/Documents/CS421 NLP/NLP/implementation in VS Code.
Set the font to 18–20 points and enable line numbers. Option+Z enables word wrap;
Command+B hides the sidebar. Open the diagram in an image tab and fit it to the window.
Use Command+P, paste each path:line destination below, and press Return.
Practice the jumps first. Move the pointer slowly to each named expression as you speak.
Briefly select a few words if useful; do not edit the code.
Record the editor window with microphone audio. A camera is not required.

## 0:00–0:40 — Architecture

Show: walkthrough/ARCHITECTURE.png

Point in order: Input → Prepare two queries → SPLADE search → Dense search →
Combine rankings → Return 10 passages. At “both versions,” point to the search branches.

> Hi, I'm Neeraj. This project finds relevant passages for questions in a conversation.
>
> Here's the overall flow. We take the latest question and earlier conversation. We keep the original question and use Qwen to make a standalone rewrite. Both versions go through two searches: SPLADE uses learned word weights, while BGE uses meaning vectors.
>
> We combine their rankings and return ten passage IDs. We aren't generating an answer. Let me show how this connects to the code.

## 0:40–1:15 — Configuration, input and first assumption

Open: experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json:5

Point to: lines 8–12, backends containing splade and dense, then query_mode: fusion.
There is no reranker entry in this selected configuration.

> This configuration selects the two retrievers and original-plus-rewrite fusion. There's no reranker in this selected version.

Then open: src/convrag/data.py:45

Point to: collection lookup/check, lines 50–52; then the field whitelist at lines 53–57,
especially Collection, domain and input.

> Here, I keep the conversation and assigned collection, excluding gold answers and relevance labels from retrieval inputs.
>
> My first assumption is that the correct collection is provided. A Cloud question searches Cloud passages; the system doesn't choose the collection itself.

## 1:15–1:55 — Rewriting and second assumption

Open: src/convrag/queries.py:4

Point to: prompt lines 4–9, especially “Do not answer ... or introduce facts.”
Then point to the final-speaker check at lines 12–14 when naming the assumption.

> Next, this prompt asks Qwen to resolve references and make the question understandable on its own, without adding facts.
>
> My second assumption is that the conversation ends with the current user question. This check enforces that.

Then open: src/convrag/queries.py:33

Point to: original=latest(task), line 34; stored rewrite, line 38; fusion branch, line 40.
Trace [original,rewrite]. dict.fromkeys removes identical query strings.

> Here, the fusion branch keeps both the original and the rewrite, removing exact duplicates. Keeping the original gives us another search path if the rewrite misunderstands something. The model-generated rewrites are saved before retrieval, so comparisons use the same wording.

## 1:55–2:50 — Representations and search

Open: src/convrag/neural.py:51

Point to these expressions as you describe them:

- SPLADE: lines 57–59, log1p(relu(...)) and values.max(dim=1).
- Dense: line 68, outputs.last_hidden_state[:,0] and normalize(...).
- Truncation: line 54, truncation=True and max_length=self.max_length; the default 256 is at line 40.
- Document pruning: lines 61–64, if not query and topk(...,k=128). This pruning applies to documents.

> These queries become numerical representations here. SPLADE creates weights for vocabulary terms, including learned expansions. BGE creates a normalized vector of 384 numbers that represents meaning. They provide complementary matching signals.
>
> There's also a resource assumption: reading the first 256 tokens and retaining 128 SPLADE document terms preserves enough useful information. This keeps local indexing practical, but can lose details.

Then open: src/convrag/neural.py:118

Point to: query encoding, line 119; matrix multiplication, line 120; candidate selection and
return, lines 121–122. The stored passage matrix is loaded at line 117.

> This search function compares the query with stored passage representations. The matrix multiplication produces similarity scores, and the strongest candidates are returned. Each search retrieves up to a hundred passages from the assigned collection.

## 2:50–3:40 — Ranking fusion and final output

Open: src/convrag/retrieval.py:11

Point to: k=60 on line 11; w/(k+rank) on line 19; descending sort on line 20.
Do not scroll into the BM25 class, which belongs to comparison experiments.

> We combine the lists using reciprocal rank fusion. This expression gives more credit to passages near the top of each list. It uses ranking positions, so we don't directly mix incompatible model scores.

Then open: src/convrag/cli.py:58

Point in sequence:

1. Line 58, views=query_views(...): original and rewritten queries.
2. Lines 60–61, nested loops: each retriever searches each query view.
3. Lines 62–63, rrf(...) and pool_size: fusion and the 100-candidate pool.
4. Scroll to lines 74–77, top=ranked[:10] and document_id: final ten original IDs.

> Here's the full loop: make the query views, search each index with each view, combine up to four lists, and keep a hundred candidates. Finally, these lines take the first ten and write their original passage IDs into the prediction.
>
> The reranker branch is skipped for our selected configuration.

## 3:40–4:05 — Saved example

Open: walkthrough/RECORDING_EXAMPLE.json:1

Point to: saved-output label, line 2; question, line 4; query views, lines 6–7;
passage excerpt, line 12. Enable word wrap so its first sentence is readable.

> Here's a saved validation example: “What are worker nodes?” The first passage explains that worker nodes carry an application's deployments and services.
>
> The rewrite only changes capitalization here, so this isn't an example of resolving a reference. I'm showing an existing result, not running a new search.

## 4:05–4:45 — Results and closing

Open: experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json:64

Point to: mean_over_scored_queries, line 64; then ndcg_cut_5, line 71.
Use this mean, not the separate weighted aggregate on line 81.

> On the final judged questions, the mean nDCG at five was point four three three four. It measures how well relevant passages rank near the top, rather than answer accuracy.
>
> We compared nineteen configurations on validation data. The hybrid clearly improved over plain BM25, while its smaller gain over BM25 with reranking remained uncertain.
>
> The main limitation is that rewrites can invent context. Improving their reliability would be my next step. Thank you.

## Rehearsal and submission notes

- Finish around 4:45 to leave room before the five-minute maximum. If a practice runs long,
  shorten the example and closing comparison. Keep the architecture, representations,
  two named assumptions and final-output code.
- The prompt lives in queries.py; actual Qwen generation lives in neural.py. Backup locations:
  model loading, lines 154–160; greedy generation, lines 161–166; prompt-builder call, line 176;
  saved rewrite output, lines 179–184. These are not extra stops in the timed route.
- No live command is required by the walkthrough brief. This route shows a saved example.
  To display it in Terminal instead, run from implementation:
  ../.venv/bin/python -m json.tool walkthrough/RECORDING_EXAMPLE.json
  That command only displays the saved file; it is not live inference.
- The example is from validation; 0.4334 is the separate final benchmark mean over 332 judged
  questions out of 507 inputs. Historical leaderboard equivalence is unverified. The GPU
  extension used a previously inspected public final set; do not call it a fresh unseen holdout.
- Watch the finished recording at normal size to check readability, audio and duration.
  Submit a shareable link with working access in the separate Canvas walkthrough assignment.
