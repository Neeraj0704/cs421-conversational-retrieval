# The project in simple words

We are building a passage finder for a conversation. It receives the current question,
the earlier conversation, and the collection to search. It must return ten existing
passages in the best order. The benchmark checks those passages against human relevance
judgments. Producing an answer is a separate SemEval subtask.

## What the program does

1. **Loads the official data.** We keep the original passages and their IDs. Each question
   searches its assigned collection: CLAPNQ, Cloud, FiQA or Government.
2. **Builds a fast search index.** BM25 finds passages containing useful question terms.
   It gives us up to 100 candidates rather than immediately keeping only ten.
3. **Tries different ways of using the conversation.** One version searches the latest
   question. Another adds earlier user questions. A small language model can also rewrite
   a follow-up as a standalone question. Fusion searches the original question and rewrite
   separately, then combines their rankings.
4. **Optionally reads the candidates more carefully.** A cross-encoder receives the question
   together with each candidate passage and scores how relevant the pair is. It reorders
   those candidates before we return the best ten.
5. **Measures the result.** nDCG@5 rewards putting relevant passages near the top. Recall
   checks how many relevant passages we found. Candidate recall tells us whether useful
   passages reached the reranker at all.

Gold answers, correct passage lists and benchmark labels are excluded from search inputs.
Only the already supplied conversation prefix is available to the model.

## What our experiments found

On the held-out development conversations:

| Method | nDCG@5 |
|---|---:|
| Latest question + BM25 | 0.2249 |
| Earlier user questions + BM25 | 0.1842 |
| Generated rewrite + BM25 | 0.2114 |
| Original + rewrite fusion | 0.2472 |
| Fusion + reranker | 0.3464 |
| Fusion + retrieval/reranker rank blend | 0.3138 |
| Latest question + BM25 + reranker | **0.3898** |

We selected the last method using development results. The tested small rewriter sometimes
leaves references unresolved or changes the question. The reranker provides the strongest
measured improvement in this setup. These findings apply to the tested models and settings;
stronger conversation models remain a useful next experiment.

The final test uses three choices fixed in advance: plain BM25, fusion, and the selected
reranked method. See [RESULTS.md](RESULTS.md) for completed final scores and the report for
uncertainty and aggregation details.

## How the research shaped the implementation

The research gave us ideas to test: resolve conversational references, preserve original
wording, combine retrieval signals, and improve the ordering of candidates. We measured
each addition. We retained the strongest validated configuration and kept negative results.

We also implemented SPLADE learned sparse retrieval and BGE dense retrieval. Small index
tests passed. Full indexing was estimated at about 10.8 and 3.2 CPU hours in this session,
so their complete benchmark experiments are deferred. Their code is available for a later
accelerator run. ColBERT, retriever training and agent loops remain future extensions.

## Where to look in the code

| File | Purpose |
|---|---|
| `src/convrag/data.py` | Load data, preserve IDs, exclude gold fields, split conversations |
| `src/convrag/queries.py` | Select question text and define the rewrite prompt |
| `src/convrag/retrieval.py` | BM25 search and reciprocal rank fusion |
| `src/convrag/neural.py` | Local rewriting, reranking and optional neural indexes |
| `src/convrag/evaluation.py` | Validate outputs and calculate retrieval scores |
| `configs/` | Exact settings for each experiment |
| `scripts/run_core.py` | Run the core study and fixed final comparisons |
| `scripts/reproduce.py` | Repeat results without overwriting recorded experiments |

## Try a question

After setup and indexing:

```sh
convrag search --domain cloud --question "How can I deploy a Python application?" --rerank
```

The output contains ten passage IDs and their text. For a conversation example, inspect
the saved demo or run `python scripts/demo.py --run R00_bm25_latest_rerank` after experiments.
