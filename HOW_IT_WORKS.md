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

We first tested BM25, conversation history, rewriting, fusion and reranking. The strongest
CPU method searched the latest question and reranked its candidates. We then completed full
SPLADE and dense indexes using your Mac's GPU and tested twelve more configurations.

Our selected system searches in four ways: sparse search with the original question, sparse
search with the rewrite, dense search with the original, and dense search with the rewrite.
It combines those rankings and returns the top ten. Duplicate query views are removed.
This hybrid worked best on validation without the cross-encoder reranker.

| Method | Final nDCG@5 | Final Recall@10 |
|---|---:|---:|
| Plain BM25 | 0.2896 | 0.4261 |
| BM25 + reranker | 0.4161 | 0.5347 |
| Selected sparse + dense fusion | **0.4334** | **0.6211** |

All three rows score the same 332 judged questions from 507 inputs. nDCG is a ranking-quality
score, not a percentage of correct answers. The separate domain-weighted nDCG@5 for the
hybrid is 0.4379. Do not mix that with the direct means above.

The hybrid finds more relevant candidates. Its small ranking improvement over the stronger
BM25 reranker is uncertain: the paired 95% interval includes zero. Its improvement over plain
BM25 is clearer. Because the public final set was already used for the CPU stage, the GPU
comparison is labelled a follow-up benchmark. We have not established an official competition rank.

## How the research shaped the implementation

We tested several research ideas: resolve conversational references, keep the original question,
combine sparse and semantic search, and rerank candidates. The results show why each addition
needs testing: raw history can dilute a question, rewrites can lose terms, and reranking can
hurt an already useful fusion. The selected configuration follows validation evidence.

Full GPU indexing and final comparisons are complete. ColBERT, retriever training, stronger
rewriters and adaptive search remain future experiments. See [RESULTS.md](RESULTS.md) and the
[combined report](report/FINAL_PROJECT_REPORT.md) for all comparisons and limitations.

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
