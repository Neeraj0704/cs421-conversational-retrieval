# Research review: building a competitive conversational retriever

Research date: November 23, 2026  
Project: CS 421, SemEval-2026 Task 8, Subtask A  
Status: literature and implementation review; no retrieval experiments have been run for this project.

## Recommendation

Build a modular, training-free system that preserves the latest question, rewrites it using conversation history, retrieves with a strong sparse model, combines complementary rankings, and reranks a bounded candidate pool. Treat dense retrieval as a measured alternative or additional branch.

For the closest reproduction of the winner, investigate ELSER v1, multiple query formulations, and nested rank fusion. For a practical implementation, compare an accessible learned sparse encoder against BM25 plus a compact dense encoder, using a small cross-encoder for reranking. Do not assume either implementation will reproduce a published score.

The recommendation is a synthesis of exact-task results and broader research, not a new experimental finding. The best published system on this task is our strongest reference; the best configuration for our resources remains an empirical question. [1,4–10]

## What this review covers

I searched both web indexes, ACL Anthology, arXiv, a full-text literature index, author repositories, and official model documentation. I opened 23 research papers or reports and inspected the relevant task definitions, methods, results, or implementation sections. This is a broad, task-focused review, not a claim that every paper or article on RAG has been read.

Priority order:

1. Evidence on the exact SemEval retrieval task.
2. Evidence on other conversational retrieval benchmarks.
3. General retrieval methods and official implementation guidance.

This matters because good answer generation, strong general embedding benchmarks, and strong conversational passage ranking are different achievements. We should avoid choosing a method solely because it is popular in general RAG tutorials.

## 1. The target we must match

Subtask A returns ten original passage IDs per conversational task, searching only its assigned collection. The ranking metric is nDCG@5. Organizer Table 4 reports 0.578 for AILS-NTUA, 0.480 for the strongest baseline, and 0.354 for BM25. These are published test results, not predictions of our performance. [1]

MTRAG is development material. The competition test used 507 MTRAG-UN tasks from the four original domains; the organizer describes 285 answerable and 47 partially answerable tasks eligible for retrieval scoring. The full MTRAG-UN benchmark has additional domains and tasks. Full-benchmark and competition-subset results cannot be exchanged. [1–3]

The ordinary BM25 system we build from the latest question may differ from the published BM25 baseline in query formulation, tokenization, settings, or aggregation. It must be measured and described as our baseline.

## 2. What the leading exact-task systems actually did

Scores below are official nDCG@5 values rounded to three decimals from the organizer overview. Method descriptions come from the teams' papers. [1,4–10]

| Rank | Team | Score | Main retrieval design | Practical lesson |
|---|---|---:|---|---|
| 1 | AILS-NTUA | 0.578 | Several query formulations, ELSER v1, reranking and nested fusion | Query diversity with a suitable retriever deserves priority. |
| 2 | uva-irlab-conv | 0.548 | Contextual rewriting, LION-SP, Qwen3 reranking, listwise LLM reranking | Strong first-stage retrieval and pointwise reranking matter. |
| 3 | Sifei | 0.545 | Original question plus rewrite, SPLADE-v3 and dense retrieval, BGE reranking | Preserve the user's wording while resolving context. |
| 4 | IIMAS-RAG | 0.545 | Rewriting, SPLADE and Voyage retrieval, RRF | Hybrid retrieval is competitive, with API dependencies. |
| 5 | CuriosAI | 0.540 | Rewriting, heterogeneous retrieval, adapted reranker | Strong but more training and components to manage. |
| 6 | GUIR | 0.539 | Complementary query views and passage reranking | A training-free system can be competitive. |
| 8 | Caraman | 0.531 | Fine-tuned rewriter, BM25 plus BGE dense search, BGE reranker | A useful implementation reference for a local project. |

### The winner: the most relevant blueprint

AILS-NTUA reports 0.5776 before rounding. Its five formulations include conservative reference resolution, domain terminology, a hypothetical passage, reasoning-oriented expansion, and anchor keywords. It retrieves top-100 candidates per formulation with ELSER v1, combines retriever and Cohere reranker rankings, then uses nested RRF to limit the influence of less reliable expansions. Its reported rewriting model is DeepSeek-V3.2. [4]

The important idea is controlled diversity. Its ablations found that adding other retrievers could improve deep recall while damaging the top ranks. The authors also identify annotation alignment as a possible advantage for ELSER. This is evidence for this benchmark, not proof that one retriever is universally best. [4]

The paper cites a repository, but I could not inspect it through the available web fetch route. Its code availability is therefore paper-reported, not verified by running or auditing that repository.

### Second place: do not pay for complexity automatically

The Amsterdam system uses GPT-4.1 rewriting, LION-SP-8B retrieval, Qwen3-Reranker-8B pointwise ranking, and a further GPT-4.1 listwise pass. Its reported ablation is especially useful: nDCG@5 rises from 0.4841 to 0.5477 after pointwise reranking, then changes to 0.5475 after listwise ranking. The last stage helps another cutoff but does not improve the main metric in that comparison. [5]

Our implication: add stages only when the selected development metric or a declared secondary objective justifies them. The large models in this particular pipeline are also an implementation burden.

### Competitive alternatives and negative results

Sifei's hybrid approach supports keeping the original last turn alongside a controlled rewrite. Its development table shows the largest improvements come from rewriting, hybrid retrieval, and reranking; the additional concatenation gain is small in that table. It preserves official passage IDs. Some preprocessing uses relevance-protected filtering, which we should not copy into test-time preprocessing. [6]

IIMAS-RAG combines SPLADE with Voyage-3-large using RRF. It reports query rewriting as an important source of improvement. This is a useful alternative, but it introduces a paid embedding dependency and a different reproducibility profile. [7]

CuriosAI uses domain-adaptive LoRA reranking in a larger pipeline. This shows fine-tuning is a credible route, but the score alone does not isolate the value of training versus the rest of the system. [8]

GUIR studies current-turn, rewritten, and history-based query views with MonoT5 reranking. Its paper distinguishes some unofficial post-submission experiments; those should not be presented as official leaderboard results. Its use of supplied rewrites is also different from generating our own rewrites. [9]

Caraman supplies particularly useful negative results: extra query expansion and domain prompting can hurt, and its BGE reranker works better with a modest candidate pool than a much larger one. The author repository includes configuration and split information. It reports 128 GB unified memory for the training setup, so “trained on a Mac” does not establish that training is easy on an arbitrary Mac. [10,25]

## 3. Broader research: what transfers, and what does not

| Research | Main idea | Evidence boundary | Decision for our project |
|---|---|---|---|
| LLM4CS, 2023 | Generate complementary conversational rewrites and hypothetical responses | Evaluated on CAsT benchmarks | Supports multi-view query experiments. [14] |
| CONQRR, 2022 | Train rewriting with retrieval rewards | QReCC and specific retrievers | Useful background; defer RL training. [15] |
| ConvSearch-R1, 2025 | Retrieval-guided warm-up and reinforcement learning | QReCC and TopiOCQA | Promising trained rewrite models; no direct SemEval-score implication. [13] |
| DVCQR, 2026 | Separate sparse-oriented and dense-oriented rewrites | Other conversational retrieval datasets | Try distinct query styles before attempting its training procedure. [12] |
| CompCQR, October 2026 | Compose query components, fuse rankings, select a leading passage | QReCC, TopiOCQA, CAsT-19/20 | Interesting extension, not a demonstrated replacement for the SemEval winner. [11] |
| Better Together, October 2026 | Study complementary rewriting on a strong baseline | HotpotQA, AmbigNQ, enterprise data | Tests should include a strong baseline and cost-matched alternatives. [19] |
| HyDE, 2023 | Search with an embedding of a generated hypothetical passage | General zero-shot retrieval | Optional low-weight branch; generated text is not factual evidence. [17] |
| RankGPT, 2023 | Ask an LLM to order candidate passages | General reranking benchmarks | Optional comparison, not the default final stage. [18] |
| SPLADE-v3, 2024 | Learned sparse representations and improved training | MS MARCO and out-of-domain retrieval | Serious sparse candidate if access and indexing are practical. [16] |
| Qwen3 Embedding, 2025 | Models for embedding and pairwise relevance ranking | Broad retrieval benchmarks | Benchmark smaller models locally; general scores do not establish MTRAGEval superiority. [22] |
| Rewrite-Retrieve-Read, 2023 | Adapt the query before retrieval and answer generation | General QA | Supports query adaptation, but its downstream objective differs from ours. [23] |

### Why the newer papers matter

CompCQR generates query components with two LLM calls, constructs many combinations, and adds a passage-selection call. Its full construction has 325 ordered variants; for bag-of-words BM25, order does not create distinct queries, reducing the component combinations to 31. Fewer LLM calls therefore do not mean a small retrieval workload. [11]

We can investigate the underlying idea with a much smaller number of carefully chosen views. That would be an inspired adaptation, not a reproduction of CompCQR.

DVCQR highlights that keyword-based and dense retrievers can prefer different query formulations. Its training procedure is substantial. A prompt-based experiment with an entity-focused query and a natural standalone query tests part of that intuition without claiming to reproduce its results. [12]

Better Together reports that combining query methods can improve coverage, but also documents harm on AmbigNQ. Some headline ensemble analyses are post-hoc and the routing evaluation is simulated. Its results should motivate controlled testing rather than a blanket claim that more queries always help. [19]

A newer corpus-feedback rewriting preprint, arXiv:2610.10955, appeared in searches. I could not retrieve its primary full text, so I excluded its reported gains from the decision evidence. Secondary summaries are not enough to establish a superior method.

## 4. Recommended architecture, explained simply

Our proposed implementation has five steps.

1. Read the current question and available past turns.
2. Form a small set of search queries that express the same information need.
3. Search the correct document collection with each query.
4. Combine the passage rankings and examine a bounded set more carefully.
5. Return ten unique, original passage IDs with scores consistent with the final order.

### Query views

Start with three views:

- **Original:** the latest user message, unchanged.
- **Conservative rewrite:** replace unclear references with the entities and constraints established in the conversation.
- **Keyword view:** preserve names, acronyms, dates, product identifiers, and necessary conditions.

Do not add facts, invent an answer, copy irrelevant history, or silently turn a new topic into a continuation. If a question is already explicit, the conservative rewrite can remain identical. Keep user and assistant roles visible to the rewriter.

Initially, generate the views in one structured response if the selected model handles this reliably. This is an engineering proposal intended to reduce overhead; it is not the exact prompting design of any cited winner.

History length is a development setting. Compare full available history with a bounded window rather than assuming one is always best. Assistant turns may be necessary to resolve references, but should not be treated as verified factual evidence.

### Retrieval choices

Two routes deserve a short feasibility comparison:

**Route A: closer to the winner.** Use ELSER v1 with the small query set, then add rank fusion and reranking. Before committing, confirm access, model version, indexing throughput, and usable inference resources.

**Route B: practical independent implementation.** Use an accessible learned sparse model, or BM25 plus a compact dense encoder, with the same query views and a modest reranker. Start with an established BGE dense model for a direct implementation reference; evaluate another model only if the first comparison leaves a material uncertainty.

ELSER is learned sparse semantic search; it is not ordinary BM25. Sparse models can expand lexical representations using learned associations. An external vector database is not required by the task; choose an index implementation based on the selected encoder and measured corpus size.

### Fusion

RRF combines rank positions instead of mixing scores whose scales may differ:

`fusion_score(passage) = sum(weight / (constant + rank))`

Start with equal weights and a conventional constant such as 60. If one view is noisy, test downweighting it on development data. Nested fusion is a later option, not a reason to build a large parameter sweep immediately. RRF is established prior work and must be credited. [20]

### Reranking

Begin with BGE-reranker-v2-m3 because multiple exact-task systems used it. A Qwen3-Reranker-0.6B comparison is a reasonable alternative if runtime permits. Both have official model documentation. Neither model is guaranteed to win on our split. [6,10,22,28,29]

Proposed candidate-pool comparison: 30, 50, and 100 passages. These values are our experiment settings, not established universal optima. Score each passage against the conservative standalone query, then compare reranking alone with a simple combination of reranker and retrieval ranks.

For a history-aware LLM reranker, provide the dialogue and preserve passage IDs, but only add this if the smaller reranker leaves a demonstrated failure mode. It must be evaluated against its extra latency and cost.

## 5. What to build first, and what counts as our contribution

The value of a course project can come from a careful independent comparison and error analysis. Using known methods is appropriate if we cite them and describe our implementation accurately.

A suitable project question is:

**When does preserving the original question alongside a history-based rewrite improve conversational passage retrieval, and is that gain worth the extra search and reranking cost?**

Suggested experiment sequence:

| Run | Query input | Retrieval | Reranking | Question answered |
|---|---|---|---|---|
| A0 | Latest question | BM25 | None | What is our simplest baseline? |
| A1 | Latest plus earlier user questions | Same BM25 | None | Does raw history help? |
| A2 | Conservative rewrite | Same BM25 | None | Does resolving references help? |
| A3 | Original plus rewrite as separate searches | Same BM25, RRF | None | Does preserving wording protect against rewrite mistakes? |
| B0 | Same selected query views | Learned sparse | None | Does a stronger retriever help independently? |
| B1 | Same views | BM25 plus dense, RRF | None | Is hybrid retrieval competitive under our resources? |
| C0 | Best development candidate set | Same retriever | Small cross-encoder | Does better ordering improve nDCG@5? |
| C1 | Same as C0 plus keyword view | Same retriever | Same reranker | Does another complementary view justify its cost? |
| Optional | Same queries and candidates | Same | Larger or LLM reranker | Is extra model capacity useful? |

This sequence separates changes to query understanding, retrieval, and ranking. It prevents crediting rewriting for a gain that actually came from swapping the retriever.

A smaller feasible result consists of A0–A3 and C0. If learned sparse indexing is available quickly, B0 should take priority over a broad search across dense models. Do not spend the entire remaining project time on model training.

Record nDCG@5, Recall@5/10, candidate Recall@50/100 where available, indexing time, query latency, LLM calls, and tokens. Cost comparisons should distinguish one-time indexing from per-question inference.

## 6. Evaluation must be established before a winner comparison

The public evaluator inspected in this review:

- Uses pytrec_eval for the underlying retrieval metrics.
- Contains fixed paths under `human/retrieval_tasks_convid/.../dev.tsv`.
- Computes per-collection means over returned scored queries.
- Weights collection means by the number of submitted predictions for that collection.
- Reads numerical context scores, so final scores must reflect our intended ranking. [27]

**Inference from the code:** when scored-query counts and submitted-query counts differ by domain, this aggregation can differ from a direct mean across all scored queries. Merely using nDCG@5 is not enough to establish identical scoring.

Before the final comparison:

1. Pin the repository revision and record input-file checksums.
2. Confirm the actual competition task IDs and four collection mappings.
3. Route evaluation to the correct MTRAG-UN relevance files.
4. Verify all submitted passage IDs exist in their assigned collection.
5. Make sure every required task receives ten unique results.
6. Count tasks with relevance judgments; do not infer the count only from prose.
7. Report both the organizer-style aggregate and the direct mean over scored queries, labeling each.
8. Re-evaluate archived official predictions on the same inputs if accessible. Agreement with a known official score is stronger evidence than matching a paper's description.
9. If historical scoring cannot be established, label the leaderboard comparison approximate.

I found a concrete reporting inconsistency: the Amsterdam paper states 332 retrieval tasks, but its displayed domain counts, 84 + 59 + 106 + 87, sum to 336. This may be a table error; it is not a reason to replace the organizer's specification. Actual task IDs and relevance files must settle it. [5]

Do not compare a MTRAG development score to 0.578 as if both were on the competition test set. Do not compare a full six-domain MTRAG-UN result with the four-domain leaderboard.

## 7. Leakage, attribution, and credible analysis

These are part of a valid experiment:

- Split MTRAG by conversation. Later and earlier turns from the same conversation must stay together.
- Keep the final competition test labels out of prompting, tuning, and preprocessing.
- The retriever receives the conversation prefix only, never future turns or reference answers.
- Treat provided rewrites as a separately labeled reference baseline, not our generated output.
- Preserve original passages and IDs. Make any text normalization deterministic and independent of relevance labels.
- If using a public task-fine-tuned rewriter, audit which MTRAG conversations trained it before reporting a held-out development result. A checkpoint trained on all MTRAG can contaminate our internal holdout.
- Credit reused code, prompts, models, and methods. Our experimental results must come from logged runs.
- Public test data and modern pretrained models mean contamination cannot always be ruled out; document provenance rather than claiming a fully clean historical reproduction.

For error analysis, inspect examples from every domain and both standalone and context-dependent questions. Useful labels are wrong reference resolution, topic carryover, missing technical terms, candidate retrieval failure, and ranking failure.

Use paired per-query differences for internal comparisons. A bootstrap that resamples conversations is preferable when several evaluation queries share a conversation. For published systems, significance testing requires their paired outputs or scores; an aggregate leaderboard score alone is insufficient.

## 8. Access and resource findings

| Component | What I verified | Consequence |
|---|---|---|
| ELSER | Elastic documentation requires an appropriate subscription or active trial; v1 remains preview and v2 is not index-compatible | Check feasibility and version before choosing a reproduction route. [26] |
| SPLADE-v3 | Official card lists a noncommercial share-alike license and access conditions | Suitable academic use still requires reviewing and accepting the model terms; do not describe it as unrestricted. [30] |
| BGE-reranker-v2-m3 | Official card lists Apache 2.0 and inference examples | A concrete reranking candidate. [28] |
| Qwen3 models | Official technical report and cards include smaller sizes and public weights | Worth a bounded efficiency comparison. [22,29] |
| Caraman code | README, configurations, and reproduction instructions were readable | Useful reference; not executed or verified end to end. [25] |
| CompCQR code | Public repository and license were readable | Possible extension; different datasets and retrieval workload. [24] |
| Winner code | Repository is cited by its paper, but fetching failed here | Do not rely on an uninspected implementation. [4] |

No paid service was activated, no model was trained, and no corpus was downloaded during this research phase. Exact memory requirements, indexing time, API cost, and our achievable score remain unmeasured.

## 9. Reading order and next action

Read these first:

1. The organizer overview: task, Table 4, and evaluation rules. [1]
2. The winner's retrieval section and ablations. [4]
3. Sifei's query formulation, fusion, and reranking ablations. [6]
4. Caraman's implementation and negative results. [10,25]
5. The public evaluator. [27]

Then read CompCQR and DVCQR for potential extensions, not as established replacements for the exact-task winner. [11,12]

The next work should be a short feasibility and data audit: confirm final course requirements, inspect hardware and permitted model access, pin the dataset, establish the correct evaluation route, and benchmark one sparse and one practical hybrid option. After that, choose the smallest system that supports the experiment sequence.

The literature supports a strong direction. It does not justify promising to beat 0.578 before our own measurements.

## Sources

The research papers below were opened; review depth focused on sections relevant to our retrieval decision. Repositories and documentation were read where accessible, not executed.

1. [Rosenthal et al., SemEval-2026 Task 8 overview](https://aclanthology.org/2026.semeval-1.447/). Authoritative task rules and official results.
2. [Katsis et al., mtRAG, TACL 2025](https://aclanthology.org/2025.tacl-1.36/). Original benchmark.
3. [Rosenthal et al., MTRAG-UN, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.503/). Expanded benchmark and evaluation challenges.
4. [AILS-NTUA system paper](https://aclanthology.org/2026.semeval-1.175/). First-place retrieval architecture.
5. [uva-irlab-conv system paper](https://aclanthology.org/2026.semeval-1.373/). Second-place pipeline and stage ablations.
6. [Sifei system paper](https://aclanthology.org/2026.semeval-1.32/). Third-place hybrid retrieval and controlled rewriting.
7. [IIMAS-RAG system paper](https://aclanthology.org/2026.semeval-1.345/). Sparse-dense fusion and query rewriting.
8. [CuriosAI system paper](https://aclanthology.org/2026.semeval-1.151/). Hybrid retrieval and adapted reranking.
9. [GUIR system paper](https://aclanthology.org/2026.semeval-1.325/). Training-free query views and MonoT5 ranking.
10. [Caraman system paper](https://aclanthology.org/2026.semeval-1.225/). Local implementation and negative results.
11. [CompCQR, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1225/). Compositional query generation.
12. [DVCQR, ACL 2026](https://aclanthology.org/2026.acl-long.1054/). Distinct sparse and dense query views.
13. [ConvSearch-R1, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1349/). Reinforcement learning from retrieval.
14. [LLM4CS, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.86/). Prompting for conversational intent.
15. [CONQRR, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.679/). Retrieval-aligned rewriting.
16. [SPLADE-v3 technical report, 2024](https://arxiv.org/abs/2403.06789). Learned sparse retrieval.
17. [HyDE, ACL 2023](https://aclanthology.org/2023.acl-long.99/). Hypothetical document retrieval.
18. [RankGPT, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.923/). LLM reranking.
19. [Better Together, EMNLP Industry 2026](https://aclanthology.org/2026.emnlp-industry.143/). Complementary rewriting under a strong pipeline.
20. [Cormack et al., RRF, SIGIR 2009](https://lintool.github.io/robust04-analysis-papers/p758-cormack.pdf). Rank fusion.
21. [Mo et al., A Survey of Conversational Search, TOIS 2025](https://playbigdata.ruc.edu.cn/dou/publication/2025_tois_conversational_survey.pdf). Broader field taxonomy and related work.
22. [Qwen3 Embedding technical report](https://arxiv.org/abs/2506.05176). Embedding and reranking model families.
23. [Ma et al., Rewrite-Retrieve-Read, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.322/). Query adaptation for RAG.
24. [CompCQR author repository](https://github.com/YunahJang/CompCQR).
25. [Caraman author repository](https://github.com/davidcaraman/semeval2026-mtrag-retrieval).
26. [Elastic ELSER documentation](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser).
27. [IBM retrieval evaluation source](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/run_retrieval_eval.py) and [evaluation instructions](https://github.com/IBM/mt-rag-benchmark/blob/main/scripts/evaluation/README.md).
28. [BGE-reranker-v2-m3 official model card](https://huggingface.co/BAAI/bge-reranker-v2-m3).
29. [Qwen3-Reranker-0.6B official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
30. [SPLADE-v3 official model card](https://huggingface.co/naver/splade-v3).

