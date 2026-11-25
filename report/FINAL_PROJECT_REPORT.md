# Improving conversational passage retrieval

Neeraj Vijayakumar Pattanashetti

CS 421 Natural Language Processing

November 25 2026

## 1 Task metric and data

I built a passage retrieval system for SemEval 2026 Task 8, MTRAGEval Subtask A [1]. Given a conversation ending in a user question, it returns ten ranked original passages from the assigned collection. A follow-up such as “What about its cost?” requires resolving what “it” refers to. The project tests whether conversation history, query rewriting, complementary retrievers and reranking improve that search. It uses pretrained models and does not generate answers or train new model weights. Compared with the proposal's BM25 and dense hybrid, I added SPLADE as a stronger vocabulary-based comparison. Small timing tests initially favoured a CPU fallback; successful MPS checks then made full neural indexing practical.

The official ranking metric is normalized discounted cumulative gain at five, nDCG@5. These relevance files are binary: a listed relevant passage has relevance 1; an unjudged result contributes 0. For ranks i from 1 to 5, DCG@5 sums relevance divided by log base 2 of (i + 1). IDCG@5 uses the same calculation with all relevant passages first, up to five. Divide DCG by IDCG, then average judged questions; inputs without judgments are excluded. Higher is better and 1 is ideal. With one relevant passage, rank 1 scores 1 and rank 5 about 0.387. Recall@10 counts relevant passages in the returned ten divided by the total relevant passages for that question, then averages questions. Candidate Recall@100 applies the same calculation before selecting the final ten.

I used IBM's official passage-level archives at commit e0970de1b1207c2a9506adf238e6e1a22037e38a [2]: CLAPNQ 183408 passages, Cloud 72442, FiQA 61022 and Government 49607, totalling 366479. Each task searches its assigned collection. Counts were checked against the archives. Passage boundaries, collection assignments and offset-bearing IDs were preserved, with no added documents or custom rechunking.

MTRAG provides 842 tasks from 110 conversations. A seed-42 split within each domain held out 22 whole conversations: 174 validation tasks, including 161 judged tasks. The other 88 conversations were not used to train weights. All 19 validation configurations use the same judged questions. This local split differs from evaluation on the full development set.

The final task file contains 507 inputs, matching the published four-domain task scope [1]. Relevance files score 332; the other 175 receive predictions but are excluded from retrieval quality averages. Inputs exclude gold answers, relevance lists, supplied rewrites and future turns. Labels are read for scoring after prediction. The GPU study followed an earlier CPU evaluation on this public final set, so its final results are disclosed follow-up evaluations, not a fresh unseen holdout.

<!-- pagebreak -->

## 2 Approach and implementation

BM25 provides a fast baseline for exact wording. I used bm25s with English stopwords, Snowball stemming, Lucene-style k1 of 1.2 and b of 0.75. The latest-only configuration searches the current question; the history baseline concatenates earlier user questions and the current one, excluding assistant responses.

### Rewriting and preserving the original

Qwen/Qwen2.5-0.5B-Instruct [5] receives the full supplied conversation prefix with speaker labels. Its prompt requests one standalone search question, resolving references while preserving names, dates, negation and numerical constraints. It must not answer or add facts; assistant messages supply context rather than verified evidence. Decoding is greedy with at most 128 new tokens; empty output falls back to the original. Cached rewrites preserve identical query text across comparisons.

A rewrite can omit useful terms or invent context. I therefore tested separate searches for the original and rewritten questions, combining their rankings. Exactly identical query views are removed. One experiment adds a keyword view by removing selected common words from the rewrite.

### Learned sparse and dense retrieval

SPLADE uses naver/splade-cocondenser-ensembledistil [3]. It learns weighted vocabulary representations, providing expansion beyond literal word overlap. Input is truncated to 256 tokens. I apply log one plus ReLU to token logits, mask padding and take the maximum across token positions. Each document retains its 128 largest term weights; query vectors are not pruned this way.

BAAI/bge-small-en-v1.5 [4] supplies a complementary semantic signal. It produces normalized 384-dimensional CLS vectors from up to 256 tokens. Queries receive the prefix “Represent this sentence for searching relevant passages: ”. Search uses exact inner products within the assigned collection. Truncation and sparse pruning change the index representation, not the source passages or IDs.

### Combining candidates and reranking

Each retriever and query view returns up to 100 candidates. Reciprocal rank fusion, RRF, adds 1 divided by the sum of 60 and a passage's rank in each list. Lists have equal weight; the fused pool is limited to 100. This combines positions without assuming that model scores share a scale.

The optional cross-encoder/ms-marco-MiniLM-L6-v2 [6] reads query-passage pairs, with a 384-token limit and batch size 32. Fusion configurations rerank using the rewrite; latest-only configurations use the current question. Pure reranking follows the model's order. Rank blending instead applies RRF to retrieval and reranker lists with weights of 0.5 each. The first ten IDs are returned with strictly decreasing scores. The selected E08 combines SPLADE and dense original/rewrite searches without a reranker.

Implementation uses Python 3.11, bm25s, PyStemmer, PyTorch, Transformers, Sentence Transformers, NumPy, SciPy and pytrec-eval-terrier. Dependencies are locked in uv.lock; model revisions, prompts, queries, settings and hashes are saved [7]. No paid inference API or external training data was used.

<!-- pagebreak -->

## 3 Validation experiments

I completed seven CPU-stage comparisons and twelve additional configurations using full neural indexes. Table 1 gives mean nDCG@5 over the same 161 judged validation questions. “Fusion” means original and rewritten query searches unless specified otherwise; “blend” combines retrieval and reranker ranks.

| Run | Configuration | nDCG at 5 |
|---|---|---:|
| E00 | BM25 latest | 0.2249 |
| E01 | BM25 raw user history | 0.1842 |
| E02 | BM25 rewrite only | 0.2114 |
| B02 | BM25 query fusion | 0.2472 |
| B03 | B02 with pure reranking | 0.3464 |
| B04 | B02 with rank blend | 0.3138 |
| R00 | BM25 latest with reranking | 0.3898 |
| D00 | Dense latest | 0.3850 |
| D01 | Dense query fusion | 0.3966 |
| G02 | Dense latest with reranking | 0.4366 |
| E03 | SPLADE latest | 0.3990 |
| E04 | SPLADE rewrite only | 0.3611 |
| E05 | SPLADE query fusion | 0.4107 |
| E06 | E05 with pure reranking | 0.3795 |
| E07 | E05 with rank blend | 0.4121 |
| E10 | SPLADE original rewrite and keyword fusion | 0.3909 |
| G01 | SPLADE latest with reranking | 0.4281 |
| G03 | Sparse and dense latest with reranking | 0.4291 |
| E08 | Sparse and dense query fusion | 0.4479 |

Raw history and rewrite-only search underperformed the latest-question baselines. Preserving both query views helped BM25 and SPLADE. Reranking helped BM25, but reduced SPLADE fusion from 0.4107 to 0.3795 on identical candidates; rank blending recovered the loss. The keyword view did not help.

E08 was selected by validation before its final run, alongside G01 and G02 as family comparators. Against R00, it improved 61 questions, tied on 62 and worsened 38. Its candidate Recall@100 of 0.7917 was almost identical to E05's 0.7925, while nDCG@5 improved. The nDCG gain was not accompanied by higher average candidate recall; this suggests that candidate composition and ordering also mattered. E08 uses more search calls than single-query methods, though the final pool remains 100. G03 and E08 differ in both queries and reranking, so they do not isolate a reranking effect on E08.

<!-- pagebreak -->

## 4 Final results and published comparisons

All six frozen configurations were scored on the same 332 judged tasks, with predictions for 507 inputs. Table 2 separates earlier CPU runs from the final three GPU follow-up runs. Mean scores weight judged questions equally. The upstream-style aggregate rounds domain scores to five decimals and weights them by submitted domain counts.

| System | Mean nDCG at 5 | Weighted nDCG at 5 | Recall at 10 | Candidate recall at 100 |
|---|---:|---:|---:|---:|
| BM25 latest | 0.2896 | 0.2918 | 0.4261 | 0.6989 |
| BM25 query fusion | 0.3306 | 0.3332 | 0.4872 | 0.8187 |
| BM25 latest reranked | 0.4161 | 0.4188 | 0.5347 | 0.6989 |
| SPLADE latest reranked | 0.4199 | 0.4239 | 0.5470 | 0.7570 |
| Dense latest reranked | 0.4040 | 0.4082 | 0.5309 | 0.6810 |
| Selected hybrid fusion | 0.4334 | 0.4379 | 0.6211 | 0.8610 |

The hybrid gains 0.1438 nDCG over plain BM25, with a 95 percent paired-bootstrap interval of 0.1017 to 0.1879. Against BM25 with reranking, the gain is 0.0173 with interval -0.0178 to 0.0525. That smaller gain is not statistically resolved. Bootstrap uses 3000 whole-conversation resamples, seed 42, without correction for multiple comparisons.

### Published results and this project

Table 3 places my hybrid beside selected final-task scores from the overview [1, Table 4]. Published rows are contextual references; the local rows use my two aggregates. My BM25 implementation is a separate baseline.

| System and score source | nDCG at 5 | Recall at 5 |
|---|---:|---:|
| BM25 baseline | 0.354 | 0.388 |
| BGE base 1.5 baseline | 0.390 | 0.425 |
| ELSER strongest baseline | 0.480 | 0.521 |
| AILS NTUA winning team | 0.578 | 0.622 |
| This project hybrid mean | 0.4334 | 0.4756 |
| This project hybrid weighted | 0.4379 | 0.4816 |

Both local nDCG aggregates are below the published ELSER and winning scores; no leaderboard rank is claimed. Task scope and judged counts agree, but historical qrels identity, aggregation and baseline query construction have not been matched end to end. Local metric parity does not establish historical equivalence. Reusing the public final set for the GPU follow-up further limits comparison.

<!-- pagebreak -->

## 5 Concrete successes and failures

I inspected four validation cases selected to illustrate gains and regressions, not estimate their frequency. Scores and ranks come from saved predictions; full IDs and passages are retained in analysis/report_case_studies.json. Passage descriptions below are paraphrases except for quotations.

### Extra history can distract

A CLAPNQ question asks, “Do you have any information about how the ancient civilization deal with the stormwater?” Earlier turns discuss modern drains and catchbasins. Latest-question BM25 ranks a passage about ancient Minoan stormwater systems first. Appending user history moves it to rank 8, behind modern drainage passages, dropping nDCG@5 from 0.6131 to 0.0000. The passage was found but ordered poorly: repeated older terms are consistent with query dilution.

### History can recover an omitted subject

Another CLAPNQ message says only “The end of use of slave labor.” Earlier questions concern the cotton gin and Southern cotton. Latest-question BM25 returns general slavery-related passages and scores 0.0000. User history brings a judged cotton-gin passage to rank 1, raising nDCG@5 to 0.6131. It discusses cotton production and the growth of slavery. Context supplies the missing subject, although relevance does not establish that the passage fully answers the fragment.

### A better candidate set can help

For the Cloud question “what are worker nodes?”, the rewrite only changes capitalization. None of the three judged relevant passages enters BM25's 100-candidate pool, so BM25 with reranking scores 0.0000. The hybrid places all three at ranks 1 to 3 and scores 1.0000. They describe worker nodes carrying deployments and services and being virtual machines managed by the cluster. A reranker cannot recover missing candidates. Some BM25 results were topically related but unjudged, so zero does not imply complete semantic irrelevance.

### A rewrite can invent a setting

A Government task begins, “If I include a scoring system, how should I total the points?” The rewriter invents a multiple-choice exam setting. Hybrid fusion ranks a standardized-test T-score passage first; a relevant NASA game-tutorial passage remains at rank 12. Its nDCG@5 is 0.0000. BM25 with reranking scores 1.0000, placing the two judged passages first. One says, “Increase the score by one or more points for each sample tube collected.” The invented setting is visible, but retriever and reranking also differ, so rewriting is not isolated as the sole cause. Retaining the original query does not guarantee protection from misleading fused branches.

<!-- pagebreak -->

## 6 Ethics limitations and conclusion

Retrieval shapes the evidence a later answer generator sees. A highly ranked passage can be biased, outdated or unsuitable for a user's situation, especially in financial and government material. Relevance does not establish truth or safe advice. A deployed interface should preserve source provenance rather than present retrieved text as an authoritative answer.

The organizers describe general or fictitious questions and anonymized annotators [1]. Real conversation logs could contain sensitive information. My pipeline caches conversations and rewrites, so deployment would require data minimization, access controls and a retention policy. No commercial inference API received these inputs. The SPLADE checkpoint's CC BY NC SA 4.0 license restricts commercial reuse [3]; attribution and license review would be needed before deployment.

Incomplete judgments can penalize useful unlabelled passages. The overview also notes possible preference for sparse methods because sparse retrieval contributed to benchmark creation [1]. Performance is uneven: hybrid final nDCG@5 is 0.3086 in FiQA, 0.4608 in Cloud, 0.4885 in CLAPNQ and 0.4364 in Government. One overall score does not establish equal reliability across domains.

The main limits are the small validation set, selection among multiple configurations and reuse of a public final set for the GPU follow-up. Token truncation, sparse pruning and the candidate budget can discard evidence. Models were not adapted to these domains. E08 was not tested with a reranker, so these results do not prove that reranking would hurt that exact configuration. With another month, I would test more reliable rewriting, unsupported-context detection and matched search-call budgets, with genuinely untouched evaluation data where available.

Both neural indexes cover all 366479 passages. Dense indexing took about 36 minutes and SPLADE 156 minutes using MPS. The hybrid's final run recorded 44.0 seconds for 507 tasks, excluding index construction and cached rewrite generation; this is not end-to-end latency. Vector search runs on CPU. All runs passed format and metric checks; 26 automated tests pass. A separate Python environment reproduced BM25 exactly. Configurations, model revisions, query snapshots and frozen source hashes support reproduction [7].

The strongest validated configuration combines sparse and dense search while preserving the original question. It improves clearly over my plain BM25 baseline; its smaller gain over the stronger reranker remains uncertain. The examples explain why additional context and models require testing.

### Use of generative AI

ChatGPT and Codex assisted with implementation, debugging, evaluation checks, and drafting and formatting this report. Scores were computed by the saved evaluation scripts. This assistance does not replace my responsibility to understand and review the submitted work.

<!-- pagebreak -->

## References

[1] Sara Rosenthal, Vraj Shah, Yannis Katsis and Marina Danilevsky. 2026. SemEval 2026 Task 8 MTRAGEval Evaluating Multi Turn RAG Conversations. Proceedings of SemEval, pages 3673 to 3690. Sections 2, 3 and 5; Table 4. https://aclanthology.org/2026.semeval-1.447/

[2] IBM Research. MTRAG benchmark data and evaluation code. Project version e0970de1b1207c2a9506adf238e6e1a22037e38a. https://github.com/IBM/mt-rag-benchmark/tree/e0970de1b1207c2a9506adf238e6e1a22037e38a

[3] NAVER. SPLADE ensemble distilled checkpoint and model documentation. https://huggingface.co/naver/splade-cocondenser-ensembledistil

[4] Beijing Academy of Artificial Intelligence. BGE small English version 1.5 model documentation. https://huggingface.co/BAAI/bge-small-en-v1.5

[5] Qwen Team. Qwen2.5 0.5B Instruct model documentation. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

[6] Sentence Transformers. MS MARCO MiniLM L6 v2 cross encoder documentation. https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

[7] Project source code and saved experiments. https://github.com/Neeraj0704/cs421-conversational-retrieval
