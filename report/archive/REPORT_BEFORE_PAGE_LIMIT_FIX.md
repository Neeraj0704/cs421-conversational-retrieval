# Improving conversational passage retrieval

Neeraj Vijayakumar Pattanashetti

CS 421 Natural Language Processing

November 25 2026

## Overview

I built a system that retrieves ten passages for the latest question in a conversation. The project addresses SemEval 2026 Task 8, MTRAGEval Subtask A. I compared 19 configurations on held-out conversations, starting with BM25 word matching and adding query rewriting, learned sparse retrieval, dense retrieval and reranking. The best validation configuration combined SPLADE and BGE searches for both the original question and a standalone rewrite. Its final nDCG@5 was 0.4334, compared with 0.2896 for my BM25 baseline and 0.4161 for BM25 with a reranker. The improvement over the stronger reranker baseline was small and its uncertainty interval included zero. The clearest practical finding was that combining search signals helped, while adding more context or another model did not always help.

## 1 Task and evaluation

A follow-up such as “What about its cost?” is difficult to search without knowing what “it” refers to. My task is to use the conversation supplied so far, search the assigned collection and return ten original passage IDs in relevance order. I implement retrieval only; I do not generate an answer. Each task ends at the current user question, so the system has no access to future turns [1, 2].

The official ranking metric is normalized discounted cumulative gain at five, or nDCG@5 [1]. It rewards relevant passages in the first five positions, with more credit for passages near the top. The score is divided by the score of an ideal ranking for that question, making the result comparable across questions. A perfect ranking scores 1. For a question with exactly one relevant passage, placing it first gives 1, whereas placing it fifth gives about 0.387. A score of 0.4334 is a ranking measure, not 43.34 percent answer accuracy.

I also report Recall@10, the fraction of judged relevant passages recovered in the ten returned results, averaged over questions. Candidate Recall@100 measures coverage in the larger candidate pool before the final ten are selected. This separates failure to find a useful passage from failure to rank it highly.

Two aggregation methods are kept separate. My main score is the arithmetic mean over judged questions. I also reproduce the upstream script's domain aggregation: round each domain score to five decimals, weight it by the number of submitted tasks in that domain, and average. The hybrid scores 0.4334 by the first method and 0.4379 by the second. I never mix these aggregates within a local comparison.

## 2 Data and experimental setup

I used the official IBM passage-level archives and task files at commit e0970de1b1207c2a9506adf238e6e1a22037e38a. Counts below come from reading the archives rather than copying a summary table. Every question searches only its assigned collection.

| Collection | Content | Passages used |
|---|---|---:|
| CLAPNQ | Wikipedia passages | 183408 |
| Cloud | Technical documentation | 72442 |
| FiQA | Financial questions and answers | 61022 |
| Government | Government and public information | 49607 |
| Total | Four original collections | 366479 |

MTRAG contains 842 conversational tasks from 110 conversations. Using seed 42, I split whole conversations within each domain: 88 conversations in the remaining development partition and 22 in validation. Validation contains 174 tasks, of which 161 have relevance judgments. All 19 validation comparisons use those same 161 judged tasks. I did not train model weights on the remaining development partition; all neural components were pretrained.

The released final task file contains 507 inputs. My qrels, the files that identify relevant passages, score 332 of them; the remaining 175 receive predictions but do not enter the retrieval quality average. The available labels cover answerable and partially answerable tasks; other question types still receive outputs.

### Differences from the published setup

I retained the official passage boundaries, collection assignments and offset-bearing IDs. I used all passages in the four archived collections, with no added documents, synthetic training examples or custom rechunking. The published SemEval task used the same four-domain, 507-task scope [1]. Neural input truncation and SPLADE term pruning affect the index representation, not the stored source passages.

My 22-conversation validation split is a local choice. It differs from evaluation on the complete development set, so my validation scores should not be compared directly with full-development results. The GPU study also followed an earlier CPU evaluation on the public final set. I froze GPU configurations using validation results before their new final runs, but these remain follow-up evaluations on a previously used test set rather than a fresh unseen holdout.

### Controls and software

The model input loader keeps only task identifiers, the collection and the supplied speaker-labelled conversation. Gold answers, relevance lists and provided benchmark rewrites are excluded. Evaluation reads relevance labels only after predictions are produced. I checked archive integrity, original IDs, collection routing and exact prediction coverage.

The implementation uses Python 3.11, bm25s and PyStemmer for lexical retrieval; PyTorch and Transformers for neural inference; Sentence Transformers for the cross-encoder; NumPy and SciPy for vector storage and search; and pytrec-eval-terrier for metrics. Dependencies are locked in uv.lock. Model revisions, data hashes, generated queries and per-run settings are saved in the repository. No paid model API or external training dataset was used.

## 3 Approach and motivation

I tested components separately because each solves a different problem. Conversation context can resolve an incomplete question, but old topics can also distract a search. Sparse models retain useful vocabulary signals, dense models can match similar meanings, and a reranker can read candidate passages more carefully. My experiments ask when those benefits survive in this particular task and resource budget.

### Query handling

The latest-only baseline searches the current user question. The history baseline concatenates all earlier user questions with the current one. It does not concatenate assistant responses. For rewriting, Qwen2.5-0.5B-Instruct [5] receives the full supplied conversation prefix with speaker labels. The prompt asks it to resolve references, preserve names, dates, negation and numerical constraints, and return one standalone search question without answering or adding facts. Assistant messages are context for reference resolution, not verified evidence.

Rewrites use greedy decoding with at most 128 new tokens. An empty rewrite falls back to the original question. Generated rewrites are cached with the prompt, model revision and conversation content so experiments reuse identical query text. Fusion searches both original and rewritten queries; exactly identical views are removed. The keyword experiment adds a simple rewrite-derived view with selected common words removed.

### Retrieval models

| Component | Model or method | Main settings |
|---|---|---|
| Lexical retriever | BM25 using bm25s | English stopwords and Snowball stemming; k1 1.2; b 0.75 |
| Sparse retriever | SPLADE ensemble distilled | 256 tokens; maximum log one plus ReLU pooling; 128 document terms |
| Dense retriever | BAAI/bge-small-en-v1.5 | 256 tokens; normalized 384-dimensional CLS vectors; exact inner-product search |
| Rewriter | Qwen/Qwen2.5-0.5B-Instruct | Full supplied prefix; greedy decoding; 128 new tokens |
| Reranker | MiniLM L6 v2 cross encoder | 384 tokens per pair; inference batch size 32 |

BM25 provides a fast, interpretable baseline for exact terms. The SPLADE checkpoint is naver/splade-cocondenser-ensembledistil. It gives vocabulary terms learned weights and can expand the matching signal beyond literal overlap [3]. I form sparse vectors by applying log one plus ReLU to token logits, masking padding and taking the maximum over token positions. Document vectors retain their 128 largest term weights; query vectors are not pruned this way.

BGE encodes passages and questions separately [4]. I normalize the CLS vector, then compare each query with the stored vectors from its collection. Query text receives the documented prefix “Represent this sentence for searching relevant passages: ”. Dense search is exact rather than approximate. These choices allow a controlled, compact semantic branch without introducing a separate approximate-search parameter sweep.

### Fusion and reranking

Each retriever and query view produces up to 100 candidates. For each list containing a passage, reciprocal rank fusion adds 1 divided by the sum of 60 and that passage's rank. All query and retriever lists have equal weight. This combines positions without assuming that BM25, sparse and dense scores share a scale. The combined pool is truncated to 100.

In reranked configurations, cross-encoder/ms-marco-MiniLM-L6-v2 reads the question and each candidate passage together [6]. Fusion configurations use the rewrite as the rerank query; latest-only configurations use the latest question. Pure reranking follows its order. The blended version instead combines the retrieval and reranker lists through RRF with equal weights of 0.5. The ten highest-ranked passages are returned. Submission scores encode that final order strictly, avoiding ties.

The selected E08 configuration has up to four searches: original and rewritten questions through SPLADE, and the same two views through BGE. It combines their rankings and returns ten passages without a cross-encoder. It uses more search calls than a single-query baseline, although the final candidate pool remains 100. The frozen settings and exact model revisions are included with the code; no model was fine-tuned for this study.

## 4 Results

I first completed seven CPU comparisons, then twelve comparisons using full neural indexes built through Apple Metal. Table 3 lists every validation configuration. nDCG@5 is the mean over the same 161 judged questions. The CPU and GPU labels describe experiment stages; they do not imply that all search arithmetic ran on the GPU.

| Run | Validation configuration | nDCG at 5 |
|---|---|---:|
| E00 | BM25 latest question | 0.2249 |
| E01 | BM25 raw user history | 0.1842 |
| E02 | BM25 rewrite only | 0.2114 |
| B02 | BM25 original and rewrite fusion | 0.2472 |
| B03 | B02 with pure reranking | 0.3464 |
| B04 | B02 with retrieval and reranker rank blend | 0.3138 |
| R00 | BM25 latest with reranking | 0.3898 |
| D00 | Dense latest question | 0.3850 |
| D01 | Dense original and rewrite fusion | 0.3966 |
| G02 | Dense latest with reranking | 0.4366 |
| E03 | SPLADE latest question | 0.3990 |
| E04 | SPLADE rewrite only | 0.3611 |
| E05 | SPLADE original and rewrite fusion | 0.4107 |
| E06 | E05 with pure reranking | 0.3795 |
| E07 | E05 with retrieval and reranker rank blend | 0.4121 |
| E10 | SPLADE original rewrite and keyword fusion | 0.3909 |
| G01 | SPLADE latest with reranking | 0.4281 |
| G03 | Sparse and dense latest with reranking | 0.4291 |
| E08 | Sparse and dense original and rewrite fusion | 0.4479 |

Raw history and rewrite-only BM25 underperformed the latest-question baseline. Preserving both query views helped BM25 and SPLADE. Reranking strongly helped BM25, but reduced SPLADE fusion from 0.4107 to 0.3795 on the same candidates. Blending ranks recovered that loss to 0.4121. Adding the keyword view did not improve SPLADE fusion. These results argue for testing each component rather than assuming that a longer pipeline is better.

E08 had the highest validation nDCG@5 and was selected before the GPU final runs. Against the earlier CPU R00, it improved 61 validation questions, tied on 62 and worsened 38. Its validation candidate Recall@100 was 0.7917, almost the same as SPLADE fusion's 0.7925, while nDCG@5 rose from 0.4107 to 0.4479. On this split, the hybrid's benefit was therefore not simply finding a larger fraction of relevant candidates; the combined ordering also mattered.

### Final comparisons

The first three rows below are the earlier frozen CPU evaluations. The last three are the frozen GPU follow-up. Every row returns ten passages for 507 tasks and is scored on the same 332 judged questions. Mean and weighted nDCG use the distinct aggregates described in Section 1.

| Final system | Mean nDCG at 5 | Weighted nDCG at 5 | Recall at 10 | Candidate recall at 100 |
|---|---:|---:|---:|---:|
| BM25 latest | 0.2896 | 0.2918 | 0.4261 | 0.6989 |
| BM25 query fusion | 0.3306 | 0.3332 | 0.4872 | 0.8187 |
| BM25 latest with reranker | 0.4161 | 0.4188 | 0.5347 | 0.6989 |
| SPLADE latest with reranker | 0.4199 | 0.4239 | 0.5470 | 0.7570 |
| Dense latest with reranker | 0.4040 | 0.4082 | 0.5309 | 0.6810 |
| Selected hybrid query fusion | 0.4334 | 0.4379 | 0.6211 | 0.8610 |

The hybrid improved mean nDCG@5 over plain BM25 by 0.1438. A paired bootstrap with 3000 whole-conversation resamples and seed 42 gave a 95 percent interval of 0.1017 to 0.1879. Against BM25 with reranking, the difference was only 0.0173, with an interval of -0.0178 to 0.0525. That interval includes zero, so the smaller ranking gain is not statistically resolved by this analysis. The final hybrid also recovered more relevant passages in its top ten, with Recall@10 of 0.6211 versus 0.5347 for the BM25 reranker. The intervals are exploratory and do not correct for multiple comparisons or validation selection.

### Published baseline and top score

The figures below come from Table 4 of the task overview, which reports the final Subtask A leaderboard, not the earlier MTRAG development baselines [1]. My local BM25 baseline is a separate implementation and must not be confused with the organizers' BM25 result.

| Published system | nDCG at 5 | Recall at 5 |
|---|---:|---:|
| BM25 baseline | 0.354 | 0.388 |
| BGE base 1.5 baseline | 0.390 | 0.425 |
| ELSER strongest baseline | 0.480 | 0.521 |
| AILS NTUA winning team | 0.578 | 0.622 |

My selected hybrid's local mean nDCG@5 is 0.4334 and mean Recall@5 is 0.4756; the weighted values are 0.4379 and 0.4816. Numerically, both nDCG aggregates are below the published ELSER and winning scores. I do not claim to beat those systems or assign myself a leaderboard rank. Although the broad task scope and judged-task count agree, exact historical qrels identity, aggregation and baseline query construction have not been matched end to end. Official metric parity and valid output files establish local evaluation correctness, not complete historical equivalence. Differences between my BM25 score and the published BM25 score reinforce the need for this qualification.

## 5 Analysis of successes and failures

I inspected four validation cases to explain the aggregate results. They were selected to illustrate gains and regressions, not sampled to estimate how frequently a behaviour occurs. Scores and ranks below come from saved predictions and relevance files. Passage descriptions are paraphrases unless shown in quotation marks. The full task IDs and supporting passages are retained in analysis/report_case_studies.json.

### History can distract from the current question

In one CLAPNQ conversation, the seventh question is “Do you have any information about how the ancient civilization deal with the stormwater?” Earlier turns discuss modern storm drains, catchbasins, litter and recycling runoff.

Latest-question BM25 puts a relevant historical passage first. It describes Minoan stormwater systems, including drains and channels in ancient Crete. With all earlier user questions appended, that same passage falls to rank 8. The first result instead defines a modern storm drain and the next discusses catchbasins. nDCG@5 drops from 0.6131 to 0.0000.

The useful historical passage was still found, but repeated vocabulary from older turns dominated the top of the list. This is a ranking failure consistent with query dilution. It explains why raw history can hurt even when every earlier message is related to the broad topic.

### History can recover a missing topic

In another CLAPNQ conversation, the seventh message is only “The end of use of slave labor.” Earlier questions concern the cotton gin, cotton production, the American Civil War and Southern cotton.

Latest-question BM25 returns general slavery-related material, including a passage about plantation labour in the Columbian Exchange. Neither judged relevant passage appears in its top five. Adding user history moves a relevant cotton-gin passage to rank 1, raising nDCG@5 from 0.0000 to 0.6131. The passage connects the invention to cotton production and the growth of slavery in the American South.

This example shows why simply ignoring context is also inadequate. The current utterance omits the subject; earlier turns supply it. The relevance label supports retrieving this passage, but I should not read the score as proof that it fully answers the user's fragment.

### Hybrid search can recover candidates that BM25 missed

In a Cloud conversation, the second question is “what are worker nodes?” The rewrite is just “What are worker nodes?” It changes capitalization rather than adding useful context.

The BM25 reranker configuration scores 0.0000. Its first result concerns planning worker-node setup, but none of the three judged relevant passages reached its 100-candidate pool. The hybrid places all three judged passages at ranks 1 to 3 and scores 1.0000. One explains that worker nodes carry the deployments and services of an application; another defines them as virtual machines managed by the cluster.

A reranker cannot promote a passage it never receives. This case illustrates the value of the hybrid's candidate set, while also showing that the improvement did not come from a more informative rewrite. Some BM25 results are topically related but unjudged, so the zero score should not be interpreted as total semantic irrelevance.

### A rewrite can invent the wrong setting

One Government task begins with “If I include a scoring system, how should I total the points?” The supplied prefix has no earlier turns. The relevant passage is from a NASA Mars sample-collection game tutorial and includes “Increase the score by one or more points for each sample tube collected.”

The rewriter produces “How do you calculate the total score for multiple-choice questions with different options?” Nothing in the input supplies an exam setting. The hybrid's first result describes standardized test T-scores, while a relevant game passage remains at rank 12 in its fused candidate list. Its nDCG@5 is 0.0000. BM25 with reranking places the two judged relevant passages first and second and scores 1.0000.

The added exam context is directly visible in the rewrite. Its role in the failed fusion is plausible, but this comparison also changes the retriever and reranking, so it does not isolate rewriting as the sole cause. Keeping an original branch reduces dependence on a rewrite; it does not guarantee protection when misleading branches influence the final order.

## 6 Ethics and limitations

Retrieval determines which information a later answer generator is likely to see. A confidently ranked passage can still be outdated, biased or unsuitable for the user's situation. This matters especially for financial and government material. My system returns evidence and makes no claim that a high relevance score establishes factual correctness or safe advice. A deployed interface should preserve provenance and distinguish source content from a generated answer.

The organizers describe the benchmark questions as general or fictitious and report anonymizing annotators [1]. Real conversations could still contain account details or other sensitive information. My local pipeline saves conversations and rewrites for reproducibility, so deployment would require data minimization, restricted access and a retention policy. I would not publish real user logs merely because benchmark logs are shareable. No commercial inference API received the benchmark inputs in this study.

Incomplete relevance judgments can penalize useful but unlabelled passages. In addition, the overview notes that a sparse retriever was involved in benchmark creation, which may favour passages found by similar methods [1]. These are reasons to avoid treating one aggregate score as a complete measure of user benefit. My own final hybrid score also varies by domain: 0.3086 in FiQA, 0.4608 in Cloud, 0.4885 in CLAPNQ and 0.4364 in Government. These figures show uneven performance, not evidence of equal reliability across domains.

I used released data and pretrained models under their respective licenses. The SPLADE checkpoint is marked CC BY NC SA 4.0, which restricts commercial reuse [3]. The repository does not redistribute model weights or the large corpora. Attribution and license review would be necessary before a different deployment.

The main methodological limits are the small validation sample, selection across several configurations and the GPU follow-up's reuse of a public final set. The small Qwen rewriter and MS MARCO reranker were not adapted to these collections. Token truncation, sparse pruning and the 100-candidate budget can discard useful evidence. E08 also uses more search calls than simpler baselines. I did not directly test E08 followed by the reranker, so the experiments do not prove that reranking would hurt that exact hybrid.

The next useful experiment is to improve query reliability: keep already explicit questions unchanged, detect unsupported additions and evaluate a stronger rewriter on a new development protocol. Other bounded comparisons could test longer encoder inputs, sparse pruning, domain-specific reranking and a matched search-call budget. Any new selection should be evaluated on genuinely untouched data if available. Fine-tuning, ColBERT and adaptive retrieval remain future work rather than completed results.

## 7 Reproducibility and conclusion

Both full neural indexes contain 366479 original passages. On the Apple Silicon Mac, dense indexing took about 36 minutes and SPLADE about 156 minutes using MPS. The full GPU validation workflow took about 3 hours 19 minutes. The selected final hybrid recorded 44.0 seconds for 507 tasks, with median per-query time 0.0629 seconds. Those timings use cached rewrites and exclude index construction and rewrite generation; they are not an end-to-end latency claim. Vector search runs on CPU while supported neural inference uses MPS.

All recorded predictions passed the official format checker and independent coverage and metric checks. The project has 26 passing automated tests. A separate Python environment reproduced the BM25 baseline with zero metric differences. Source and configuration fingerprints preserve the frozen evaluations. Exact query snapshots make it possible to replay the same experiments without regenerating different rewrites.

The code and run artifacts are available in the project repository [7]. scripts/reproduce.py recreates the CPU comparisons in an isolated output folder. scripts/start_gpu_study.command and scripts/start_gpu_final.command restore and resume the separate GPU study. The original CPU checkpoint is preserved; changes to models or settings should use a new study rather than overwrite these results.

My strongest validated configuration combines complementary retrieval methods and retains the original question alongside its rewrite. It improves clearly over my plain BM25 baseline, but its smaller nDCG gain over the stronger BM25 reranker remains uncertain. The failure examples explain why the result is not a simple endorsement of adding more components: context can distract, a rewrite can invent a setting, and reranking depends on the candidates it receives.

AI assistance was used in implementation, experiment verification and report preparation. The reported scores come from saved program outputs. This assistance does not replace the author's responsibility to understand the methods and review the submitted work.

## References

[1] Sara Rosenthal, Vraj Shah, Yannis Katsis and Marina Danilevsky. 2026. SemEval 2026 Task 8 MTRAGEval Evaluating Multi Turn RAG Conversations. Proceedings of SemEval, pages 3673 to 3690. Sections 2, 3 and 5; Table 4. https://aclanthology.org/2026.semeval-1.447/

[2] IBM Research. MTRAG benchmark data and evaluation code. Archived project version e0970de1b1207c2a9506adf238e6e1a22037e38a. https://github.com/IBM/mt-rag-benchmark/tree/e0970de1b1207c2a9506adf238e6e1a22037e38a

[3] NAVER. SPLADE ensemble distilled checkpoint and model documentation. https://huggingface.co/naver/splade-cocondenser-ensembledistil

[4] Beijing Academy of Artificial Intelligence. BGE small English version 1.5 model documentation. https://huggingface.co/BAAI/bge-small-en-v1.5

[5] Qwen Team. Qwen2.5 0.5B Instruct model documentation. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

[6] Sentence Transformers. MS MARCO MiniLM L6 v2 cross encoder documentation. https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2

[7] Project source code and saved experiments. https://github.com/Neeraj0704/cs421-conversational-retrieval
