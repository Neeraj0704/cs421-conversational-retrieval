# Automated walkthrough transcript

## The task and the working system

This is an automated narrated walkthrough of the CS 421 conversational retrieval project. Given a conversation ending in a user question, the program searches the assigned collection and returns ten original passages. It implements Subtask A, retrieval only. It does not generate a final answer. The repository includes code, reproducible experiment configurations, evaluation checks and measured results.

## Data separation and leakage prevention

The data module pins the upstream IBM commit and preserves the original passage identifiers. It uses a field whitelist, so gold answers, reference contexts and labels do not enter the retrieval model. The development split is by whole conversation, not individual turns. Twenty two conversations are held out for validation. Final test configurations are frozen before scores are used.

## Rewriting and preserving the original question

The small local Qwen model rewrites the latest question using the supplied past conversation. The prompt asks it to resolve references while preserving names, dates and constraints, without answering the question. A rewrite can be wrong. The fusion experiment therefore searches both the original question and the rewrite. Each search produces a ranking, and reciprocal rank fusion combines their positions.

## Retrieval, fusion and reranking

BM 25 provides a fast lexical baseline over the complete original collections. Reciprocal rank fusion combines query views without assuming their numeric scores are calibrated. A Mini L M cross encoder then scores the query with each candidate passage. We compare pure reranking and a fifty fifty blend of retrieval and reranker ranks, using the same candidate pool. This separates better retrieval from better ordering.

## Scoring and verification

The primary metric is n D C G at five. Recall at ten measures final retrieval coverage, and candidate recall at one hundred diagnoses whether a relevant passage was available to the reranker. The official metric function matches our wrapper on the baseline. The official format checker and automated tests check output coverage, duplicates, score validity, ID preservation and leakage. Both the direct average and the upstream style weighted aggregate are recorded.

## Measured results and practical limits

The strongest completed validation run is R00 bm25 latest rerank, with n D C G at five of 0.3898. The frozen selected system scores 0.4161 on the final test, averaging over 332 judged questions out of 507 inputs. The screen shows only completed, format checked experiments. Development and final results are separate. Historical leaderboard equivalence is not established. The execution session has CPU access but no Metal GPU. Full learned sparse and dense indexing were deferred after timing their cost. Review the report, negative results and limitations, and adapt any submission to the actual course instructions. This automated video is a review aid; it is not a student recorded presentation.
