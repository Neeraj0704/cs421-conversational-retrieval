# Automated walkthrough transcript

## The task and the working system

This is an automated narrated walkthrough of the CS 421 conversational retrieval project. Given a conversation ending in a user question, the program searches the assigned collection and returns ten original passages. It implements Subtask A, retrieval only. It does not generate a final answer. The repository includes code, reproducible experiment configurations, evaluation checks and measured results.

## Data separation and leakage prevention

The data module pins the upstream IBM commit and preserves the original passage identifiers. It uses a field whitelist, so gold answers, reference contexts and labels do not enter the retrieval model. The development split is by whole conversation, not individual turns. Twenty two conversations are held out for validation. CPU and GPU final configurations were frozen separately. The GPU follow up reuses a public final set already evaluated in the CPU stage.

## Rewriting and preserving the original question

The small local Qwen model rewrites the latest question using the supplied past conversation. The prompt asks it to resolve references while preserving names, dates and constraints, without answering the question. A rewrite can be wrong. The fusion experiment therefore searches both the original question and the rewrite. Each search produces a ranking, and reciprocal rank fusion combines their positions.

## Retrieval, fusion and reranking

BM 25 provides a fast lexical baseline over the complete original collections. Reciprocal rank fusion combines query views without assuming their numeric scores are calibrated. A Mini L M cross encoder then scores the query with each candidate passage. We compare pure reranking and a fifty fifty blend of retrieval and reranker ranks, using the same candidate pool. The GPU study adds learned sparse SPLADE and B G E semantic search. Its selected hybrid combines original and rewritten query rankings without a cross encoder. This separates better retrieval from better ordering.

## Scoring and verification

The primary metric is n D C G at five. Recall at ten measures final retrieval coverage, and candidate recall at one hundred diagnoses whether a relevant passage was available to the reranker. The official metric function matches our wrapper on the baseline. The official format checker and automated tests check output coverage, duplicates, score validity, ID preservation and leakage. Both the direct average and the upstream style weighted aggregate are recorded.

## Measured results and practical limits

The strongest GPU validation run is hybrid fusion, with n D C G at five of 0.4479. The prespecified hybrid primary scores 0.4334 on the final follow up, over 332 judged questions out of 507 inputs. Recall at ten is 0.6211. The improvement over plain B M 25 is clear. The smaller gain over B M 25 plus reranking has a paired uncertainty interval including zero. Both complete neural indexes and all twelve GPU validation runs finished in the user's Metal enabled Terminal. The agent process itself could not access Metal. These are follow up results on a public final set already used for the CPU study. Historical leaderboard equivalence remains unverified. Review the combined report and the actual course instructions. This synthetic voice video is an automated review aid, not a student recorded presentation.
