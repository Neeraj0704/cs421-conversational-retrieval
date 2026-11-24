# CS 421 project implementation plan and tracker

**Project:** Improving Multi-Turn Retrieval with Conversation History  
**Task:** SemEval-2026 Task 8, MTRAGEval, Subtask A  
**Plan date:** November 23, 2026 — America/Chicago  
**Status:** Seven CPU and twelve GPU validation experiments, full neural indexes and six total final comparisons complete. Combined report and review video updated; course-format review remains.
**Working folder:** /Users/neerajvp/Documents/CS421 NLP/NLP/implementation

## 1. What we are trying to finish

Build a reproducible program that reads a conversation prefix and the latest question, searches the assigned document collection, and returns ten ranked original passage IDs. Measure whether conversation history, rewriting, retrieval choice, and reranking improve the results.

Our course deliverables are a working implementation, measured results, an explanatory final report, and a video code walkthrough. The proposal instructions state November 25 for the report and walkthrough; the exact cutoff, report format, video length, code-submission requirements, and final rubric still need confirmation from the separate final-project brief.

The scientific objective is to improve our own strong baselines and compare the frozen system with the historical winner if the evaluation protocol can be matched. Beating the published score is an ambition, not a completion requirement or a promised outcome.

**Immediate priority:** review the completed results and adapt deliverables to the final course brief. See the latest completion checkpoint below; earlier checkpoints record project history.

## 2. How we will use the research

The reports are inputs to the design, not substitutes for primary-source checks or our own experiments. Their paper lists overlap; their counts must not be added together. The original request for at least 50 primary papers examined beyond their abstracts is not independently verified as complete. One report explicitly falls short.

Before implementing an unusual detail, inspect the primary paper, official model documentation, or source code that justifies it. Focus those checks on components we actually choose. A completed 50-paper audit remains a separate literature-review task; it must not silently consume the time reserved for implementation and submission.

### Research-to-experiment map

| Research finding or disagreement | Implementation decision | How we will test it | Priority |
|---|---|---|---|
| Follow-up questions need earlier context | Compare latest question, raw question history, and one conservative rewrite | Hold retriever fixed across query formulations | Core |
| Rewriting can remove useful terms | Preserve the original question and search it separately | Compare rewrite-only with original-plus-rewrite fusion | Core |
| Learned sparse models have strong exact-task evidence | Include SPLADE, or exact ELSER version if feasible | Compare latest-only and rewritten queries on the same sparse index | Core, subject to resource gate |
| Some research favors many query views; other work finds expansion harmful | Add a keyword view before considering a larger set | Compare one, two, and three views with controlled candidate budgets | Stretch |
| Rerankers improve ordering but can also override useful retrieval signals | Save both retrieval and reranker ranks | Compare no reranking, pure reranking, and rank fusion on identical candidates | Core |
| Dense retrieval can complement sparse retrieval but may add noise | Keep dense retrieval modular | Compare sparse-only with sparse+dense under a fixed final pool | Stretch |
| ColBERT offers promising fine-grained matching, without a proven win here | Treat it as a bounded modern-method experiment | Compare sparse+ColBERT with sparse-only using the same queries and reranker | Stretch after feasibility check |
| History selection can help topic changes; full history may already be manageable | Start with speaker-aware conversation prefixes | Compare complete available history with selected history on development only | Stretch |
| Sparse and dense retrievers may prefer different rewrites | Add retriever-specific query styles only after a hybrid benefit appears | Compare specific styles with generic views at the same call count | Later |
| RL, preference training, and agent loops have weaker direct evidence or greater cost | Start from pretrained models and a fixed pipeline | Consider training or a bounded second search only after diagnosing a concrete failure | Later |
| Historical evaluation and current code may differ | Audit data routing, query counts, and aggregation first | Check official archived predictions or known-score artifacts where available | Core |

We will not adopt absolute statements from a report such as “never use pure reranking” or “hybrid is best” as universal rules. Those are hypotheses to compare.

## 3. Initial system design

The first strong candidate is:

**Speaker-aware conversation → original question and conservative rewrite → learned sparse retrieval → rank fusion → compact reranker → ten original passage IDs.**

A separate BM25 implementation supplies the simplest comparisons. A dense or ColBERT branch is added only after the core works and resources permit.

### Component choices

| Component | Initial choice | Alternative or fallback |
|---|---|---|
| Basic retrieval | BM25 with a documented tokenizer | Efficient CPU index implementation selected during setup |
| Strong sparse retrieval | An accessible SPLADE checkpoint, with exact version and terms recorded | ELSER only if access/version/deployment are practical; BM25 remains usable |
| Rewriter | Existing instruction model selected after a small timing and quality check | Smaller local model; raw-history baseline if model inference cannot finish in time |
| Fusion | Simple RRF; initial rank constant 60 | Small validation sweep only if justified |
| Reranker | BGE-reranker-v2-m3 candidate | Smaller MiniLM-class model or Qwen3-Reranker-0.6B if feasible |
| Dense experiment | One compact audited BGE model; BGE-M3 is a candidate | No dense component if indexing is too slow |
| Late-interaction experiment | ColBERTv2 candidate | Defer if setup, indexing, or memory costs threaten core deliverables |

These are candidate choices, not installed components or guaranteed hardware fits. Check library compatibility, checkpoint access, inference time, and memory before downloading large weights or encoding the full corpus.

### Rewriter behavior

The rewriter must:

- Resolve only the latest question.
- Use past turns only to recover necessary entities and constraints.
- Preserve negation, dates, numerical limits, and exact technical names.
- Return an already explicit question unchanged when appropriate.
- Treat assistant statements as context for reference resolution, not verified truth.
- Avoid inventing answers or unsupported facts.
- Produce a machine-readable result with a safe fallback to the original question.
- Save its output so failures can be inspected and experiments replayed.

Initially provide the complete available prefix if it fits the chosen model's supported input. If truncation is necessary, log it and use a deterministic policy. A learned history selector is optional.

## 4. Milestones with completion gates

Do not proceed past a gate when a failure would invalidate later scores.

### M0 — Confirm scope and resources

**Tasks**

- [ ] Obtain the final-project instructions and confirm deadline time/timezone, report format, walkthrough length, code packaging, citations, and any assistance-disclosure requirements.
- [x] Record Python version, operating system, available memory, accelerator support, and free disk.
- [x] Choose a local environment and a minimal dependency set for retrieval evaluation.
- [x] Audit candidate checkpoint access and licenses.
- [x] Select one rewriter and one reranker for a short feasibility check.
- [x] Record decisions and unresolved items in the decision log.

**Done when:** we have a feasible core configuration and an explicit list of unknown submission requirements.

If the final brief remains unavailable, continue independent setup and experiments, but do not invent page limits, video length, or rubric items.

### M1 — Download and audit official data

**Tasks**

- [x] Pin the IBM repository commit.
- [x] Download official passage-level corpora, MTRAG tasks, relevance files, and evaluation/format-checking code.
- [x] Verify archive integrity and inventory actual record counts.
- [x] Inspect several real examples across all four domains.
- [x] Preserve passage boundaries and identifiers.
- [x] Verify relevance-file IDs against actual corpus IDs; document any ingestion-offset mapping explicitly.
- [x] Establish query-to-collection routing.
- [x] Save file checksums and an immutable data manifest.
- [x] Prepare final-test inputs and scoring files separately; do not inspect final per-query relevance to guide development.

**Done when:** sampled tasks route correctly, all relevant corpus IDs can be resolved, counts are documented, and a fresh run can locate every required file.

Published counts differ between some reports and repository descriptions. Record actual counts from the pinned revision rather than forcing a reported number. Never strip identifier suffixes merely to make scores rise.

### M2 — Establish trustworthy evaluation

**Tasks**

- [x] Make a small valid prediction file and pass the official format checker.
- [x] Fix explicit paths/configuration needed to select development versus final relevance files; record every change.
- [x] Confirm how numerical scores determine order.
- [x] Check duplicate IDs, ties, omitted tasks, empty results, wrong-domain IDs, and nonfinite scores with small controlled inputs.
- [x] Compare the official-style domain-weighted aggregate with a direct mean over scored queries.
- [x] Count submitted, scored, unscored, and missing judged queries separately.
- [x] Use a known-score official prediction artifact as a reproduction check if available.
- [x] Verify a tiny hand-understood ranking against the metric calculation.

**Done when:** scores respond as expected, query coverage is explicit, and there is no silent use of development qrels for final evaluation.

Tasks with no judgments are not automatically retrieval failures. A required judged task missing from our output must be reported and repaired; it must not disappear from our denominator.

### M3 — Build and record baseline experiments

**Tasks**

- [x] Create and save the development split.
- [x] Build the BM25 index for each assigned collection.
- [x] Run latest-question-only BM25.
- [x] Run BM25 with prior user questions plus the latest question.
- [x] Generate and cache conservative rewrites, if a model is feasible.
- [x] Run BM25 using those rewrites.
- [x] Save predictions, per-query scores, per-domain metrics, runtime, and configuration.

**Done when:** the original project question can be answered with real comparisons, even if later enhancements fail.

### M4 — Build the strong research-based candidate

**Tasks**

- [x] Time sparse encoding on a sample and estimate full indexing work.
- [x] Build one learned sparse index if it passes the feasibility gate.
- [x] Run sparse retrieval with latest question and with rewrite separately.
- [x] Fuse original and rewrite rankings.
- [x] Rerank a fixed candidate pool with the selected compact model.
- [x] Compare pure reranking with a blend of retrieval and reranker ranks.
- [x] Check retrieval and ranking failures separately.

**Done when:** the main pipeline runs across the development evaluation set, produces ten valid passages per task, and each stage has a measured contribution.

If learned sparse indexing cannot fit the schedule, complete the same query/fusion/reranking comparison over BM25 and document the narrower scope.

### M5 — Run bounded modern-method experiments

**Order**

1. One dense channel.
2. ColBERT only if setup/indexing is feasible and core deliverables are secure.
3. One additional query view or history-selection experiment, chosen from observed failures.

**Tasks**

- [x] Choose one experiment based on current error analysis and remaining time.
- [x] Hold the final reranking pool and model fixed.
- [ ] Measure unique relevant candidates contributed by the new branch.
- [ ] Record quality, indexing cost, query time, and memory.
- [x] Keep or reject the component using the rules below.

**Done when:** there is a documented outcome for each attempted extension, including negative results or feasibility failures.

No requirement to run every method family. RL training, full retriever fine-tuning, graph construction, and unrestricted agent search remain future work unless the schedule is extended.

### M6 — Freeze and evaluate the final candidates

**Tasks**

- [x] Select the primary method from development results.
- [x] Freeze code/configuration, model revisions, prompts, and selected run list.
- [x] Preserve our simple baseline and strongest sparse baseline as final comparisons.
- [x] Generate final predictions before examining final relevance outcomes.
- [x] Validate task coverage, ten unique IDs per task, collection routing, finite scores, and deterministic ordering.
- [x] Score the prespecified runs on the historical competition subset.
- [x] Report official-style and per-scored-query aggregates with clear labels.
- [x] Compare with published results only to the degree the protocol has been established.

**Done when:** final results have complete provenance and comparison limits are stated.

An implementation bug may justify a corrected run with an explicit explanation. Changing models or prompts after seeing test scores is post-test tuning and must be labeled separately.

### M7 — Finish and package the course submission

**Tasks**

- [ ] Complete report according to the actual final brief.
- [x] Include results, ablations, errors, limitations, references, and resource measurements.
- [x] Explain which components follow earlier papers and what we implemented or adapted.
- [x] Write a README with setup, data preparation, indexing, retrieval, and evaluation commands.
- [x] Pin dependencies and model versions.
- [x] Re-run a small example from a clean environment or documented fresh setup.
- [ ] Record the walkthrough using the actual implementation and saved results.
- [x] Verify video playback and that report tables match recorded result files.
- [x] Package code, configurations, and permitted sample outputs; exclude credentials, local environments, caches, and unnecessary large model/data files.
- [ ] Review the submission checklist and upload the required materials through the course channel when the user is ready.

**Done when:** all required deliverables are complete, inspectable, reproducible to the stated extent, and submitted by the confirmed cutoff.

## 5. Experiment register

Seven core BM25 comparisons and twelve GPU validation comparisons are complete. Both full neural indexes and the frozen GPU final follow-up are complete. Consult `runs/` for completed results; configuration files alone are not results. “Sparse” means the single feasible learned sparse model selected at M0/M4. If we fall back, label the model explicitly rather than retaining the sparse label.

| ID | Configuration | Primary comparison | Priority | Status |
|---|---|---|---|---|
| E00 | BM25, latest question | Starting point | Required | Complete: validation nDCG@5 0.22495 |
| E01 | Same BM25, prior user questions + latest | E00: raw history | Required | Complete: validation nDCG@5 0.18417 |
| E02 | Same BM25, conservative rewrite | E00/E01: rewriting | Required if rewriter feasible | Complete: validation nDCG@5 0.21136 |
| E03 | Sparse, latest question | E00: stronger retriever | Core target | Complete: GPU validation nDCG@5 0.3990 |
| E04 | Same sparse, conservative rewrite | E03: rewrite effect | Core target | Complete: GPU validation nDCG@5 0.3611 |
| E05 | Same sparse, original+rewrite searches, flat RRF | E04: preserve original wording | Core target | Complete: GPU validation nDCG@5 0.4107 |
| E06 | E05 candidates, pure cross-encoder ordering | E05: reranking | Core target | Complete: GPU validation nDCG@5 0.3795 |
| E07 | Exact same E06 candidates/scores, retrieval-rank + reranker-rank fusion | E06: preserve retrieval signal | Core target | Complete: GPU validation nDCG@5 0.4121 |
| E08 | Best fixed query views, sparse+dense, same final pool/reranker | Best sparse-only: dense complementarity | Stretch | Complete: GPU validation nDCG@5 0.4479 |
| E09 | Same views, sparse+ColBERT, same final pool/reranker | Best sparse-only and E08: late interaction | Stretch | Deferred full-index experiment after CPU feasibility gate |
| E10 | Add one keyword-focused view to best feasible system | Two versus three views | Stretch | Complete: GPU validation nDCG@5 0.3909 |
| E11 | Same system, selected history instead of full prefix | History filtering | Stretch | Planned |
| E12 | More views with flat versus nested RRF | Fusion structure at identical views and budget | Later | Planned |
| E13 | Distinct sparse/dense rewrites versus generic views | Retriever-specific query formulation | Later | Planned |

Only perform E12/E13 if their dependencies have produced a credible benefit. Preserve the original question branch in all relevant comparisons.

For the tight deadline, the minimum useful completion is E00/E01 plus E02 when possible, and one fusion/reranking improvement. The main target is E00–E07. E08/E09 are explicit modern-method opportunities, not hidden prerequisites for a valid submission.

## 6. Development splits and selection discipline

- Initial design: approximately 80% of MTRAG conversations for development and 20% for validation, seed 42, with domain representation preserved where possible. Record actual conversation/task counts after loading the data.
- Keep all turns from a conversation in one split.
- Use a small development-only sample for timing, prompt debugging, and smoke checks.
- Avoid repeated selection of prompts and thresholds on the same validation cases. Record each validation comparison and keep the candidate list small.
- Confirm the best two or three candidates on the full fixed validation split.
- If resources and time permit, check the leading comparison on a second grouped split or grouped folds; do not promise full five-fold neural experiments under a two-day deadline.
- Record any public task-fine-tuned checkpoint's training membership. A checkpoint trained on all MTRAG is not clean for our internal MTRAG holdout; use it only in a clearly labeled separate arm or choose a different model.
- Distinguish data used for prompt examples, training, parameter selection, validation reporting, and final scoring.

Use only the assigned document collection. Freeze final choices before test scoring. Keep reference rewrites as an optional separately labeled diagnostic baseline.

## 7. What we measure and how we decide

### Required measurements

- Primary: nDCG@5.
- Secondary: Recall@5 and Recall@10 when supported by the evaluator.
- Candidate diagnostic: Recall@50/100 when the output depth allows it.
- Per-domain results and query coverage.
- Index build time and disk size.
- Rewrite, search, and rerank latency separately.
- LLM calls/tokens when available, plus peak memory where measurable.
- For final candidate systems, median and p95 query time after a warm-up; record cache conditions.

If the official script omits @10, add a clearly documented supplemental calculation using the same underlying metric conventions; do not relabel it as historical official output.

### Fair comparisons

Separate two questions:

1. Does the new method produce better results with its chosen resources?
2. Does it still help at a matched search/candidate/reranking budget?

For multiple query views, compare against one view with a comparable retrieved-candidate allowance. For extra retrievers, use the same final unique candidate cap and reranker. Record raw and deduplicated candidate counts; equal raw counts alone do not establish equal compute.

### Keep/reject rules

These are project decisions, not universal findings from the literature:

- Keep a component when it improves development ranking reliably and its cost is acceptable.
- Prefer the simpler system when differences are tiny, inconsistent, or within uncertainty.
- Use +0.005 absolute nDCG@5 as a flag for further inspection, not an automatic significance threshold or a guarantee of meaningful gain.
- Review domain and context-dependence regressions. A small domain drop is not an automatic veto; explain tradeoffs and avoid severe unexplained degradation.
- A reranker can justify itself through better ordering without improving candidate recall.
- A new retrieval branch should contribute useful candidates or a better final ranking; high deep recall alone does not establish better top-five quality.
- Record negative outcomes rather than deleting unsuccessful experiments.

For leading comparisons, calculate paired differences and conversation-clustered bootstrap uncertainty. With a small validation set, avoid overclaiming significance. Published aggregate winner scores alone do not support paired significance testing against our system.

## 8. Error analysis that guides the next experiment

Inspect a domain-balanced set of approximately 20–30 validation examples, including improvements and regressions. This is a proposed practical sample, not a representative statistical study.

Classify:

1. The latest question needed history but was not resolved.
2. The rewrite carried the wrong entity or an old topic.
3. The rewrite dropped a numerical, negative, temporal, or technical constraint.
4. The rewrite added unsupported information.
5. Relevant history was omitted or irrelevant history dominated.
6. The retriever never found a relevant candidate.
7. The candidate existed but fusion or reranking pushed it down.
8. Collection, identifier, or scoring errors affected the result.
9. Relevance judgments appeared incomplete or ambiguous.

Use the diagnosis to choose the next component:

| Observed problem | Next useful experiment |
|---|---|
| Relevant passages absent from the pool | Alternative retrieval channel or query view |
| Relevant passages retrieved but ranked poorly | Reranker/input/fusion comparison |
| Wrong references or topic carryover | Rewriter prompt or history selection |
| Original exact terms disappear | Original-query branch or keyword view |
| No improvement after more searches | Drop the branch or inspect fusion noise |
| Strange zeros or inconsistent coverage | Stop model work and inspect routing/IDs/scorer |

## 9. Schedule and scope control

**Planning assumption:** the November 25 deadline in the proposal instructions remains current. All dates below use Chicago time. Exact submission time is unconfirmed, so do not interpret November 25 as permission to work until midnight.

Active-work ranges below are estimates; indexing/download time depends on hardware and access.

| Work block | Target | Rough active effort | Required outcome |
|---|---|---|---|
| November 23, remaining evening | M0 and begin M1 | 1–2 hours | Requirements/resource notes, environment, official data access |
| November 24, first work block | Finish M1/M2 and E00 | 2–3 hours | Trustworthy score and valid baseline output |
| November 24, next work block | E01/E02 and sparse feasibility | 2–3 hours | First results table; indexing decision |
| November 24, next work block | M4, strongest feasible candidate | 3–5 hours plus encoding | Core comparisons and a saved usable system |
| November 24, remaining capacity | At most one M5 extension | 1–3 hours plus indexing | Measured improvement or explicit negative result |
| November 25, before final cutoff | M6/M7 | Reserve 4–6 hours | Frozen results, report, walkthrough, package, submission buffer |

The schedule is aggressive. If indexing or model setup consumes the available block, reduce scope immediately rather than borrowing the report/video buffer.

**Freeze rule:** stop optional architecture changes by the end of November 24, or earlier if needed to preserve at least 4–6 hours before the actual deadline. No new model family on submission day.

### Fallback ladder

1. Full feasible target: sparse + rewrite + fusion + reranking, with one tested modern extension.
2. Core: sparse + rewrite + fusion + reranking.
3. Reduced: BM25 + rewrite + fusion + small reranker.
4. Minimum research result: BM25 latest versus raw history, plus generated rewriting if feasible, with honest error analysis.

The fallback preserves a valid experiment; it does not justify inventing results or presenting incomplete experiments as completed.

### Checkpoints

After each milestone, update this file with:

- Completed task IDs and output locations.
- Actual scores and runtime, or explicit failure.
- Current strongest validated configuration.
- The biggest unresolved uncertainty.
- The next experiment and why it is worth doing.
- Remaining deliverable work and schedule risk.

Checkpoint actions are part of implementation sessions; no scheduled automation or reminders have been created.

## 10. Planned repository organization and reproducibility

Create these during implementation as needed; this section does not mean the folders already exist.

| Location | Purpose |
|---|---|
| PROJECT_PLAN.md | This plan, status, and decisions |
| README.md | Reproduction instructions and project explanation |
| configs/ | Experiment settings, model revisions, and collection mappings |
| src/ | Data loading, query preparation, retrieval, fusion, ranking, evaluation |
| scripts/ | Download/validate/index/run/report entry points |
| data/ | Pinned local raw/processed data and manifests |
| splits/ | Saved conversation IDs for development/validation |
| indexes/ | Reusable per-collection retrieval indexes |
| cache/ | Versioned rewrite and model-output caches |
| runs/ | Immutable run configurations, predictions, metrics, logs |
| analysis/ | Error examples, comparisons, tables, plots |
| report/ | Final report source, references, and exports |
| walkthrough/ | Script and final video |

A run must retain its ID, code revision or source hash, dataset revision, split, model/tokenizer revisions, prompt hash, cache identity, retrieval depth, fusion weights, reranker settings, seed/decoding settings, query coverage, and metrics.

A cached rewrite must be keyed by conversation-prefix content and model/prompt settings, not merely the task ID. Reuse caches for fair reranker/retriever comparisons without regeneration.

Keep secrets out of source, configs, logs, and packages. No commercial accounts, paid GPU sessions, or paid APIs are activated by this plan. Any future paid option needs an explicit resource decision.

## 11. Final report and walkthrough outline

Adapt this outline to the actual final rubric once obtained.

### Report

1. Task, motivation, and research question.
2. Relevant work and the evidence behind selected components.
3. Data, conversation-level splits, model provenance, and leakage controls.
4. Implementation: query handling, indexing, fusion, reranking, and output.
5. Evaluation protocol, scored-query counts, and comparability limits.
6. Results and independent ablations.
7. Error examples and resource tradeoffs.
8. Limitations, negative findings, and future work.
9. References and any required assistance/reuse disclosure.

### Walkthrough

- Explain the question the project tests.
- Show the data/task format.
- Trace one real question through the implemented pipeline.
- Show how configurations select the baseline and stronger methods.
- Run a small deterministic example.
- Show actual saved evaluation output and explain the main results.
- Explain a failure case and one implementation tradeoff.

Do not script a video around components we did not implement. Ensure its duration and file format match the final instructions.

## 12. Current status and next action

| Milestone | Status | Evidence |
|---|---|---|
| Research synthesis and plan | Complete for implementation scope | Research map and source links; 50-primary-paper audit not verified |
| Final-project brief | Pending | Proposal instructions only; exact final rubric unavailable |
| Hardware/model feasibility | Complete | `analysis/feasibility_*.json`, `analysis/model_inventory.json` |
| Data audit | Complete | 366,479 passages; pinned source and checksums in `analysis/data_manifest.json` |
| Evaluation | Complete | Official format checks, metric parity, automated tests |
| Baselines and core pipeline | Complete | Seven validation runs, rewrites, fusion, reranking and rank blending |
| Learned sparse/dense extension | Complete | Full 366,479-passage MPS indexes and twelve verified validation runs |
| Final evaluation | Complete | Three original CPU runs and three frozen GPU follow-up runs; all 507 inputs covered |
| Report and walkthrough | Generated; course-format review pending | `report/`, `walkthrough/`; automated review aid is separate from student presentation |
| Course submission | Pending | No course upload has been performed |

**Next action:** Review report/FINAL_PROJECT_REPORT.pdf and the walkthrough against the final course brief. Preserve the completed experiment checkpoints.

### Decision log

| Decision | Rationale | Status |
|---|---|---|
| Keep Subtask A and retrieval-only scope | Matches proposal and measurable task | Accepted |
| Establish baselines and scorer before complex models | Protects the validity of every later result | Accepted |
| Preserve original question and passage IDs | Addresses rewrite and evaluation failure modes | Accepted |
| Compare sparse-only, dense complementarity, and ColBERT conditionally | Reports disagree; experiment can settle the useful part | Accepted as staged experiments |
| Compare pure reranking with rank-preserving fusion | Avoids assuming the reranker always improves ordering | Accepted |
| Defer RL/fine-tuning/agent loops from the core schedule | Resource and time requirements not established | Accepted |
| Core model names and pool sizes | Qwen2.5-0.5B-Instruct, MiniLM-L6, pool 100; selected by fixed validation comparisons | Frozen |
| Exact winner-comparison protocol | Requires audit and known-score artifact checks | Open |

## Sources for this plan

- [Submitted proposal](<sandbox:/Users/neerajvp/Documents/CS421 NLP/NLP/cs421-proposal.pdf>) and [course proposal instructions](<sandbox:/Users/neerajvp/Documents/CS421 NLP/NLP/proposal.pdf>), both read in this conversation. The deadline statement comes from the latter; the separate final brief has not been located.
- [Initial research review](https://chatgpt.com/library?file_id=libfile_94986ae024608191949a8b1a8d02640b&account_id=personal).
- [Critical literature review and system recommendation](https://chatgpt.com/library?file_id=libfile_85e7eef635248191bca1204bfbc0c22e&account_id=personal).
- [Conversational Passage Retrieval Literature Review.pdf](https://chatgpt.com/library?file_id=libfile_adbad58ce85c8191b9d0393b4db26d3d&account_id=personal), especially recommendation, architecture, and experiment sections.
- [Latest evidence-backed architecture report](https://chatgpt.com/library?file_id=libfile_9aa5b04828c48191a9f97cc260f6c13e&account_id=personal).

These reports contain secondary synthesis and proposed adaptations. Their claims about specific models, primary-paper review counts, scorer behavior, and resource requirements must be checked where they affect an implementation decision. This plan does not claim that an untested architecture is the best system or will exceed the winner's score.



## Implementation checkpoint — November 23, 2026

Repository: https://github.com/Neeraj0704/cs421-conversational-retrieval (private).

- Audited 366,479 original passages, 842 development prefixes, 507 current final inputs. Zero missing development relevance passage IDs.
- Validation: 174 submitted, 161 judged tasks in 22 held-out conversations.
- Official format checks passed for E00/E01; unmodified upstream metric function matches per-query nDCG@1,3,5 and Recall@5.
- 13 meaningful automated tests pass.
- CPU feasibility: SPLADE ~10.8 hours and BGE-small ~3.2 hours estimated full indexing. These are sample estimates. Metal/MPS is not exposed in this session.
- Execute the plan’s BM25 + rewrite + fusion + compact reranker fallback; full learned sparse/dense code remains available and is not reported as a completed full-corpus experiment.
- Final brief, report rubric, walkthrough duration and exact cutoff remain unknown. No submission has been made.

## Latest implementation results

| Run | Dataset | Scored / input | nDCG@5 |
|---|---|---|---|
| B02_bm25_fusion | mtrag | 161 / 174 | 0.24724 |
| B03_bm25_rerank | mtrag | 161 / 174 | 0.34643 |
| B04_bm25_blend | mtrag | 161 / 174 | 0.31378 |
| E00_bm25_latest | mtrag | 161 / 174 | 0.22495 |
| E01_bm25_history | mtrag | 161 / 174 | 0.18417 |
| E02_bm25_rewrite | mtrag | 161 / 174 | 0.21136 |
| FINAL_B02_bm25_fusion | mtragun | 332 / 507 | 0.33063 |
| FINAL_E00_bm25_latest | mtragun | 332 / 507 | 0.28961 |
| FINAL_R00_bm25_latest_rerank | mtragun | 332 / 507 | 0.41608 |
| R00_bm25_latest_rerank | mtrag | 161 / 174 | 0.38980 |

The source/configuration hashes in the freeze manifest are authoritative; its Git SHA identifies the preceding checkpoint. The completed commit also includes the hashed working-tree changes. Final test outcomes are not used to change the selected method.

## Delivery verification

All ten saved runs passed independent metric/coverage checks and the 17-test suite. The five-page PDF tables were inspected. The 179-second automated narrated review video decoded without errors and its results frame was checked. A source ZIP was validated for CRC integrity and contains code, exact query snapshots, report and video while excluding corpus, model, index and environment files. Course-format review and any required student-recorded presentation remain pending the final brief.

## GPU extension checkpoint

The user's normal macOS Terminal passed actual MPS computation and numerical agreement
checks for SPLADE and BGE-small. Saved evidence: `analysis/gpu/20261124T042807Z/`.
Batch-16 sample estimates are approximately 2.6 hours for sparse indexing and 35 minutes
for dense indexing; these remain estimates rather than completed full-index measurements.
The agent's tool process still cannot see Metal, regardless of the reported UI permission mode.

A separate, resumable twelve-configuration development study is prepared under
`experiments/gpu_extension_v1/`. Its launcher is `scripts/start_gpu_study.command`.
It refuses CPU fallback, preserves original CPU/test artifacts and performs no new final-test
evaluation. Twenty-two automated tests pass, including isolation, resume and runtime-restore checks.
Full GPU indexing has not been started by the agent; it requires launch from the working Terminal.

## GPU dense checkpoint

The Terminal-run study completed the full 366,479-passage dense index on MPS in approximately 36 minutes. Three validation runs passed independent metric, format and ID/coverage checks: dense latest 0.3850, dense fusion 0.3966, dense latest + reranker 0.4366 nDCG@5. The earlier CPU validation reference is 0.3898. The best dense paired improvement is +0.0468, with a 95% conversation-bootstrap interval of [+0.0057, +0.0939]. SPLADE indexing is running. These are development results; no additional final-test evaluation has been performed.

## GPU study completion — November 24, 2026

The Terminal-run GPU study finished at 2:02 AM Chicago time, after about 3 hours 19 minutes.
Dense indexing took approximately 36 minutes; SPLADE indexing took 2 hours 36 minutes.
Both indexes cover all 366,479 original passages. All twelve validation runs passed an
independent recalculation of metrics and checks for task coverage, original IDs, format,
configuration hashes and unchanged source.

The best observed validation configuration is E08: SPLADE + dense original/rewrite fusion,
nDCG@5 0.4479, compared with the original CPU validation reference 0.3898. The paired delta
is +0.0581, with a 95% conversation-bootstrap interval [+0.0138, +0.1033]. Selection across
twelve validation configurations still introduces selection optimism. This is a development
result; the new GPU-selected configuration has not been evaluated on the final set.

Evidence: `experiments/gpu_extension_v1/GPU_EXTENSION_RESULTS.md` and
`experiments/gpu_extension_v1/analysis/full_study_verification.json`. The original CPU
final-test results and report are preserved. A separate GPU configuration freeze and
clearly labelled final evaluation are the next evaluation step.

## GPU final follow-up preparation

Frozen the GPU primary (E08 hybrid fusion, validation nDCG@5 0.4479) and family comparators
G01 sparse + reranker (0.4281) and G02 dense + reranker (0.4366). Their exact configurations,
source hashes, models and query snapshot are recorded in the GPU workspace’s separate
`analysis/frozen_selection.json`. This preserves the original CPU freeze.

The public final set has already been used for the earlier CPU systems. New GPU runs are
therefore disclosed follow-up benchmarks. No new GPU final scores were used for selection.
The ready launcher is `scripts/start_gpu_final.command`, with no indexing or rewriting
required. All 26 automated tests pass. GPU final evaluation has not been started by the agent.

## GPU final follow-up completed — November 24, 2026

All three validation-selected final configurations completed in the user’s GPU-enabled Terminal
and passed an independent metric/coverage recheck. The primary E08 hybrid scores 0.4334 mean
nDCG@5, 0.4379 submitted-domain-weighted nDCG@5 and 0.6211 Recall@10. Sparse + reranker scores
0.4199; dense + reranker scores 0.4040 (direct means). Each covers 507 inputs and 332 judged tasks.

Against plain BM25, the hybrid’s paired nDCG@5 delta is +0.1438, 95% conversation-bootstrap
interval [+0.1017, +0.1879]. Against the earlier BM25 reranker the delta is +0.0173 with interval
[-0.0178, +0.0525], so that smaller improvement remains uncertain. Primary selection is unchanged.
The public final set was already used for CPU evaluation; this is a disclosed follow-up, not
a fresh unseen holdout. Historical leaderboard equivalence remains unverified.

The combined report is report/FINAL_PROJECT_REPORT.md and .pdf; the earlier CPU report remains
intact. RESULTS.md, the plain-language guide and automated review walkthrough reflect both stages.
Final course formatting, any required student presentation and course upload remain pending.
