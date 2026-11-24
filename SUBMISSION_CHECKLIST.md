# Submission checkpoint

Completed and saved:

- Working retrieval implementation, configuration files and setup instructions.
- Seven CPU and twelve GPU validation comparisons; three CPU and three GPU final comparisons.
- Predictions for all 507 final inputs; 332 questions have relevance judgments.
- Official format checks, metric parity, 26 passing tests and independent saved-result verification.
- A separate clean-environment BM25 reproduction with zero metric differences.
- Combined CPU/GPU report source and PDF, preserved CPU report, plain-language guide and updated tracker.
- Automated narrated walkthrough, transcript and recording guide.
- Exact generated query snapshots and model/data provenance.

Review before course submission:

- Confirm the separate final-project brief: exact deadline, page limit, required sections,
  team/author names, video length, packaging and assistance-disclosure requirements.
- Read and understand the code, results, negative outcomes and limitations.
- Check whether the course requires your own recorded explanation; the supplied video
  is explicitly an automated review aid with a synthetic voice.
- Upload the required materials through the course submission channel.

Full SPLADE/dense experiments are complete. The validation-selected hybrid scores 0.4334
mean final nDCG@5 (0.4379 weighted). Its gain over plain BM25 is clear; the smaller
gain over BM25 + reranker has a paired interval including zero. GPU final scores are disclosed
follow-up results on a previously used public set. ColBERT and training remain future work.
Historical leaderboard equivalence is unverified. No course submission has been made.
