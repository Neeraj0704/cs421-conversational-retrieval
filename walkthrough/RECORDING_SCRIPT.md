# Expanded recording script — explain the functions and their purpose

This is the expanded explanation requested for rehearsal and editing. It is deliberately
longer than the earlier timed script. It is not a five-minute cut; trim it after rehearsing.
The submitted walkthrough still has a five-minute maximum.

Read only the blockquotes. OPEN, SHOW and POINT instructions are screen directions.
Introduce the function by name and explain its job before pointing to its implementation.
You do not need to read line numbers, arguments or expressions aloud.

## Prepare the screen

Open /Users/neerajvp/Documents/CS421 NLP/NLP/implementation in VS Code.
Use an 18–20 point font, visible line numbers and Option+Z word wrap. Command+B hides
the sidebar. Keep this script on another device or print it.
Use Command+P, enter a path:line below, and press Return.
Open the destinations before recording so you are comfortable moving between them.
Fit the full diagram in its image tab. Record microphone audio with your screen.

## 1. Explain what the project does

SHOW: walkthrough/ARCHITECTURE.png

POINT: Follow the diagram from Input to the query preparation box, then both search branches,
the ranking fusion box and the final output. Pause at each box as you explain its role.

> Hi, I'm Neeraj. For this project, I worked on conversational passage retrieval, which is Subtask A of SemEval Task Eight.
>
> The goal is to find relevant passages for the latest question in a conversation. That can be tricky because a follow-up question often depends on something said earlier.
>
> This diagram shows our approach. We keep the original question and use Qwen to create a version that makes sense on its own. We search with both versions using SPLADE and BGE, combine the rankings, and return ten passage IDs.
>
> Let me walk through the functions behind each step, starting with the input.

## 2. Identify the configuration being demonstrated

OPEN: experiments/gpu_extension_v1/configs/FINAL_E08_hybrid_fusion.json:5

POINT: backends, lines 8–11; query_mode, line 12; retrieve_k and pool_size, lines 5–6.
This is configuration data, not a function. Do not call it the search implementation.

> Before getting into the functions, this is the configuration for the selected system. It tells the program to use SPLADE and the dense retriever, with original-and-rewritten queries.
>
> Each individual search retrieves up to a hundred passages, and we also keep a hundred after combining the rankings. BM25 and rerankers were tested separately; they aren't part of this selected configuration.

## 3. tasks: build the retrieval inputs

OPEN: src/convrag/data.py:45

POINT: Start at the tasks function name. Then show the collection check at lines 50–52,
the retained fields at lines 53–57, and return out at line 60.

> The first function I want to show is tasks. Its job is to read the dataset and turn each example into the input our retrieval system needs.
>
> It keeps the conversation, its identifiers, and the assigned collection. It deliberately leaves out gold answers, relevance labels, and supplied reference passages. We want the retriever to work from the question and available conversation, without being given the answer.
>
> This also brings up our first assumption: the correct collection is already provided. For example, a Cloud question searches the Cloud collection. Our system doesn't have to decide which collection it belongs to.

SCREEN NOTE: tasks runs during data preparation. The benchmark runner later loads the prepared
records with select_tasks, defined at data.py line 113. Do not say tasks rereads the original
reference file for every individual search.

## 4. latest and rewrite_messages: identify the question and prepare context

OPEN: src/convrag/queries.py:11

POINT: latest function name, the final-speaker check at lines 12–13, then return on line 14.

> With the inputs prepared, we need to identify the question we're actually answering. That's the job of latest. It checks that the last turn is a user question, then returns its text.
>
> So our second assumption is that the supplied conversation ends at the current user question. The check here stops us from accidentally treating an assistant response as the query.

OPEN: src/convrag/queries.py:24

POINT: rewrite_messages function name; history selection and speaker formatting, lines 25–26;
returned system/user messages, line 27. Briefly scroll up to PROMPT at lines 4–9.

> Finding the latest question is straightforward, but understanding it may require earlier turns. That's where rewrite_messages comes in.
>
> It formats the conversation with speaker labels and combines it with our rewriting instructions. In the selected setup, it uses the full supplied conversation prefix.
>
> The prompt asks the model to resolve missing references while preserving the user's meaning. It also tells the model not to answer the question or introduce facts. This function prepares the messages; the next function actually calls the model.

## 5. generate_rewrites: run Qwen and save its outputs

OPEN: src/convrag/neural.py:141

POINT: generate_rewrites function name. Then move to the selected Transformers branch.

OPEN: src/convrag/neural.py:160

POINT: model loading, line 160; nested infer_many function, lines 161–166; model.generate,
line 165. Then scroll to the rewrite_messages call on line 176 and file write on line 184.

> That happens in generate_rewrites. Its job is to load Qwen, generate standalone questions for the selected examples, and save them for retrieval.
>
> Inside it, the helper infer_many handles a batch of prompts: it converts the messages into model inputs, calls the model, and decodes the generated text.
>
> Here we use greedy generation with a limit of 128 new tokens. The outputs are saved so our retrieval comparisons can reuse exactly the same rewrites, rather than generating new wording for every experiment.
>
> But even with these instructions, a rewrite can misunderstand the conversation. So we don't rely on it alone.

SCREEN NOTE: These are the functions used before the recorded demo to generate the saved
rewrites. Do not imply that showing this code runs generation. The selected model is
Qwen/Qwen2.5-0.5B-Instruct; the alternative MLX branch is not the selected model path.

## 6. query_views: choose the queries sent to retrieval

OPEN: src/convrag/queries.py:33

POINT: query_views function name; original on line 34; saved rewrite on line 38;
fusion return on line 40.

> That brings us to query_views. This function decides which query strings the retrievers will receive.
>
> In our selected fusion mode, it returns the original question and the saved rewrite. If they're exactly identical, it keeps just one copy.
>
> Keeping the original gives us another chance to find useful passages when a rewrite loses an important term or adds the wrong context. We now have our search queries; the next step is representing them numerically.

## 7. Encoder.encode: represent queries and passages

OPEN: src/convrag/neural.py:39

POINT: Encoder class and its initializer, lines 39–49. Introduce the class briefly, then move
to encode. The model names are defined in MODELS at line 14 if you need a backup reference.

OPEN: src/convrag/neural.py:51

POINT: encode function name; tokenization, line 54; SPLADE branch, lines 57–59; dense branch,
lines 68–69. Then show document-only pruning at lines 61–64.

> The Encoder class loads the chosen model and tokenizer. Its encode method is what turns a batch of text into numerical representations. We use it for both passages and queries.
>
> For SPLADE, the output is a sparse set of vocabulary weights. It can give weight to useful related terms as well as terms in the text, which helps with matching different wording.
>
> For BGE, the output is a dense vector of 384 numbers. We normalize that vector and use it to match queries with passages based on meaning.
>
> The reason for combining these approaches is that they provide different matching signals. A passage missed by one retriever may still be found by the other.
>
> There is a practical tradeoff here: encoder inputs are limited to 256 tokens, and SPLADE passage vectors keep 128 terms. That reduces local resource use, but can discard useful information.

SCREEN NOTE: The 256 limit includes tokenizer special tokens and the dense query prefix.
The 128-term pruning applies to SPLADE documents; sparse query vectors are not pruned
this way. The encoder uses pretrained weights; this project does not train those weights.

## 8. build_neural: prepare passage indexes before searching

OPEN: src/convrag/neural.py:71

POINT: build_neural function name; corpus loading, line 77; batched encode call, line 92.
Then scroll to saved vectors at lines 98–102 and saved passage IDs at line 103.

> Before we can search, the passages need their own representations too. That's what build_neural prepares.
>
> It reads the passages from each collection, encodes them in batches, and saves the vectors alongside their original passage IDs. This indexing step is done beforehand, so we don't have to encode the entire collection for every question.
>
> Once those indexes are available, searching becomes a query-to-passage comparison.

EDITING NOTE: This is additional explanation for the expanded version. You can shorten it
to one sentence while showing search if you need to reduce the final recording.

## 9. NeuralIndex.search: retrieve candidates from a stored index

OPEN: src/convrag/neural.py:111

POINT: NeuralIndex class and initializer; metadata/IDs on lines 113–115 and stored matrix
on line 117. Then point to the search method at lines 118–122.

> NeuralIndex loads those saved vectors and passage IDs. Its search method takes one query and returns the highest-scoring candidate passages.
>
> It first calls encode for the query. It then compares that representation with the stored passage matrix, selects the strongest matches, and maps them back to their original IDs.
>
> The same search method handles either SPLADE or BGE, depending on the index we loaded. Since we search with each query version through both retrievers, we can end up with four ranked lists. We need to turn those into one final ranking.

## 10. rrf: combine the ranked lists

OPEN: src/convrag/retrieval.py:11

POINT: rrf function name; list iteration on line 15; per-list ranks on line 17;
w/(k+rank) on line 19; descending sort on line 20. Stay above the BM25 class.

> That's the purpose of rrf, our reciprocal rank fusion function. It takes the ranked lists and gives each passage a combined score based on its position in those lists.
>
> A passage gets more credit when it appears near the top, and its contributions add up if it appears across multiple lists.
>
> This is useful because the retrievers' raw scores have different scales. Combining ranking positions gives us a common way to bring their results together.
>
> Now that we've seen the individual functions, I'll show where the program connects them.

## 11. run: connect the steps and produce predictions

OPEN: src/convrag/cli.py:16

POINT: run function name; configuration loading, line 18. Scroll to prepared-input loading,
line 34, saved-rewrite loading at lines 35–38, and domain grouping on lines 40–41.

> The run function is the main coordinator. It reads the experiment configuration, loads the prepared questions and saved rewrites, and groups questions by collection.
>
> It then loads the retrievers selected in that configuration. That is how the same code supports different experiments while keeping this particular run focused on SPLADE and BGE.

OPEN: src/convrag/cli.py:58

POINT: query_views, line 58; nested index/query loops, lines 60–61; rrf and pool limit,
lines 62–63. Scroll to top=ranked[:10], line 74, and document_id serialization, lines 76–77.

> For each question, it calls query_views, sends each query to each index's search method, and passes the resulting lists to rrf.
>
> After fusion, it keeps a hundred candidates and takes the first ten as the final prediction. Those results use the original passage IDs, so the evaluator can match them to the relevance judgments.
>
> There is an optional reranker branch in this function, but our selected configuration skips it. The prediction here comes directly from the fused ranking.
>
> So that's the full path from a conversation to the ten returned passages. Let me show what one of those outputs looks like.

SCREEN NOTE: run also validates and evaluates the predictions at lines 87–88. Results
are written later in the function. The output score field at line 77 encodes final ordering;
it is not a probability or the original neural similarity score.

## 12. Show a genuine saved output

OPEN: walkthrough/RECORDING_EXAMPLE.json:1

POINT: saved-output label, line 2; latest_question, line 4; query_views, lines 6–7;
passage_excerpt, line 12. Wrap the text so the first sentence is readable.

> This is a saved validation example asking, “What are worker nodes?” The first retrieved passage explains that worker nodes carry the deployments and services of an application.
>
> You can see the question, the two query versions, and the passage returned at rank one.
>
> In this case, the rewrite only changes capitalization, so I wouldn't present it as a successful reference resolution. It's an example of the actual retrieval output, and I'm showing a saved result rather than running a new search.

## 13. Explain the overall result and close

OPEN: experiments/gpu_extension_v1/runs/FINAL_E08_hybrid_fusion/metrics.json:64

POINT: mean_over_scored_queries on line 64, then ndcg_cut_5 on line 71. Use this mean,
not the separate weighted aggregate below it.

> Of course, one example isn't enough to judge the whole system. Across the final judged questions, the mean nDCG at five was point four three three four.
>
> That metric rewards putting relevant passages near the top of the ranking. It isn't a percentage of correctly answered questions.
>
> Across development, we compared nineteen configurations. The selected hybrid improved clearly over our plain BM25 baseline, while its smaller gain over BM25 with reranking remained uncertain.
>
> The main lesson is that combining retrieval signals helps, but a rewrite can still introduce the wrong context. Improving rewrite reliability would be my next step. Thank you.

## Function map for rehearsal

| Function or method | Takes | Produces or does |
|---|---|---|
| tasks | Upstream dataset location and dataset name | A list of whitelisted retrieval-input records |
| latest | One conversation task | The current user question, after checking the final speaker |
| rewrite_messages | A task and history mode | Model messages containing the prompt and formatted conversation |
| generate_rewrites | Project location and generation settings | Generates and writes cached rewrite records |
| infer_many, inside generate_rewrites | A batch of model messages | Decoded generated rewrite strings |
| query_views | A task, query mode and saved rewrites | Original/rewrite query strings, with exact duplicates removed |
| Encoder.encode | A batch of texts and a query/document flag | Sparse SPLADE or normalized dense BGE representations |
| build_neural | Corpus location and index settings | Saved passage vectors, original IDs and index metadata |
| NeuralIndex.search | One query and a candidate limit | Ranked passage-ID and similarity-score pairs |
| rrf | Ranked lists, optional weights and rank constant | A merged ranking with combined reciprocal-rank scores |
| run | Experiment arguments/configuration | Coordinates retrieval, validates/evaluates and writes run artifacts |

## Accuracy and editing notes

- This expanded version is for understanding and editing. Do not label it as a five-minute
  script. When trimming, retain the architecture, representation explanation, two explicitly
  named assumptions and prediction-output code.
- tasks prepares source inputs. select_tasks loads prepared inputs for a run and applies
  the conversation split when needed. They do not have the same role.
- Rewriting and corpus indexing happen before the retrieval run in these experiments.
  Showing code does not perform inference. No live command is required by the assignment.
- The selected E08 method has no BM25 component or cross-encoder reranker. Code for
  other experiments exists in the same files, but should not be described as selected behavior.
- The saved example is from validation. The final mean is over 332 judged questions out of
  507 inputs. Historical leaderboard equivalence is unverified. The GPU final study is a
  disclosed follow-up on a previously inspected public set, not a fresh unseen holdout.
- See DEMO_VERIFICATION.md for provenance checks. No retrieval source code was changed.
- Rehearse the shortened version, verify readable code and sound, and keep the submitted
  recording within five minutes.
