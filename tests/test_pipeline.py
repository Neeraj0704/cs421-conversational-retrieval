import json
import math
import pytest
from convrag.retrieval import rrf, BM25Index
from convrag.queries import query_views, rewrite_messages, cache_key
from convrag.evaluation import validate_predictions, evaluate, bootstrap_difference
from convrag.data import COLLECTIONS, tasks


def task():
    return {
        "task_id": "c<::>2",
        "conversation_id": "c",
        "domain": "cloud",
        "Collection": COLLECTIONS["cloud"],
        "input": [
            {"speaker": "user", "text": "What is IBM Cloud?"},
            {"speaker": "agent", "text": "A cloud platform."},
            {"speaker": "user", "text": "Does it support Python?"},
        ],
    }


def test_fusion_deduplicates_and_ignores_score_scales():
    out = rrf([[("a", 100), ("a", 99), ("b", 80)], [("b", 0.1), ("c", 0.01)]])
    assert len(out) == 3 and out[0][0] == "b"
    assert rrf([[("a", 999), ("b", -2)]], [1]) == rrf([[("a", 1), ("b", 0)]], [1])


def test_history_and_prompt_only_see_prefix():
    t = task()
    t["targets"] = [{"text": "DO NOT USE THIS GOLD ANSWER"}]
    assert query_views(t, "latest") == ["Does it support Python?"]
    assert "What is IBM Cloud?" in query_views(t, "history")[0]
    assert "DO NOT USE" not in json.dumps(rewrite_messages(t))
    assert cache_key(t, "m", "r", "full") != cache_key(t, "m", "r2", "full")


def test_prediction_contract():
    t = task()
    p = {
        "task_id": t["task_id"],
        "Collection": t["Collection"],
        "contexts": [{"document_id": str(i), "score": 10 - i} for i in range(10)],
    }
    validate_predictions([p], [t], {"cloud": set(map(str, range(10)))})
    p["contexts"][-1]["score"] = float("nan")
    with pytest.raises(ValueError):
        validate_predictions([p], [t])


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_task",
        "duplicate_passage",
        "tie",
        "wrong_collection",
        "missing",
        "wrong_passage",
    ],
)
def test_invalid_outputs_cannot_silently_score(failure):
    t = task()
    p = {
        "task_id": t["task_id"],
        "Collection": t["Collection"],
        "contexts": [{"document_id": str(i), "score": 10 - i} for i in range(10)],
    }
    preds = [p]
    if failure == "duplicate_task":
        preds = [p, p]
    elif failure == "duplicate_passage":
        p["contexts"][1]["document_id"] = "0"
    elif failure == "tie":
        p["contexts"][1]["score"] = 10
    elif failure == "wrong_collection":
        p["Collection"] = "wrong"
    elif failure == "missing":
        preds = []
    elif failure == "wrong_passage":
        p["contexts"][0]["document_id"] = "absent"
    with pytest.raises(ValueError):
        validate_predictions(preds, [t], {"cloud": set(map(str, range(10)))})


def test_evaluation_does_not_drop_empty_or_failed_queries(tmp_path):
    folder = tmp_path / "mtrag-human/retrieval_tasks/cloud/qrels"
    folder.mkdir(parents=True)
    (folder / "dev.tsv").write_text(
        "query_id\tcorpus_id\tscore\nc<::>2\ta\t1\nc<::>3\tb\t1\n"
    )
    one = task()
    two = {**task(), "task_id": "c<::>3"}
    preds = [
        {"task_id": one["task_id"], "contexts": [{"document_id": "a", "score": 1}]},
        {"task_id": two["task_id"], "contexts": []},
    ]
    summary, scored = evaluate(preds, [one, two], tmp_path)
    assert summary["scored"] == 2
    assert summary["mean_over_scored_queries"]["ndcg_cut_5"] == 0.5
    assert scored[two["task_id"]]["ndcg_cut_5"] == 0


def test_bm25_real_index_preserves_passage_ids(tmp_path):
    docs = [
        {"id": "p-offset-0-100", "text": "Python cloud deployment"},
        {"id": "other", "text": "banana bread recipe"},
    ]
    BM25Index.build(tmp_path, docs)
    assert BM25Index(tmp_path).search("cloud Python", 2)[0][0] == "p-offset-0-100"


def test_bootstrap_is_paired_by_conversation():
    a = {"q": {"conversation_id": "c", "ndcg_cut_5": 0.2}}
    b = {"q": {"conversation_id": "c", "ndcg_cut_5": 0.5}}
    result = bootstrap_difference(a, b, samples=100)
    assert result["ci95"] == pytest.approx([0.3, 0.3])
    with pytest.raises(ValueError):
        bootstrap_difference(a, {})


def test_reference_loader_whitelists_gold(tmp_path):
    folder = tmp_path / "mtrag-human/generation_tasks"
    folder.mkdir(parents=True)
    t = task()
    t.update(targets=[{"text": "secret"}], contexts=[{"text": "secret"}])
    (folder / "reference.jsonl").write_text(json.dumps(t) + "\n")
    assert "secret" not in json.dumps(tasks(tmp_path, "mtrag"))
