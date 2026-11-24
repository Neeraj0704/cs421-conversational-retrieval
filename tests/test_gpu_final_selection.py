"""Keep finalist selection tied to validation rather than new test outcomes."""

from pathlib import Path
import pytest


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_gpu_final import select_finalists

    return select_finalists


def config(backends, dataset="mtrag"):
    return {"dataset": dataset, "split": "validation", "backends": backends}


def test_selects_overall_and_each_retrieval_family(selector):
    configs = {
        "sparse_a": config(["splade"]),
        "sparse_b": config(["splade"]),
        "dense": config(["dense"]),
        "hybrid": config(["splade", "dense"]),
    }
    primary, winners = selector(
        configs, {"sparse_a": 0.4, "sparse_b": 0.43, "dense": 0.44, "hybrid": 0.45}
    )
    assert primary == "hybrid"
    assert winners == {"sparse": "sparse_b", "dense": "dense", "hybrid": "hybrid"}


def test_rejects_test_dataset_score_during_selection(selector):
    with pytest.raises(ValueError, match="validation only"):
        selector({"test": config(["splade"], "mtragun")}, {"test": 0.99})


def test_requires_all_declared_validation_scores(selector):
    with pytest.raises(ValueError, match="Every declared"):
        selector({"sparse": config(["splade"])}, {})


def test_requires_retrieval_family_comparators(selector):
    with pytest.raises(ValueError, match="Each retrieval family"):
        selector({"sparse": config(["splade"])}, {"sparse": 0.4})
