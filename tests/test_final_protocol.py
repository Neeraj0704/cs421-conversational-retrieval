import argparse
import pytest
from convrag.cli import run
from convrag.io import write_json, fingerprint


def setup(tmp_path, final=True):
    cfg = {
        "id": "FINAL_trial",
        "dataset": "mtragun",
        "split": "all",
        "backends": ["bm25"],
        "query_mode": "latest",
    }
    path = tmp_path / "config.json"
    write_json(path, cfg)
    args = argparse.Namespace(
        root=str(tmp_path),
        upstream="data/upstream",
        config=str(path),
        final=final,
        overwrite=False,
    )
    return cfg, args


def test_final_dataset_requires_explicit_mode(tmp_path):
    cfg, args = setup(tmp_path, False)
    with pytest.raises(ValueError, match="explicit --final"):
        run(args)


def test_final_requires_frozen_development_decision(tmp_path):
    cfg, args = setup(tmp_path)
    with pytest.raises(ValueError, match="Freeze the selected"):
        run(args)


def test_unregistered_final_config_rejected_before_loading_data(tmp_path):
    cfg, args = setup(tmp_path)
    write_json(
        tmp_path / "analysis/frozen_selection.json",
        {"final_configs": {"FINAL_other": "wrong"}, "source_sha256": {}},
    )
    with pytest.raises(ValueError, match="not prespecified"):
        run(args)


def test_source_drift_rejected_before_final_qrels_are_read(tmp_path):
    cfg, args = setup(tmp_path)
    write_json(tmp_path / "src/module.py", {"changed": True})
    write_json(
        tmp_path / "analysis/frozen_selection.json",
        {
            "final_configs": {cfg["id"]: fingerprint(cfg)},
            "source_sha256": {"src/module.py": "incorrect"},
        },
    )
    with pytest.raises(ValueError, match="Source changed after freeze"):
        run(args)
