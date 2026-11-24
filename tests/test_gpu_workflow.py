"""Check the long-running study's isolation and resume guards without a GPU."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "gpu_study", SOURCE / "scripts/run_gpu_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.fixture
def project(tmp_path):
    for name in ["src", "configs", "splits"]:
        shutil.copytree(
            SOURCE / name,
            tmp_path / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
    (tmp_path / "analysis").mkdir()
    for name in ["data_manifest.json", "model_inventory.json", "frozen_selection.json"]:
        shutil.copy2(SOURCE / "analysis" / name, tmp_path / "analysis" / name)
    (tmp_path / "artifacts/queries").mkdir(parents=True)
    shutil.copy2(
        SOURCE / "artifacts/queries/mtrag.jsonl",
        tmp_path / "artifacts/queries/mtrag.jsonl",
    )
    for name in ["data/upstream", "data/processed", "indexes", "cache/huggingface"]:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_study_prepare_is_repeatable_and_preserves_core(project):
    before = (project / "analysis/frozen_selection.json").read_bytes()
    workspace, manifest = study.prepare(project)
    again, repeated = study.prepare(project)
    assert workspace == again and manifest == repeated
    assert len(manifest["config_sha256"]) == 12
    assert not manifest["final_test_evaluation_enabled"]
    assert not (project / "runs").exists()
    assert (project / "analysis/frozen_selection.json").read_bytes() == before
    assert (workspace / "indexes").resolve() == project / "indexes"


def test_resume_rejects_changed_experiment(project):
    workspace, manifest = study.prepare(project)
    p = workspace / "configs/G01_splade_latest_rerank.json"
    cfg = json.loads(p.read_text())
    cfg["pool_size"] = 5
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="configuration changed"):
        study.prepare(project)


def test_prepare_restores_runtime_files_absent_from_fresh_clone(project):
    workspace, manifest = study.prepare(project)
    shutil.rmtree(workspace / "src")
    shutil.rmtree(workspace / "cache/rewrites")
    for relative in [
        "indexes",
        "splits",
        "data/upstream",
        "data/processed",
        "cache/huggingface",
    ]:
        (workspace / relative).unlink()
    restored, again = study.prepare(project)
    assert restored == workspace and again == manifest
    assert (workspace / "src/convrag/neural.py").is_file()
    assert (workspace / "cache/rewrites/mtrag.jsonl").read_bytes() == (
        project / "artifacts/queries/mtrag.jsonl"
    ).read_bytes()


def test_prepare_rejects_changed_original_source(project):
    p = project / "src/convrag/queries.py"
    p.write_text(p.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="Original study source has changed"):
        study.prepare(project)


def test_gpu_stage_stops_before_cli_when_gpu_unavailable():
    # In a CPU-only test installation, probe handles an absent torch package as well.
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("This check targets a CPU-only process")
    result = subprocess.run(
        [sys.executable, str(SOURCE / "scripts/run_gpu_stage.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No CPU fallback stage was started" in result.stdout + result.stderr
    assert "Conversational passage retrieval" not in result.stdout
