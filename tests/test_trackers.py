"""Tracker integrations exercised through optimize() with a tiny in-process backend."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from prompt_optimiser import Event
from prompt_optimiser import Example
from prompt_optimiser import ExperimentResult
from prompt_optimiser import FitResult
from prompt_optimiser import Prompt
from prompt_optimiser import optimize
from prompt_optimiser.tracking import JSONLTracker
from prompt_optimiser.tracking import WandbTracker


class TwoPromptBackend:
    """Baseline says no, the selected prompt says yes. No model calls."""

    def fit(self, **request: Any) -> FitResult:
        def save(directory: Path) -> None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "state.txt").write_text("native state")

        return FitResult(
            Prompt("seed", lambda text: "no", save),
            Prompt("improved", lambda text: "yes", save),
        )


def _run(tmp_path: Path, trackers: Any) -> ExperimentResult:
    return optimize(
        problem="Return yes",
        model="fake-model",
        backend=TwoPromptBackend(),
        train_data=[Example("train", "yes")],
        validation_data=[Example("val", "yes")],
        test_data=[Example("test", "yes")],
        output_dir=tmp_path / "run",
        trackers=trackers,
    )


class TestTrackers:
    def test_jsonl_tracker_records_start_finish_and_close(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        result = _run(tmp_path, [JSONLTracker(path)])
        kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
        assert kinds[0] == "start" and kinds[-2:] == ["finish", "close"]
        assert result.scores["final"]["test"]["score"] == 1

    def test_wandb_offline_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("wandb")
        monkeypatch.setenv("WANDB_SILENT", "true")
        _run(tmp_path, [WandbTracker("test", mode="offline", dir=str(tmp_path))])
        reports = list(tmp_path.glob("wandb/offline-run-*/files/result.json"))
        assert len(reports) == 1
        assert json.loads(reports[0].read_text())["scores"]["final"]["test"]["score"] == 1

    def test_core_import_does_not_load_integrations(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import prompt_optimiser; "
                "assert not {'dspy', 'textgrad', 'litellm', 'mlflow', 'wandb'} & set(sys.modules)",
            ],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        assert result.returncode == 0, result.stderr

    def test_default_output_dir_is_fresh_and_status_is_final_before_upload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        seen = []

        class UploadTracker:
            def log(self, event: Event) -> None:
                if event.kind == "artifact":
                    status = json.loads(
                        (tmp_path / event.data["path"] / "status.json").read_text()
                    )
                    seen.append(status["status"])

            def close(self, status: str) -> None:
                seen.append(f"close:{status}")

        kwargs = dict(
            problem="Return yes",
            model="fake-model",
            backend=TwoPromptBackend(),
            train_data=[Example("train", "yes")],
            validation_data=[Example("val", "yes")],
            trackers=[UploadTracker()],
        )
        first = optimize(**kwargs)
        second = optimize(**kwargs)
        assert first.output_dir != second.output_dir
        assert first.output_dir.resolve().parent == tmp_path / "runs"
        assert seen == ["finished", "close:finished", "finished", "close:finished"]
