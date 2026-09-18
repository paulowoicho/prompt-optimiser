"""Behavioural requirements of optimize(): data checks, isolation, failure handling, artifacts."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from prompt_optimiser import Example, FitResult, Prompt, optimize
from prompt_optimiser.data import split_validation


@dataclass
class EchoBackend:
    """Records what fit() received; the selected prompt answers with a fixed label."""

    answer: str = "yes"
    optimizer_kwargs: dict = field(default_factory=dict)
    api_key: str = "hunter2"
    calls: list = field(default_factory=list, repr=False)

    def fit(self, **request):
        self.calls.append(request)

        def save(directory):
            directory.mkdir(parents=True, exist_ok=True)

        baseline = Prompt(request["seed_prompt"], lambda text: "no", save)
        best = Prompt("selected", lambda text: self.answer, save)
        return FitResult(baseline, best)


ROWS = [Example(f"input {i}", "yes") for i in range(10)]


def test_automatic_split_is_reproducible_and_keeps_duplicate_inputs_together():
    rows = tuple(ROWS) + (Example("input 0", "yes"),)
    first = split_validation(rows, 0.2, seed=7)
    second = split_validation(rows, 0.2, seed=7)
    assert first == second
    train, validation = first
    assert len(validation) in (2, 3)
    assert not {row.input for row in train} & {row.input for row in validation}
    duplicates = [row for row in rows if row.input == "input 0"]
    assert all(row in train for row in duplicates) or all(row in validation for row in duplicates)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_data": []},
        {"train_data": ROWS, "validation_data": []},
        {"train_data": ROWS, "validation_data": [ROWS[0]]},
        {"train_data": ROWS[:5], "validation_data": ROWS[5:], "test_data": [ROWS[0]]},
        {"train_data": [ROWS[0]]},
        {"train_data": ROWS, "problem": " "},
        {"train_data": ROWS, "seed_prompt": ""},
    ],
)
def test_bad_data_fails_before_the_backend_runs(tmp_path, kwargs):
    backend = EchoBackend()
    arguments = {"problem": "Return yes", "model": "fake", "backend": backend, **kwargs}
    with pytest.raises(ValueError):
        optimize(output_dir=tmp_path / "run", **arguments)
    assert backend.calls == []


def test_backend_receives_train_and_validation_but_never_test(tmp_path):
    backend = EchoBackend()
    result = optimize(
        problem="Return yes",
        model="fake",
        backend=backend,
        train_data=ROWS[:6],
        validation_data=ROWS[6:8],
        test_data=[Example("test-secret", "yes")],
        output_dir=tmp_path / "run",
    )
    (request,) = backend.calls
    assert "test" not in request and "test_data" not in request
    assert "test-secret" not in json.dumps({k: str(v) for k, v in request.items()})
    assert request["seed_prompt"] == "Return yes"
    assert result.scores["final"]["test"]["score"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_metric_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="finite"):
        optimize(
            problem="Return yes",
            model="fake",
            backend=EchoBackend(),
            train_data=ROWS,
            metric=lambda expected, predicted: value,
            output_dir=tmp_path / "run",
        )


def test_failure_marks_status_failed_and_closes_trackers(tmp_path):
    closed = []
    uploaded = []

    class Tracker:
        def log(self, event):
            if event.kind == "artifact":
                path = tmp_path / event.data["path"]
                uploaded.append(json.loads((path / "status.json").read_text())["status"])

        def close(self, status):
            closed.append(status)

    class Broken:
        def fit(self, **request):
            raise RuntimeError("native training exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        optimize(
            problem="Return yes",
            model="fake",
            backend=Broken(),
            train_data=ROWS,
            output_dir=tmp_path / "run",
            trackers=[Tracker()],
        )
    assert closed == ["failed"]
    assert uploaded == ["failed"], "partial directory is uploaded for diagnosis"
    assert json.loads((tmp_path / "run" / "status.json").read_text()) == {"status": "failed"}


def test_config_keeps_nested_backend_settings_and_redacts_secrets(tmp_path):
    backend = EchoBackend(
        optimizer_kwargs={
            "num_candidates": 3,
            "max_tokens": 512,
            "token_budget": 1000,
            "nested": {"api_key": "k", "hf_token": "t"},
        }
    )
    result = optimize(
        problem="Return yes",
        model="fake",
        backend=backend,
        train_data=ROWS,
        output_dir=tmp_path / "run",
    )
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["backend"] == "EchoBackend"
    assert config["backend_config"]["optimizer_kwargs"]["num_candidates"] == 3
    assert config["backend_config"]["optimizer_kwargs"]["nested"]["api_key"] == "[redacted]"
    assert config["backend_config"]["optimizer_kwargs"]["nested"]["hf_token"] == "[redacted]"
    assert config["backend_config"]["optimizer_kwargs"]["max_tokens"] == 512
    assert config["backend_config"]["optimizer_kwargs"]["token_budget"] == 1000
    assert config["backend_config"]["api_key"] == "[redacted]"
    assert "hunter2" not in json.dumps(config)


def test_per_row_predictions_are_saved_for_every_evaluated_split(tmp_path):
    result = optimize(
        problem="Return yes",
        model="fake",
        backend=EchoBackend(answer="maybe"),
        train_data=ROWS[:6],
        validation_data=ROWS[6:8],
        test_data=ROWS[8:],
        output_dir=tmp_path / "run",
    )
    files = sorted(p.name for p in (result.output_dir / "predictions").iterdir())
    assert files == [
        "baseline-test.jsonl",
        "baseline-train.jsonl",
        "baseline-validation.jsonl",
        "final-test.jsonl",
        "final-train.jsonl",
        "final-validation.jsonl",
    ]
    saved = (result.output_dir / "predictions" / "final-test.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in saved]
    assert rows == [
        {"input": "input 8", "target": "yes", "prediction": "maybe", "score": 0.0},
        {"input": "input 9", "target": "yes", "prediction": "maybe", "score": 0.0},
    ]
    assert result.scores["final"]["test"] == {"score": 0.0, "n_examples": 2}


def test_training_failure_survives_a_failing_diagnostics_upload(tmp_path):
    closed = []

    class BrokenUploader:
        def log(self, event):
            if event.kind == "artifact":
                raise ConnectionError("tracker offline")

        def close(self, status):
            closed.append(("uploader", status))

    class Quiet:
        def log(self, event):
            pass

        def close(self, status):
            closed.append(("quiet", status))

    class Broken:
        def fit(self, **request):
            raise RuntimeError("native training exploded")

    with pytest.warns(RuntimeWarning, match="tracker offline"), pytest.raises(
        RuntimeError, match="exploded"
    ):
        optimize(
            problem="Return yes",
            model="fake",
            backend=Broken(),
            train_data=ROWS,
            output_dir=tmp_path / "run",
            trackers=[BrokenUploader(), Quiet()],
        )
    assert closed == [("uploader", "failed"), ("quiet", "failed")]
    assert json.loads((tmp_path / "run" / "status.json").read_text()) == {"status": "failed"}


def test_a_new_backend_needs_only_fit_and_prompts_without_save(tmp_path):
    """The contract for a third-party optimiser: one method, two Prompts, nothing else."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "custom_backend", Path(__file__).resolve().parents[1] / "examples" / "custom_backend.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)

    def fake_model(system_prompt, text):
        if system_prompt.startswith("Rewrite"):
            return "Always answer yes."
        return "yes" if system_prompt == "Always answer yes." else "no"

    result = optimize(
        problem="Return yes",
        model=fake_model,
        backend=module.RewriteSearch(rewrites=2),
        train_data=ROWS[:6],
        validation_data=ROWS[6:8],
        test_data=ROWS[8:],
        output_dir=tmp_path / "run",
    )
    assert result.prompt == "Always answer yes."
    assert result.scores["baseline"]["test"]["score"] == 0
    assert result.scores["final"]["test"]["score"] == 1
    assert (result.output_dir / "best" / "prompt.txt").read_text() == "Always answer yes."
    assert len(result.history) == 3
    config = json.loads((result.output_dir / "config.json").read_text())
    assert config["backend_config"] == {"rewrites": 2}
