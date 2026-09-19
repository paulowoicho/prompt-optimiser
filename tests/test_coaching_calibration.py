"""Coaching calibration preserves datasets, judge settings and uploaded reports."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from prompt_optimiser import VLLM, Example, FitResult, Prompt, optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

stage1 = importlib.import_module("examples.career_coaching.calibrate_judge")
stage2 = importlib.import_module("examples.career_coaching.optimize_with_judge")


def calibration_config():
    questions = [{"question_id": i, "question": f"Career question {i}"} for i in range(6)]
    splits = {name: [i] for i, name in enumerate(stage2.pdm.SPLIT_FRACTIONS)}
    return {
        "metadata": {"questions": questions, "question_splits": splits},
        "model_config": VLLM("calibrated", seed=7, max_tokens=23).public_config(),
    }


def test_response_stage_uses_snapshot_and_saved_judge_settings(tmp_path, monkeypatch):
    config = calibration_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    # No global dataset cache is needed to reproduce the calibrated splits.
    monkeypatch.setattr(stage2.pdm, "CACHE", tmp_path / "absent-cache")
    restored = stage2.load_stage1_config(tmp_path)
    assert restored == config
    model = stage2.saved_judge_model(restored, "calibrated", "http://new-host:8000/v1")
    assert model.seed == 7 and model.max_tokens == 23
    assert model.base_url == "http://new-host:8000/v1"
    with pytest.raises(ValueError, match="must match"):
        stage2.saved_judge_model(restored, "different-model", model.base_url)


@pytest.mark.parametrize(
    "fault", ["overlap", "missing", "duplicate_text", "wrong_id", "no_snapshot"]
)
def test_response_stage_rejects_invalid_calibration_dataset(tmp_path, fault):
    config = calibration_config()
    metadata = config["metadata"]
    if fault == "overlap":
        metadata["question_splits"]["response_test"] = metadata["question_splits"]["judge_train"]
    elif fault == "missing":
        metadata["question_splits"]["response_test"] = []
    elif fault == "duplicate_text":
        metadata["questions"][1]["question"] = "  CAREER   QUESTION 0  "
    elif fault == "wrong_id":
        metadata["questions"][0]["question_id"] = 99
    else:
        del metadata["questions"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        stage2.load_stage1_config(tmp_path)


@pytest.mark.parametrize("keep_seed", [True, False])
def test_calibration_report_is_uploaded_including_when_seed_is_retained(tmp_path, keep_seed):
    class Backend:
        def fit(self, *, seed_prompt, **kwargs):
            baseline = Prompt(seed_prompt, lambda text: "INVALID")
            final = baseline if keep_seed else Prompt("improved", lambda text: "TIE")
            return FitResult(baseline, final)

    uploaded = []

    class Upload:
        def log(self, event):
            if event.kind == "artifact":
                uploaded.append(json.loads((Path(event.data["path"]) / "report.json").read_text()))

        def close(self, status):
            pass

    meta = [{"source": "committee", "kind": "natural"}] * 2
    result = optimize(
        problem="Judge coaching", model="fake", backend=Backend(),
        train_data=[Example("train", "TIE")],
        validation_data=[Example("validation", "TIE")],
        test_data=[Example("forward", "TIE"), Example("backward", "TIE")],
        metric=stage1.label_match, metadata={"question_splits": {}},
        output_dir=tmp_path / "run", trackers=[stage1.ReportArtifacts(meta), Upload()],
    )
    assert len(uploaded) == 1
    report = uploaded[0]
    assert report["judge_test"]["baseline"]["invalid_outputs"] == 2
    assert report["judge_test"]["final"]["invalid_outputs"] == (2 if keep_seed else 0)
    assert report["judge_test_by_source_and_kind"]["final"]["committee"] == (
        0.0 if keep_seed else 1.0
    )
    assert (result.output_dir / "predictions/final-test.jsonl").exists() != keep_seed


def test_response_report_and_verdicts_are_ready_for_upload(tmp_path):
    incumbents = {name: f"seed {name}" for name in ("train", "validation", "test")}
    metric = stage2.PreferenceMetric(lambda text: "TIE", {v: k for k, v in incumbents.items()})

    class Backend:
        def fit(self, *, seed_prompt, **kwargs):
            prompt = Prompt(seed_prompt, lambda text: incumbents[text])
            return FitResult(prompt, prompt)

    uploaded = []

    class Upload:
        def log(self, event):
            if event.kind == "artifact":
                directory = Path(event.data["path"])
                uploaded.append(json.loads((directory / "report.json").read_text()))
                assert (directory / "judge_verdicts.jsonl").read_text().count("\n") == 6

        def close(self, status):
            pass

    optimize(
        problem="Coach", model="fake", backend=Backend(),
        train_data=[Example("train", incumbents["train"])],
        validation_data=[Example("validation", incumbents["validation"])],
        test_data=[Example("test", incumbents["test"])], metric=metric,
        metadata={"judge": "best", "judge_run": "stage1"}, output_dir=tmp_path / "run",
        trackers=[stage2.JudgeArtifacts(metric), Upload()],
    )
    assert len(uploaded) == 1
    assert uploaded[0]["preference_vs_seed"]["final"]["test"] == 0.5
    assert uploaded[0]["judge"] == "best"
