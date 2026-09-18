import json

import pytest

from prompt_optimiser import VLLM, Example, FitResult, Prompt, optimize
from prompt_optimiser.native import DSPy, TextGrad
from prompt_optimiser.tracking import MLflowTracker


def data():
    return {
        "train_data": [Example("training", "yes")],
        "validation_data": [Example("validation", "yes")],
        "test_data": [Example("test-secret", "yes")],
    }


def test_native_mipro_predictor_is_saved_and_reloaded(tmp_path):
    dspy = pytest.importorskip("dspy")
    pytest.importorskip("optuna")
    from dspy.utils import DummyLM

    class Target(DummyLM):
        def __init__(self):
            super().__init__([])

        def forward(self, prompt=None, messages=None, **kwargs):
            answer = "yes" if "improved" in messages[0]["content"] else "no"
            self.answers = iter([{"answer": answer}] * kwargs.get("n", 1))
            return super().forward(prompt, messages, **kwargs)

    proposer = DummyLM(
        [
            {"observations": "Return yes."},
            {"summary": "Every reference is yes."},
            {"proposed_instruction": "improved instruction to return yes"},
            {"proposed_instruction": "improved instruction to return yes"},
        ]
    )
    target = Target()
    result = optimize(
        problem="Return yes",
        seed_prompt="seed",
        model=target,
        backend=DSPy(
            optimizer="MIPROv2",
            optimizer_model=proposer,
            optimizer_kwargs={"auto": None, "num_candidates": 2, "max_errors": 1},
            compile_kwargs={
                "num_trials": 5,
                "minibatch": False,
                "program_aware_proposer": False,
                "data_aware_proposer": True,
                "view_data_batch_size": 1,
                "tip_aware_proposer": False,
            },
        ),
        output_dir=tmp_path / "dspy",
        **data(),
    )
    assert result.scores["baseline"]["test"]["score"] == 0
    assert result.scores["final"]["test"]["score"] == 1
    assert result.predict(["new"]) == ["yes"]
    assert len(result.history) >= 2
    restored = dspy.Predict("text -> answer")
    restored.load(result.output_dir / "best/program.json")
    with dspy.context(lm=target):
        assert restored(text="new").answer == result.predict(["new"])[0]
    assert "test-secret" not in json.dumps(proposer.history)


def test_native_textgrad_keeps_gradient_loss_internal(tmp_path):
    pytest.importorskip("textgrad")
    from textgrad.engine import EngineLM

    class Engine(EngineLM):
        model_string = "test-engine"

        def generate(self, prompt, system_prompt=None, **kwargs):
            return "yes" if system_prompt == "improved" else "no"

        def __call__(self, *args, **kwargs):
            return self.generate(*args, **kwargs)

    class Critic(Engine):
        def generate(self, prompt, system_prompt=None, **kwargs):
            assert "test-secret" not in prompt
            return "<IMPROVED_VARIABLE>improved</IMPROVED_VARIABLE>"

    result = optimize(
        problem="Return yes",
        seed_prompt="seed",
        model=Engine(),
        backend=TextGrad(steps=1, optimizer_model=Critic()),
        output_dir=tmp_path / "textgrad",
        **data(),
    )
    assert result.scores["final"]["test"]["score"] == 1
    assert result.predict(["new"]) == ["yes"]
    assert (result.output_dir / "best/prompt.txt").read_text() == result.prompt


def test_independent_backend_has_no_test_data_and_native_artifacts_reach_mlflow(
    tmp_path, monkeypatch
):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    class ExternalOptimizer:
        def fit(self, **request):
            assert "test" not in request and "test_data" not in request
            assert "test-secret" not in str(request["train"])

            def save(directory):
                directory.mkdir(parents=True)
                (directory / "native.txt").write_text("external optimizer state")

            baseline = Prompt("seed", lambda text: "no", save)
            best = Prompt("selected", lambda text: "yes", save)
            return FitResult(baseline, best)

    uri = (tmp_path / "mlruns").as_uri()
    result = optimize(
        problem="Return yes",
        model="custom model",
        backend=ExternalOptimizer(),
        trackers=[MLflowTracker("native-artifacts", tracking_uri=uri)],
        output_dir=tmp_path / "external",
        **data(),
    )
    assert result.scores["final"]["test"]["score"] == 1
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("native-artifacts")
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert client.list_artifacts(run.info.run_id, "experiment/best")[0].path.endswith("native.txt")


def test_vllm_configuration_does_not_export_key(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    model = VLLM("served-model")
    assert "test-secret-key" not in json.dumps(model.public_config())
    dspy = pytest.importorskip("dspy")
    assert isinstance(model.as_dspy(), dspy.LM)
    assert model.as_dspy().kwargs["api_base"] == "http://localhost:8000/v1"
