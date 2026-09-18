"""Exercise real alternative optimisers with deterministic local model fixtures."""

import json

import pytest

from prompt_optimiser import Example, optimize
from prompt_optimiser.native import DSPy, TextGrad


def splits():
    return {
        "train_data": [Example("training-only", "yes")],
        "validation_data": [Example("validation-only", "yes")],
        "test_data": [Example("test-secret", "yes")],
    }


def target():
    pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    class Target(DummyLM):
        def __init__(self):
            super().__init__([])

        def forward(self, prompt=None, messages=None, **kwargs):
            improved = "improved" in messages[0]["content"] or len(messages) > 2
            answer = {"answer": "yes" if improved else "no"}
            if "reasoning" in messages[0]["content"]:
                answer["reasoning"] = "Use the labelled demonstration."
            self.answers = iter([answer] * kwargs.get("n", 1))
            return super().forward(prompt, messages, **kwargs)

    return Target()


def test_real_copro_preserves_native_prefix_and_labels_training_history(tmp_path):
    dspy = pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    proposer = DummyLM(
        [
            {
                "proposed_instruction": "improved instructions",
                "proposed_prefix_for_output_field": "The final answer:",
            }
        ]
    )
    result = optimize(
        problem="Answer yes",
        seed_prompt="seed",
        model=target(),
        backend=DSPy(
            optimizer=dspy.COPRO,
            optimizer_model=proposer,
            optimizer_kwargs={"breadth": 2, "depth": 1},
            compile_kwargs={"eval_kwargs": {"num_threads": 1, "display_progress": False}},
        ),
        output_dir=tmp_path / "copro",
        **splits(),
    )
    assert result.scores["baseline"]["test"]["score"] == 0
    assert result.scores["final"]["test"]["score"] == 1
    training_events = [e for e in result.history if e["data"].get("evaluated_split") == "train"]
    assert training_events
    assert all("train/score" in e["metrics"] for e in training_events)
    assert all("validation/score" not in e["metrics"] for e in training_events)
    assert "The final answer:" in (result.output_dir / "best/program.json").read_text()
    assert "test-secret" not in json.dumps(proposer.history)


@pytest.mark.parametrize("program_name", ["Predict", "ChainOfThought"])
def test_labeled_fewshot_needs_no_metric_constructor_or_validation_compile_kwarg(
    tmp_path,
    program_name,
):
    dspy = pytest.importorskip("dspy")
    result = optimize(
        problem="Answer yes",
        seed_prompt="seed",
        model=target(),
        backend=DSPy(
            optimizer="LabeledFewShot",
            optimizer_kwargs={"k": 1},
            program=getattr(dspy, program_name),
        ),
        output_dir=tmp_path / program_name,
        **splits(),
    )
    assert result.scores["final"]["test"]["score"] == 1
    assert "training-only" in (result.output_dir / "best/messages.json").read_text()
    assert result.predict(["unseen"]) == ["yes"]


def test_preconfigured_optimizer_and_program_are_used_without_rebuilding(tmp_path):
    dspy = pytest.importorskip("dspy")
    program = dspy.ChainOfThought(dspy.Signature("text -> answer", instructions="native seed"))
    result = optimize(
        problem="Answer yes",
        model=target(),
        backend=DSPy(optimizer=dspy.LabeledFewShot(k=1), program=program),
        output_dir=tmp_path / "instances",
        **splits(),
    )
    assert result.prompt == "native seed"
    assert result.scores["final"]["test"]["score"] == 1
    assert not program.predictors()[0].demos


def test_native_multistage_module_keeps_all_predictor_state(tmp_path):
    dspy = pytest.importorskip("dspy")

    class TwoStages(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict(dspy.Signature("text -> answer", instructions="first seed"))
            self.second = dspy.Predict(dspy.Signature("text -> answer", instructions="second seed"))

        def forward(self, text):
            return self.second(text=self.first(text=text).answer)

    model = target()
    result = optimize(
        problem="Answer yes",
        model=model,
        backend=DSPy(optimizer="LabeledFewShot", optimizer_kwargs={"k": 1}, program=TwoStages()),
        output_dir=tmp_path / "stages",
        **splits(),
    )
    assert result.scores["final"]["test"]["score"] == 1
    templates = json.loads((result.output_dir / "best/messages.json").read_text())
    assert set(templates) == {"first", "second"}
    restored = TwoStages()
    restored.load(result.output_dir / "best/program.json")
    with dspy.context(lm=model):
        assert restored(text="new").answer == result.predict(["new"])[0]


def engines(malformed=False):
    pytest.importorskip("textgrad")
    from textgrad.engine import EngineLM

    class Engine(EngineLM):
        model_string = "test-engine"

        def generate(self, prompt, system_prompt=None, **kwargs):
            return "yes" if system_prompt == "improved" else "no"

        def __call__(self, *args, **kwargs):
            return self.generate(*args, **kwargs)

    class Critic(Engine):
        def __init__(self):
            self.seen = []

        def generate(self, prompt, system_prompt=None, **kwargs):
            assert "test-secret" not in prompt
            assert "validation-only" not in prompt
            self.seen.append(prompt)
            return (
                "malformed update"
                if malformed
                else ("<IMPROVED_VARIABLE>improved</IMPROVED_VARIABLE>")
            )

    return Engine(), Critic()


@pytest.mark.parametrize("gradient_memory", [0, 2])
def test_textgrad_optimizer_factories_and_custom_loss_keep_backward_graph(
    tmp_path,
    gradient_memory,
):
    tg = pytest.importorskip("textgrad")
    model, critic = engines()
    loss_inputs = []

    def loss(response, example, engine):
        loss_inputs.append((response.requires_grad, example.input))
        return tg.TextLoss("CUSTOM RUBRIC: return the affirmative label", engine=engine)(response)

    def optimizer(**kwargs):
        return tg.TextualGradientDescent(gradient_memory=gradient_memory, **kwargs)

    result = optimize(
        problem="Answer yes",
        seed_prompt="seed",
        model=model,
        backend=TextGrad(
            steps=1,
            optimizer=optimizer,
            optimizer_model=critic,
            loss=loss,
            constraints=("Use only the affirmative label.",),
        ),
        output_dir=tmp_path / str(gradient_memory),
        **splits(),
    )
    assert loss_inputs == [(True, "training-only")]
    assert any("CUSTOM RUBRIC" in prompt for prompt in critic.seen)
    assert any("Use only the affirmative label." in prompt for prompt in critic.seen)
    assert result.scores["final"]["test"]["score"] == 1


def test_textgrad_string_loss_and_malformed_update_are_recorded(tmp_path):
    model, critic = engines(malformed=True)
    directory = tmp_path / "failure"
    with pytest.raises(IndexError, match="malformed update"):
        optimize(
            problem="Answer yes",
            model=model,
            backend=TextGrad(steps=1, optimizer_model=critic, loss="CUSTOM LABEL RUBRIC"),
            output_dir=directory,
            **splits(),
        )
    assert any("CUSTOM LABEL RUBRIC" in prompt for prompt in critic.seen)
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    error = next(e for e in events if e["kind"] == "error")
    assert error["step"] == 1
    assert "malformed update" in error["data"]["message"]
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
