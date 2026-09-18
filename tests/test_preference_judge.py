"""The two-stage judge example: leak-free splits, both orderings, majority votes, reloaded judge."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from prompt_optimiser import Example, FitResult, Prompt, optimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

data = importlib.import_module("examples.preference_judge.data")
stage2 = importlib.import_module("examples.preference_judge.stage2_responses")


def synthetic_pairs():
    """Two pairs per MT-bench question id, deterministic labels."""
    pairs = []
    for question_id in range(81, 161):
        for k, (a, b) in enumerate((("alpaca", "gpt-4"), ("claude", "vicuna"))):
            pairs.append(
                data.Pair(
                    question_id=question_id,
                    category=data.category(question_id),
                    question=f"Q{question_id}",
                    model_a=a,
                    model_b=b,
                    response_a=f"A{question_id}.{k}",
                    response_b=f"B{question_id}.{k}",
                    label=data.LABELS[(question_id + k) % 3],
                    votes=1,
                    unanimous=True,
                )
            )
    return pairs


def test_votes_merge_across_orderings_and_majority_wins_or_ties():
    pd = pytest.importorskip("pandas")

    def conv(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    rows = [
        # Three votes on the same pair, two of them with A and B swapped: 2 x "gpt-4 wins".
        dict(question_id=81, model_a="alpaca", model_b="gpt-4", winner="model_b", turn=1,
             conversation_a=conv("q", "alpaca says"), conversation_b=conv("q", "gpt says")),
        dict(question_id=81, model_a="gpt-4", model_b="alpaca", winner="model_a", turn=1,
             conversation_a=conv("q", "gpt says"), conversation_b=conv("q", "alpaca says")),
        dict(question_id=81, model_a="alpaca", model_b="gpt-4", winner="tie", turn=1,
             conversation_a=conv("q", "alpaca says"), conversation_b=conv("q", "gpt says")),
        # Split decision: one vote each way -> TIE.
        dict(question_id=82, model_a="claude", model_b="vicuna", winner="model_a", turn=1,
             conversation_a=conv("q2", "c"), conversation_b=conv("q2", "v")),
        dict(question_id=82, model_a="claude", model_b="vicuna", winner="model_b", turn=1,
             conversation_a=conv("q2", "c"), conversation_b=conv("q2", "v")),
        # Second turn is ignored.
        dict(question_id=83, model_a="claude", model_b="vicuna", winner="model_a", turn=2,
             conversation_a=conv("q3", "c"), conversation_b=conv("q3", "v")),
    ]
    pairs = data.pairs_from_frame(pd.DataFrame(rows))
    facts = [(p.question_id, p.model_a, p.model_b, p.label, p.votes, p.unanimous) for p in pairs]
    assert facts == [
        (81, "alpaca", "gpt-4", "B_BETTER", 3, False),
        (82, "claude", "vicuna", "TIE", 2, False),
    ]
    assert pairs[0].response_a == "alpaca says" and pairs[0].response_b == "gpt says"
    assert data.summary(pairs)["tied_vote_pairs"] == 1


def test_question_splits_are_disjoint_stratified_and_reproducible():
    pairs = synthetic_pairs()
    splits = data.split_questions(pairs, seed=3)
    ids = [q for s in splits.values() for q in s]
    assert len(ids) == 80 == len(set(ids))
    assert {name: len(s) for name, s in splits.items()} == {
        "judge_train": 24, "judge_validation": 8, "judge_test": 8,
        "response_train": 16, "response_validation": 8, "response_test": 16,
    }
    for name, size in data.SPLIT_SIZES.items():
        per_category = {
            c: sum(1 for q in splits[name] if data.category(q) == c) for c in data.CATEGORIES
        }
        assert set(per_category.values()) == {size}, name
    assert splits == data.split_questions(pairs, seed=3)
    assert splits != data.split_questions(pairs, seed=4)


def test_judge_examples_contain_both_orderings_with_swapped_labels_and_cap_pairs():
    pairs = synthetic_pairs()
    rows = data.judge_examples(pairs, [81, 82], pairs_per_question=1, seed=0)
    assert len(rows) == 4  # 2 questions x 1 pair x 2 orderings
    forward, backward = rows[0], rows[1]
    f, b = json.loads(forward.input), json.loads(backward.input)
    assert f["question"] == b["question"]
    assert (f["response_a"], f["response_b"]) == (b["response_b"], b["response_a"])
    assert backward.target == data.SWAP[forward.target]
    assert all(row.target in data.LABELS for row in rows)
    assert data.questions(pairs, [82, 81]) == ["Q81", "Q82"]


def test_preference_metric_swaps_scores_and_counts_invalid_output():
    incumbent = {"q1": "plain", "q2": "plain2"}
    seen = []

    def judge(text):
        payload = json.loads(text)
        seen.append(payload)
        a, b = payload["response_a"], payload["response_b"]
        if "better" in a and "better" not in b:
            return "a_better."
        if "better" in b and "better" not in a:
            return "B_BETTER"
        if a == "garbage":
            return "I cannot decide"
        return "TIE"

    metric = stage2.PreferenceMetric(judge, question_for={r: q for q, r in incumbent.items()})
    assert metric("plain", "a better answer") == 1.0
    assert metric("plain", "plain") == 0.5
    assert seen[0] == {"question": "q1", "response_a": "a better answer", "response_b": "plain"}
    assert seen[1] == {"question": "q1", "response_a": "plain", "response_b": "a better answer"}
    assert metric("plain2", "garbage") == 0.5  # invalid in one ordering -> tie, counted
    assert metric.counts["INVALID"] == 1
    strict = stage2.PreferenceMetric(judge, question_for={"plain2": "q2"}, on_invalid="error")
    with pytest.raises(ValueError, match="invalid label"):
        strict("plain2", "garbage")


def test_load_judge_reloads_textgrad_prompt_and_dspy_program(tmp_path, monkeypatch):
    run = tmp_path / "stage1"
    (run / "best").mkdir(parents=True)
    (run / "best" / "prompt.txt").write_text("BE A JUDGE")
    calls = []

    class FakeVLLM:
        def as_litellm(self):
            return lambda system, text: calls.append((system, text)) or "TIE"

    judge = stage2.load_judge(run, "best", FakeVLLM())
    assert judge('{"question": "q"}') == "TIE" and calls == [("BE A JUDGE", '{"question": "q"}')]
    with pytest.raises(FileNotFoundError):
        stage2.load_judge(run, "baseline", FakeVLLM())

    dspy = pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    (run / "baseline").mkdir()
    program = dspy.Predict(dspy.Signature("text -> answer", instructions="seed judge"))
    program.demos = [dspy.Example(text="TRAINING DEMO", answer="TIE")]
    program.save(run / "baseline" / "program.json")
    lm = DummyLM([{"answer": "A_BETTER"}] * 4)

    class FakeDSPyVLLM:
        def as_dspy(self):
            return lm

    judge = stage2.load_judge(run, "baseline", FakeDSPyVLLM())
    assert judge('{"question": "q"}') == "A_BETTER"
    assert any(message["content"] == "TRAINING DEMO" or "TRAINING DEMO" in message["content"]
               for message in lm.history[-1]["messages"])


def test_stage2_metric_runs_through_optimize(tmp_path):
    incumbent = {f"q{i}": f"seed answer {i}" for i in range(5)}

    def judge(text):
        payload = json.loads(text)
        a, b = payload["response_a"], payload["response_b"]
        if a == b:
            return "TIE"
        return "A_BETTER" if a.startswith("improved") else "B_BETTER"

    metric = stage2.PreferenceMetric(judge, question_for={r: q for q, r in incumbent.items()})

    class TwoPrompts:
        def fit(self, *, seed_prompt, **_):
            return FitResult(
                Prompt(seed_prompt, lambda q: incumbent[q]),
                Prompt("be better", lambda q: "improved " + incumbent[q]),
            )

    rows = [Example(q, r) for q, r in incumbent.items()]
    result = optimize(
        problem="Answer well",
        model="fake",
        backend=TwoPrompts(),
        train_data=rows[:2],
        validation_data=rows[2:4],
        test_data=rows[4:],
        metric=metric,
        output_dir=tmp_path / "run",
        trackers=[stage2.JudgeArtifacts(metric)],
    )
    assert result.scores["baseline"]["test"]["score"] == 0.5
    assert result.scores["final"]["test"]["score"] == 1.0
    assert json.loads((result.output_dir / "judge_counts.json").read_text())["A_BETTER"] >= 1
    lines = (result.output_dir / "judge_verdicts.jsonl").read_text().count("\n")
    assert lines == len(metric.verdicts)
    assert json.loads((result.output_dir / "report.json").read_text())[
        "preference_vs_seed"
    ]["final"]["test"] == 1.0


def test_stage2_reuses_stage1_splits_and_rejects_overlap(tmp_path):
    pairs = synthetic_pairs()
    splits = data.split_questions(pairs, seed=7)
    config = {"metadata": {"question_splits": splits}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    restored = stage2.load_stage1_config(tmp_path, pairs)["metadata"]["question_splits"]
    assert restored == splits
    assert restored != data.split_questions(pairs, seed=42)
    splits["response_test"][0] = splits["judge_train"][0]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlap"):
        stage2.load_stage1_config(tmp_path, pairs)


def test_seed_responder_matches_native_adapter_rendering(monkeypatch):
    pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    from prompt_optimiser import VLLM, DSPy

    lm = DummyLM([{"answer": "answer"}] * 10)
    monkeypatch.setattr(VLLM, "as_dspy", lambda self: lm)
    target = VLLM("fake")
    seed = "Answer the question carefully."
    question = "What is the answer?"
    assert stage2.seed_responder(target, seed, "dspy")(question) == "answer"
    incumbent_messages = lm.history[-1]["messages"]

    class KeepSeed:
        def compile(self, student, trainset, valset):
            return student

    trained = DSPy(optimizer=KeepSeed).fit(
        problem="Answer", model=target, seed_prompt=seed,
        train=[Example("train", "answer")], validation=[Example("val", "answer")],
        metric=lambda expected, predicted: float(expected == predicted),
        greater_is_better=True, random_state=42, report=lambda event: None,
    )
    assert trained.baseline.predict_one(question) == "answer"
    assert lm.history[-1]["messages"] == incumbent_messages


def test_stage1_retained_seed_report_exists_before_artifact_upload(tmp_path):
    stage1 = importlib.import_module("examples.preference_judge.stage1_judge")

    class KeepSeed:
        def fit(self, *, seed_prompt, **kwargs):
            baseline = Prompt(seed_prompt, lambda text: "TIE")
            return FitResult(baseline, baseline)

    uploaded = []

    class Upload:
        def log(self, event):
            if event.kind == "artifact":
                uploaded.append(json.loads((Path(event.data["path"]) / "report.json").read_text()))

        def close(self, status):
            pass

    result = optimize(
        problem="Judge", model="fake", backend=KeepSeed(),
        train_data=[Example("train", "TIE")],
        validation_data=[Example("val", "TIE")],
        test_data=[Example("forward", "TIE"), Example("swapped", "TIE")],
        output_dir=tmp_path / "run", metadata={"question_splits": {}},
        trackers=[stage1.ReportArtifacts(), Upload()],
    )
    assert not (result.output_dir / "predictions/final-test.jsonl").exists()
    assert len(uploaded) == 1
    assert uploaded[0]["judge_test"]["final"] == uploaded[0]["judge_test"]["baseline"]
    assert uploaded[0]["judge_test"]["final"]["position_consistency"] == 1.0
