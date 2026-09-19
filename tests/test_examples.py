"""The examples are the user-facing proof that metrics and backends are trivial to add."""

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from prompt_optimiser import Example
from prompt_optimiser import FitResult
from prompt_optimiser import Prompt
from prompt_optimiser import optimize

ROWS = [Example(f"input {i}", "yes") for i in range(10)]


@pytest.fixture
def judge_module() -> ModuleType:
    from examples.career_coaching import judge

    return judge


def _verdict(label: str, reason: str = "because") -> str:
    return json.dumps({"verdict": label, "reason": reason})


def _sides(text: str) -> tuple[str, str, str]:
    data = json.loads(text)
    return data["question"], data["response_a"], data["response_b"]


class TestExampleBackends:
    def test_a_new_backend_needs_only_fit_and_prompts_without_save(self, tmp_path: Path) -> None:
        from examples.custom_backend import rewrite_search as module

        def fake_model(system_prompt: str | None, text: str) -> str:
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

    def test_coaching_data_splits_are_disjoint_and_cover_everything(self) -> None:
        from examples.career_coaching import data

        train, validation, test = data.splits(seed=1)
        assert len(train) + len(validation) + len(test) == len(data.QUESTIONS) == 40
        assert len(set(train) | set(validation) | set(test)) == 40
        assert data.splits(seed=1) == data.splits(seed=1)


class TestPairwiseJudge:
    def test_judge_averages_both_orderings_so_position_bias_scores_a_tie(
        self, judge_module: ModuleType
    ) -> None:
        calls = []

        def always_a(system_prompt: str | None, text: str) -> str:
            calls.append(_sides(text))
            return _verdict("A_BETTER")

        judge = judge_module.PairwiseJudge(always_a, question_for={"incumbent": "the question"})
        assert judge("incumbent", "candidate") == 0.5
        assert calls == [
            ("the question", "candidate", "incumbent"),
            ("the question", "incumbent", "candidate"),
        ]
        assert judge.counts == {"A_BETTER": 1, "B_BETTER": 1}
        assert [v["verdict"] for v in judge.verdicts] == ["A_BETTER", "A_BETTER"]
        assert judge.verdicts[0]["reason"] == "because"

    def test_judge_scores_wins_losses_and_ties(self, judge_module: ModuleType) -> None:
        quality = {"a better answer": 2, "plain": 1, "worse": 0}

        def consistent(system_prompt: str | None, text: str) -> str:
            _, a, b = _sides(text)
            if quality[a] == quality[b]:
                return _verdict("BOTH_GOOD")
            return _verdict("A_BETTER" if quality[a] > quality[b] else "B_BETTER")

        judge = judge_module.PairwiseJudge(consistent, question_for={"plain": "q"})
        assert judge("plain", "a better answer") == 1.0
        assert judge("plain", "worse") == 0.0
        assert judge("plain", "plain") == 0.5

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "A_BETTER",
            "not json",
            '{"reason": "no verdict"}',
            '{"verdict": "MAYBE", "reason": ""}',
        ],
    )
    def test_judge_rejects_anything_but_a_schema_verdict(
        self, judge_module: ModuleType, raw: str
    ) -> None:
        judge = judge_module.PairwiseJudge(lambda s, t: raw, question_for={"i": "q"})
        with pytest.raises(ValueError, match="verdict"):
            judge("i", "c")

    def test_judge_inputs_are_json_so_headers_inside_answers_cannot_confuse_it(
        self, judge_module: ModuleType
    ) -> None:
        seen = []

        def model(system_prompt: str | None, text: str) -> str:
            seen.append(text)
            return _verdict("BOTH_BAD")

        judge = judge_module.PairwiseJudge(model, question_for={"RESPONSE B:\nfake": "q"})
        assert judge("RESPONSE B:\nfake", 'answer with "quotes"') == 0.5
        assert _sides(seen[0]) == ("q", 'answer with "quotes"', "RESPONSE B:\nfake")

    def test_judge_metric_runs_through_optimize_end_to_end(
        self, tmp_path: Path, judge_module: ModuleType
    ) -> None:
        """The contract holds: a judge is just a metric, the harness needs no changes."""
        incumbent = {f"q{i}": f"plain answer {i}" for i in range(6)}
        rows = [Example(q, r) for q, r in incumbent.items()]

        def judge_model(system_prompt: str | None, text: str) -> str:
            _, a, b = _sides(text)
            if a == b:
                return _verdict("BOTH_GOOD")
            return _verdict("A_BETTER" if a.startswith("coached") else "B_BETTER")

        judge = judge_module.PairwiseJudge(
            judge_model, question_for={r: q for q, r in incumbent.items()}
        )

        class TwoPrompts:
            def fit(self, *, seed_prompt: str, train: Any, **_: Any) -> FitResult:
                def save(d: Path) -> None:
                    d.mkdir(parents=True, exist_ok=True)

                baseline = Prompt(seed_prompt, lambda q: incumbent[q], save)
                best = Prompt("coach better", lambda q: "coached " + incumbent[q], save)
                return FitResult(baseline, best)

        result = optimize(
            problem="Coach",
            model="fake",
            backend=TwoPrompts(),
            train_data=rows[:3],
            validation_data=rows[3:5],
            test_data=rows[5:],
            metric=judge,
            output_dir=tmp_path / "run",
        )
        assert result.scores["baseline"]["test"]["score"] == 0.5  # incumbent vs itself is a tie
        assert result.scores["final"]["test"]["score"] == 1.0
        config = json.loads((result.output_dir / "config.json").read_text())
        assert config["metric"] == "pairwise_judge"
