"""Stage 2: optimise a response prompt using the judge that stage 1 calibrated against humans.

The judge is reloaded as the exact predictor stage 1 scored: for DSPy that is ``best/program.json``
(instructions plus any demonstrations), for TextGrad ``best/prompt.txt``. ``--judge baseline``
reloads the same run's unoptimised judge instead, so the effect of calibration is visible on
identical data. The response questions are disjoint from every judge split.

    python -m examples.preference_judge.stage2_responses \\
        --judge-run runs/preference_judge/stage1-gepa \\
        --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \\
        --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \\
        --backend dspy --optimizer GEPA \\
        --optimizer-kwargs '{"max_metric_calls": 400, "reflection_minibatch_size": 3}' \\
        --tracker mlflow --output runs/preference_judge/stage2-gepa-best-judge

Score per question is a tie-adjusted preference against the seed prompt's own answer: 1 if the
judge prefers the candidate in both orderings, 0 if it prefers the incumbent in both, 0.5 for
ties or disagreement between orderings. The incumbent is rendered the same way the backend renders
candidates (see ``seed_responder``), so the baseline sits near 0.5 up to serving nondeterminism.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path

from examples.preference_judge.data import LABELS
from examples.preference_judge.data import SPLIT_SIZES
from examples.preference_judge.data import SWAP
from examples.preference_judge.data import load_pairs
from examples.preference_judge.data import questions
from prompt_optimiser import VLLM
from prompt_optimiser import DSPy
from prompt_optimiser import Example
from prompt_optimiser import TextGrad
from prompt_optimiser import optimize
from prompt_optimiser.tracking import ConsoleTracker
from prompt_optimiser.tracking import MLflowTracker
from prompt_optimiser.tracking import WandbTracker

# GEPA is the example default: reflective instruction evolution, budgeted by metric calls.
# Any other DSPy teleprompter is one --optimizer flag away, e.g. --optimizer MIPROv2.
GEPA_DEFAULTS = {"max_metric_calls": 400, "reflection_minibatch_size": 3, "num_threads": 4}

RESPONSE_SEED = "You are a helpful assistant. Answer the user's request."
PREFERENCE = {"A_BETTER": 1.0, "B_BETTER": 0.0, "TIE": 0.5}


def load_stage1_config(run_dir: Path, pairs) -> dict:
    """Reuse the recorded split assignment, independently of the response optimiser seed."""
    config = json.loads((run_dir / "config.json").read_text())
    splits = config["metadata"]["question_splits"]
    if set(splits) != set(SPLIT_SIZES) or any(not ids for ids in splits.values()):
        raise ValueError("Stage 1 must record all six nonempty question splits")
    ids = [question_id for split in splits.values() for question_id in split]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage-1 question splits overlap")
    if set(ids) != {pair.question_id for pair in pairs}:
        raise ValueError("Dataset questions differ from the stage-1 split assignment")
    return config


def normalise(label: str) -> str:
    return label.strip().upper().strip(".,:;!?\"'`*()[] \n")


def load_judge(run_dir: Path, which: str, judge_model: VLLM) -> Callable[[str], str]:
    """The exact judge predictor from a stage-1 run: ``judge(json_text) -> label text``.

    ``which`` is ``best`` or ``baseline``. A DSPy run leaves ``program.json`` (instructions and
    demonstrations); a TextGrad run leaves ``prompt.txt`` (a system prompt).
    """
    directory = run_dir / which
    if (directory / "program.json").exists():
        import dspy

        program = dspy.Predict("text -> answer")
        program.load(directory / "program.json")
        lm = judge_model.as_dspy()

        def judge(text: str) -> str:
            with dspy.context(lm=lm):
                return program(text=text).answer

        return judge
    if (directory / "prompt.txt").exists():
        system_prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
        call = judge_model.as_litellm()
        return lambda text: call(system_prompt, text)
    raise FileNotFoundError(f"No program.json or prompt.txt under {directory}")


@dataclass
class PreferenceMetric:
    """``metric(expected, predicted)`` where expected is the incumbent answer.

    The judge sees JSON {question, response_a, response_b} in both orderings. Invalid judge output
    is scored as a tie and counted (``on_invalid="tie"``) or raises (``on_invalid="error"``).
    """

    judge: Callable[[str], str]
    question_for: Mapping[str, str]
    on_invalid: str = "tie"
    counts: Counter = field(default_factory=Counter, repr=False)
    verdicts: list[dict] = field(default_factory=list, repr=False)
    __name__ = "calibrated_preference"

    def classify(self, question: str, a: str, b: str) -> str:
        raw = self.judge(json.dumps({"question": question, "response_a": a, "response_b": b}))
        label = normalise(raw)
        if label not in LABELS:
            if self.on_invalid == "error":
                raise ValueError(f"Judge returned an invalid label: {raw!r}")
            self.counts["INVALID"] += 1
            label = "TIE"
        self.verdicts.append({"question": question, "a": a, "b": b, "label": label, "raw": raw})
        return label

    def __call__(self, expected: str, predicted: str) -> float:
        question = self.question_for[expected]
        first = self.classify(question, a=predicted, b=expected)
        second = SWAP[self.classify(question, a=expected, b=predicted)]
        self.counts.update([first, second])
        return (PREFERENCE[first] + PREFERENCE[second]) / 2


class JudgeArtifacts:
    """Write verdicts and tallies into the run directory before any tracker uploads it."""

    def __init__(self, metric: PreferenceMetric):
        self.metric = metric

    def log(self, event):
        if event.kind == "artifact":
            directory = Path(event.data["path"])
            (directory / "judge_counts.json").write_text(json.dumps(self.metric.counts, indent=2))
            with (directory / "judge_verdicts.jsonl").open("w", encoding="utf-8") as handle:
                for verdict in self.metric.verdicts:
                    handle.write(json.dumps(verdict) + "\n")
            result_file = directory / "result.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                metadata = result["config"]["metadata"]
                report = {
                    "preference_vs_seed": {
                        phase: {split: value["score"] for split, value in scores.items()}
                        for phase, scores in result["scores"].items()
                    },
                    "judge": metadata.get("judge"),
                    "judge_run": metadata.get("judge_run"),
                    "verdict_counts": dict(self.metric.counts),
                }
                (directory / "report.json").write_text(json.dumps(report, indent=2))

    def close(self, status):
        pass


def seed_responder(target: VLLM, prompt: str, backend: str):
    """Answer with the seed prompt rendered exactly as the backend will render it.

    A DSPy candidate is rendered through a signature with field markers and the model answers
    differently under that format (shorter, for the 8B). If the incumbent came from a plain chat
    call, every DSPy candidate including the unchanged seed would be compared against a different
    rendering, and the baseline would not sit near 0.5. So the incumbent uses the same rendering.
    """
    if backend == "dspy":
        import dspy

        program = dspy.Predict(dspy.Signature("text -> answer", instructions=prompt))
        lm = target.as_dspy()

        def answer(question: str) -> str:
            with dspy.context(lm=lm):
                return program(text=question).answer

        return answer
    call = target.as_litellm()
    return lambda question: call(prompt, question)


def incumbent_responses(answer, config: dict, prompt: str, texts, cache: Path) -> dict[str, str]:
    """Answer each question once with the seed prompt; cached per rendering, model and prompt."""
    key = {"model": config, "seed_prompt": prompt}
    responses = None
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("key") == key and set(stored.get("responses", {})) >= set(texts):
            responses = {q: stored["responses"][q] for q in texts}
    if responses is None:
        responses = {q: answer(q).strip() for q in texts}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "responses": responses}, indent=2))
    if len(set(responses.values())) != len(responses):
        raise ValueError("Two questions got identical incumbent answers; keys must be unique")
    return responses


def build_backend(args):
    if args.backend == "dspy":
        optimizer = args.optimizer or "GEPA"
        optimizer_kwargs = json.loads(args.optimizer_kwargs)
        if optimizer == "GEPA" and not optimizer_kwargs:
            optimizer_kwargs = dict(GEPA_DEFAULTS)  # a budget is required; only for GEPA
        return DSPy(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            compile_kwargs=json.loads(args.compile_kwargs),
        )
    return TextGrad(
        optimizer=args.optimizer or "TextualGradientDescent",
        optimizer_kwargs=json.loads(args.optimizer_kwargs),
        steps=args.steps,
        batch_size=args.batch_size,
        loss=(
            "Evaluate correctness, helpfulness, relevance, depth and clarity for the input. "
            "The reference is the incumbent answer, not a gold answer to copy. Explain how "
            "reusable system instructions could produce a better answer than this incumbent."
        ),
        constraints=(
            "Write reusable instructions for answering any request. Never include an answer to a "
            "specific question.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-run", type=Path, required=True, help="A stage-1 run directory")
    parser.add_argument("--judge", choices=["best", "baseline"], default="best")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--on-invalid", choices=["tie", "error"], default="tie")
    parser.add_argument("--model", required=True, help="Model whose response prompt is optimised")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--backend", choices=["dspy", "textgrad"], default="dspy")
    parser.add_argument("--optimizer")
    parser.add_argument("--optimizer-kwargs", default="{}")
    parser.add_argument("--compile-kwargs", default="{}")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-prompt", default=RESPONSE_SEED)
    parser.add_argument(
        "--responses", type=Path, default=Path("runs/preference_judge/incumbent_responses.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracker", choices=["local", "mlflow", "wandb"], default="local")
    parser.add_argument("--tracking-uri", default="sqlite:///runs/mlflow.db")
    args = parser.parse_args()

    pairs = load_pairs()
    stage1 = load_stage1_config(args.judge_run, pairs)
    splits = stage1["metadata"]["question_splits"]
    texts = {
        name: questions(pairs, splits[f"response_{name}"])
        for name in ("train", "validation", "test")
    }
    target = VLLM(args.model, base_url=args.base_url, max_tokens=args.max_tokens, seed=args.seed)
    judge_config = stage1["model_config"]
    if args.judge_model != judge_config["model"]:
        parser.error("--judge-model must match the model evaluated in the stage-1 run")
    # The endpoint may move; sampling settings belong to the judge that stage 1 evaluated.
    judge_model = VLLM(**{**judge_config, "base_url": args.judge_base_url})
    incumbent = incumbent_responses(
        seed_responder(target, args.seed_prompt, args.backend),
        {**target.public_config(), "rendering": args.backend},
        args.seed_prompt,
        [*texts["train"], *texts["validation"], *texts["test"]],
        args.responses,
    )
    metric = PreferenceMetric(
        load_judge(args.judge_run, args.judge, judge_model),
        question_for={r: q for q, r in incumbent.items()},
        on_invalid=args.on_invalid,
    )

    def rows(names):
        return [Example(q, incumbent[q]) for q in names]

    trackers = [ConsoleTracker(), JudgeArtifacts(metric)]
    if args.tracker == "mlflow":
        trackers.append(MLflowTracker("preference-judge", tracking_uri=args.tracking_uri))
    elif args.tracker == "wandb":
        trackers.append(WandbTracker("preference-judge", mode="offline"))
    output = args.output or Path("runs/preference_judge") / (
        f"stage2-{args.backend}-{args.optimizer or 'default'}-{args.judge}-judge-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    result = optimize(
        problem="Answer the user's request as helpfully as possible.",
        seed_prompt=args.seed_prompt,
        model=target,
        backend=build_backend(args),
        train_data=rows(texts["train"]),
        validation_data=rows(texts["validation"]),
        test_data=rows(texts["test"]),
        metric=metric,
        random_state=args.seed,
        output_dir=output,
        trackers=trackers,
        metadata={
            "judge_run": str(args.judge_run),
            "judge": args.judge,
            "judge_model": judge_model.public_config(),
            "question_splits": splits,
            "incumbent_prompt": args.seed_prompt,
            "on_invalid": args.on_invalid,
        },
    )
    report = json.loads((result.output_dir / "report.json").read_text())
    print(result.prompt)
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
