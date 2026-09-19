"""Stage 2 for coaching: optimise the 8B's coaching prompt with the calibrated judge as the metric.

Reloads the exact judge that ``calibrate_judge.py`` scored (DSPy ``best/program.json``, or
TextGrad ``best/prompt.txt``), wraps it in a pairwise metric with position swap, and optimises
the coaching prompt on the response questions, which are disjoint from every judge split. The
incumbent is the seed prompt's own answer, rendered the way the backend renders candidates, so the
baseline sits at 0.5. ``--judge baseline`` uses the uncalibrated judge from the same run for
contrast.

    python -m examples.career_coaching.optimize_with_judge \\
        --judge-run runs/coaching_preference/stage1-gepa --judge best \\
        --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \\
        --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \\
        --backend dspy --optimizer GEPA \\
        --tracker mlflow --output runs/coaching_preference/stage2-gepa-best-judge
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

from examples.career_coaching import preference_data as pdm
from prompt_optimiser import VLLM
from prompt_optimiser import DSPy
from prompt_optimiser import Example
from prompt_optimiser import TextGrad
from prompt_optimiser import optimize
from prompt_optimiser.tracking import ConsoleTracker
from prompt_optimiser.tracking import MLflowTracker
from prompt_optimiser.tracking import WandbTracker

GEPA_DEFAULTS = {"max_metric_calls": 400, "reflection_minibatch_size": 3, "num_threads": 4}
PREFERENCE = {"A_BETTER": 1.0, "B_BETTER": 0.0, "TIE": 0.5}


def load_stage1_config(run_dir: Path) -> dict:
    """Load the immutable question snapshot and split assignment from calibration."""
    config = json.loads((run_dir / "config.json").read_text())
    metadata = config["metadata"]
    questions = metadata.get("questions")
    if not questions:
        raise ValueError("Stage 1 must record its question snapshot; rerun calibration")
    question_ids = [q["question_id"] for q in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Stage-1 question snapshot contains duplicate ids")
    texts = [" ".join(q["question"].casefold().split()) for q in questions]
    if not all(texts) or len(texts) != len(set(texts)):
        raise ValueError("Stage-1 questions must contain distinct nonempty text")
    splits = metadata["question_splits"]
    if set(splits) != set(pdm.SPLIT_FRACTIONS) or any(not ids for ids in splits.values()):
        raise ValueError("Stage 1 must record all six nonempty question splits")
    ids = [qid for split in splits.values() for qid in split]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage-1 question splits overlap")
    if set(ids) != set(question_ids):
        raise ValueError("Stage-1 questions differ from its split assignment")
    return config


def saved_judge_model(stage1: dict, model_name: str, base_url: str) -> VLLM:
    """Keep calibrated sampling settings; only the serving address may change."""
    config = stage1["model_config"]
    if model_name != config["model"]:
        raise ValueError("Judge model must match the stage-1 model")
    return VLLM(**{**config, "base_url": base_url})


def normalise(label: str) -> str:
    return label.strip().upper().strip(".,:;!?\"'`*()[] \n")


def load_judge(run_dir: Path, which: str, judge_model: VLLM) -> Callable[[str], str]:
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
    judge: Callable[[str], str]
    question_for: Mapping[str, str]
    on_invalid: str = "tie"
    counts: Counter = field(default_factory=Counter, repr=False)
    verdicts: list[dict] = field(default_factory=list, repr=False)
    __name__ = "calibrated_coaching_preference"

    def classify(self, question: str, a: str, b: str) -> str:
        raw = self.judge(json.dumps({"question": question, "response_a": a, "response_b": b}))
        label = normalise(raw)
        if label not in pdm.LABELS:
            if self.on_invalid == "error":
                raise ValueError(f"Judge returned an invalid label: {raw!r}")
            self.counts["INVALID"] += 1
            label = "TIE"
        self.verdicts.append({"question": question, "a": a, "b": b, "label": label, "raw": raw})
        return label

    def __call__(self, expected: str, predicted: str) -> float:
        question = self.question_for[expected]
        first = self.classify(question, a=predicted, b=expected)
        second = pdm.SWAP[self.classify(question, a=expected, b=predicted)]
        self.counts.update([first, second])
        return (PREFERENCE[first] + PREFERENCE[second]) / 2


class JudgeArtifacts:
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
    """Render the incumbent exactly as the backend renders candidates."""
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


def incumbent_responses(answer, key: dict, texts, cache: Path) -> dict[str, str]:
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
        kwargs = json.loads(args.optimizer_kwargs)
        if optimizer == "GEPA" and not kwargs:
            kwargs = dict(GEPA_DEFAULTS)
        return DSPy(
            optimizer=optimizer,
            optimizer_kwargs=kwargs,
            compile_kwargs=json.loads(args.compile_kwargs),
        )
    return TextGrad(
        optimizer=args.optimizer or "TextualGradientDescent",
        optimizer_kwargs=json.loads(args.optimizer_kwargs),
        steps=args.steps,
        batch_size=args.batch_size,
        loss=(
            "Evaluate how well the response addresses this person's career situation, with "
            "realistic, ethical and actionable advice. The reference is the incumbent answer, "
            "not a gold answer to copy. Explain how reusable system instructions could produce "
            "a better coaching answer than this incumbent."
        ),
        constraints=(
            "Write reusable coaching instructions. Never include an answer to any specific "
            "question or refer to a particular person's situation.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--judge", choices=["best", "baseline"], default="best")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--on-invalid", choices=["tie", "error"], default="tie")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--backend", choices=["dspy", "textgrad"], default="dspy")
    parser.add_argument("--optimizer")
    parser.add_argument("--optimizer-kwargs", default="{}")
    parser.add_argument("--compile-kwargs", default="{}")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-prompt", default=pdm.SEED_PROMPT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracker", choices=["local", "mlflow", "wandb"], default="local")
    parser.add_argument("--tracking-uri", default="sqlite:///runs/mlflow.db")
    args = parser.parse_args()

    # Reuse the calibration run's question splits so the two stages never overlap.
    stage1 = load_stage1_config(args.judge_run)
    splits = stage1["metadata"]["question_splits"]
    questions = stage1["metadata"]["questions"]
    by_id = {q["question_id"]: q["question"] for q in questions}
    texts = {
        name: [by_id[i] for i in splits[f"response_{name}"]]
        for name in ("train", "validation", "test")
    }
    target = VLLM(args.model, base_url=args.base_url, max_tokens=args.max_tokens, seed=args.seed)
    judge_model = saved_judge_model(stage1, args.judge_model, args.judge_base_url)
    incumbent = incumbent_responses(
        seed_responder(target, args.seed_prompt, args.backend),
        {**target.public_config(), "rendering": args.backend, "seed_prompt": args.seed_prompt},
        [*texts["train"], *texts["validation"], *texts["test"]],
        pdm.CACHE / "incumbent_responses.json",
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
        trackers.append(MLflowTracker("coaching-preference", tracking_uri=args.tracking_uri))
    elif args.tracker == "wandb":
        trackers.append(WandbTracker("coaching-preference", mode="offline"))
    output = args.output or Path("runs/coaching_preference") / (
        f"stage2-{args.backend}-{args.optimizer or 'gepa'}-{args.judge}-judge-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    result = optimize(
        problem="Give the most helpful career coaching response to the person's message.",
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
            "questions": questions,
            "incumbent_prompt": args.seed_prompt,
        },
    )
    report = json.loads((result.output_dir / "report.json").read_text())
    print(result.prompt)
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
