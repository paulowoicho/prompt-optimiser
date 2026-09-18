"""Which system prompt produces better career coaching? Optimise against a pairwise AI judge.

Preference against the seed prompt: the seed prompt answers every question once, those answers
become the targets, and the judge scores each candidate answer against the incumbent for the same
question (1 win, 0 loss, 0.5 tie or disagreement between orderings). The baseline compares the
seed prompt with itself, so it sits near 0.5; it is not exactly 0.5 because the backend renders
the seed its own way (DSPy adds field markers) and serving is not perfectly deterministic. How
far the baseline is from 0.5 is the size of that formatting and noise effect.

    python examples/career_coaching/run.py \
        --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
        --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \
        --backend dspy --optimizer GEPA \
        --optimizer-kwargs '{"max_metric_calls": 120, "reflection_minibatch_size": 3}' \
        --tracker mlflow --tracking-uri sqlite:///runs/mlflow.db --output runs/coaching/dspy-gepa

GEPA is the default; swap --optimizer MIPROv2 (with its kwargs) or --backend textgrad. Judge
with a different, larger model than the one being optimised, or you are measuring self-preference.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for qualified imports
from examples.career_coaching.data import splits  # noqa: E402
from examples.career_coaching.judge import (  # noqa: E402
    CRITERIA,
    VERDICT_SCHEMA,
    WHITESPACE,
    PairwiseJudge,
)
from examples.career_coaching.loss import make_coaching_loss  # noqa: E402
from prompt_optimiser import VLLM, DSPy, Example, TextGrad, optimize  # noqa: E402
from prompt_optimiser.tracking import ConsoleTracker, MLflowTracker, WandbTracker  # noqa: E402

# GEPA is the example default: reflective instruction evolution, budgeted by metric calls.
# Any other DSPy teleprompter is one --optimizer flag away, e.g. --optimizer MIPROv2.
GEPA_DEFAULTS = {"max_metric_calls": 400, "reflection_minibatch_size": 3, "num_threads": 4}

PROBLEM = "You are a career coach. Give a helpful response to the person's career question."


def incumbent_responses(call, config: dict, prompt: str, questions, cache: Path) -> dict[str, str]:
    """Answer each question once with the seed prompt. Cached per model config and seed prompt."""
    key = {"model": config, "seed_prompt": prompt}
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("key") == key and set(stored.get("responses", {})) >= set(questions):
            responses = {q: stored["responses"][q] for q in questions}
        else:
            responses = None
    else:
        responses = None
    if responses is None:
        responses = {question: call(prompt, question).strip() for question in questions}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "responses": responses}, indent=2))
    if len(set(responses.values())) != len(responses):
        raise ValueError(
            "Two questions got identical incumbent responses; the judge lookup needs unique keys"
        )
    return responses


class JudgeArtifacts:
    """Example-local tracker: write the judge's tallies and reasons into the run directory
    when the artifact event fires, so MLflow/W&B (listed after it) upload them too."""

    def __init__(self, judge):
        self.judge = judge

    def log(self, event):
        if event.kind == "artifact":
            directory = Path(event.data["path"])
            (directory / "judge_counts.json").write_text(json.dumps(self.judge.counts, indent=2))
            with (directory / "judge_verdicts.jsonl").open("w", encoding="utf-8") as handle:
                for verdict in self.judge.verdicts:
                    handle.write(json.dumps(verdict) + "\n")

    def close(self, status):
        pass


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
        # Same criteria for the textual critique as for the judge; --loss overrides them.
        loss=make_coaching_loss(args.loss or CRITERIA),
        # A live smoke run learned a prompt that embedded an answer to one training question.
        constraints=(
            "Write reusable coaching instructions. Never include an answer to any specific "
            "question or refer to a particular person's situation.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Model whose prompt is optimised")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--judge-model", required=True, help="A different, stronger model")
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument(
        "--max-tokens", type=int, default=700, help="Answer budget; cut-off answers judge as bad"
    )
    parser.add_argument("--backend", choices=["dspy", "textgrad"], default="dspy")
    parser.add_argument("--optimizer")
    parser.add_argument("--optimizer-kwargs", default="{}")
    parser.add_argument("--compile-kwargs", default="{}")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--loss", help="TextGrad critique criteria (default: the judge's CRITERIA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-prompt", default=PROBLEM)
    parser.add_argument(
        "--responses", type=Path, default=Path("runs/coaching/incumbent_responses.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracker", choices=["local", "mlflow", "wandb"], default="local")
    parser.add_argument("--tracking-uri", default="sqlite:///runs/mlflow.db")
    args = parser.parse_args()

    target = VLLM(args.model, base_url=args.base_url, max_tokens=args.max_tokens, seed=args.seed)
    judge_model = VLLM(
        args.judge_model, base_url=args.judge_base_url, max_tokens=200, seed=args.seed
    )
    # Constrained decoding: the server only lets the judge emit the schema (enum verdict + reason),
    # and only single spaces between tokens (a live run looped on newlines without this).
    judge_call = judge_model.as_litellm()
    judge_call.kwargs["response_format"] = VERDICT_SCHEMA
    judge_call.kwargs["extra_body"] = {"guided_whitespace_pattern": WHITESPACE}

    train_q, validation_q, test_q = splits(args.seed)
    incumbent = incumbent_responses(
        target.as_litellm(),
        target.public_config(),
        args.seed_prompt,
        [*train_q, *validation_q, *test_q],
        args.responses,
    )
    judge = PairwiseJudge(judge_call, question_for={r: q for q, r in incumbent.items()})

    def as_examples(questions):
        return [Example(q, incumbent[q]) for q in questions]

    trackers = [ConsoleTracker(), JudgeArtifacts(judge)]  # judge files land before any upload
    if args.tracker == "mlflow":
        trackers.append(MLflowTracker("career-coaching", tracking_uri=args.tracking_uri))
    elif args.tracker == "wandb":
        trackers.append(WandbTracker("career-coaching", mode="offline"))
    output = args.output or Path("runs/coaching") / (
        f"{args.backend}-{args.optimizer or 'default'}-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )

    result = optimize(
        problem=PROBLEM,
        seed_prompt=args.seed_prompt,
        model=target,
        backend=build_backend(args),
        train_data=as_examples(train_q),
        validation_data=as_examples(validation_q),
        test_data=as_examples(test_q),
        metric=judge,
        random_state=args.seed,
        output_dir=output,
        trackers=trackers,
        metadata={
            "judge_model": judge_model.public_config(),
            "judge_scores": judge.scores,
            "incumbent_prompt": args.seed_prompt,
        },
    )
    print(result.prompt)
    print(json.dumps(result.scores, indent=2))
    print("judge verdicts:", dict(judge.counts))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
