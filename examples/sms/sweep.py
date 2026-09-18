"""Run several optimiser configurations on the same SMS splits and model, one after another.

    python examples/sms/sweep.py --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
        --output runs/sweep-8b --tracking-uri sqlite:///runs/mlflow.db

Each configuration gets its own directory under --output and its own MLflow run. Failures are
recorded in summary.json and the sweep continues. Edit CONFIGS to add or change configurations;
that is the point of the exercise. Derived from Codex's runs/review_8b.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from examples.sms.common import PROBLEM  # noqa: E402
from examples.sms.data import load_dataset, make_splits  # noqa: E402
from prompt_optimiser import VLLM, DSPy, TextGrad, exact_match, optimize  # noqa: E402
from prompt_optimiser.tracking import ConsoleTracker, MLflowTracker  # noqa: E402

CONFIGS = {
    "dspy-gepa": lambda: DSPy(
        optimizer="GEPA",
        optimizer_kwargs={"max_metric_calls": 400, "reflection_minibatch_size": 3},
    ),
    "dspy-miprov2": lambda: DSPy(
        optimizer="MIPROv2",
        optimizer_kwargs={"auto": None, "num_candidates": 3, "max_errors": 1},
        compile_kwargs={"num_trials": 3, "minibatch": False},
    ),
    "dspy-copro": lambda: DSPy(
        optimizer="COPRO",
        optimizer_kwargs={"breadth": 3, "depth": 2},
        compile_kwargs={"eval_kwargs": {"num_threads": 4, "display_progress": False}},
    ),
    "dspy-labeled-fewshot": lambda: DSPy(optimizer="LabeledFewShot", optimizer_kwargs={"k": 4}),
    "dspy-bootstrap-random-search": lambda: DSPy(
        optimizer="BootstrapFewShotWithRandomSearch",
        optimizer_kwargs={"max_bootstrapped_demos": 2, "num_candidate_programs": 3},
    ),
    "textgrad-tgd": lambda: TextGrad(steps=3, batch_size=2),
    "textgrad-tgd-memory": lambda: TextGrad(
        steps=3, batch_size=2, optimizer_kwargs={"gradient_memory": 2}
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--optimizer-model", help="Proposer/critic served on the same URL")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--train-per-class", type=int, default=10)
    parser.add_argument("--eval-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", help="Subset of configuration names to run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="sms-sweep")
    parser.add_argument("--tracking-uri", default="sqlite:///runs/mlflow.db")
    args = parser.parse_args()

    rows, digest = load_dataset(Path("runs/datasets/sms.zip"))
    train, validation, test = make_splits(
        rows, args.train_per_class, args.eval_per_class, args.seed
    )
    model = VLLM(args.model, base_url=args.base_url, max_tokens=args.max_tokens, seed=args.seed)
    proposer = (
        VLLM(args.optimizer_model, base_url=args.base_url, max_tokens=args.max_tokens)
        if args.optimizer_model
        else None
    )
    names = args.only or list(CONFIGS)
    summary = []
    for name in names:
        backend = CONFIGS[name]()
        if proposer is not None and hasattr(backend, "optimizer_model"):
            backend.optimizer_model = proposer
        started = time.monotonic()
        print(f"START {name}", flush=True)
        entry = {"name": name, "output_dir": str(args.output / name)}
        try:
            result = optimize(
                problem=PROBLEM,
                model=model,
                backend=backend,
                train_data=train,
                validation_data=validation,
                test_data=test,
                metric=exact_match,
                random_state=args.seed,
                output_dir=args.output / name,
                trackers=[
                    ConsoleTracker(),
                    MLflowTracker(args.experiment, tracking_uri=args.tracking_uri),
                ],
                metadata={
                    "dataset": "UCI SMS Spam Collection",
                    "archive_sha256": digest,
                    "config_name": name,
                },
            )
            entry.update(status="finished", scores=result.scores, prompt=result.prompt)
        except Exception as exc:
            traceback.print_exc()
            entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        entry["seconds"] = round(time.monotonic() - started, 1)
        summary.append(entry)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("RESULT " + json.dumps({k: v for k, v in entry.items() if k != "prompt"}), flush=True)


if __name__ == "__main__":
    main()
