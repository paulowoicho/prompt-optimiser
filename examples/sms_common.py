"""Dataset and reporting shared by the two standalone native reference scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sms_data import load_dataset, make_splits

from prompt_optimiser.experiment import measure
from prompt_optimiser.tracking import ConsoleTracker, JSONLTracker, MLflowTracker, WandbTracker
from prompt_optimiser.types import Event
from prompt_optimiser.vllm import VLLM

PROBLEM = (
    "Classify an SMS message as ham (legitimate) or spam (unsolicited promotion or scam). "
    "Return exactly ham or spam, without explanation."
)


def setup(name, add_arguments=None):
    parser = argparse.ArgumentParser(description=f"{name}: UCI SMS prompt optimization")
    if add_arguments is not None:
        add_arguments(parser)
    parser.add_argument("--model", required=True, help="Provider/model identifier")
    parser.add_argument("--base-url", help="vLLM OpenAI-compatible URL ending in /v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--optimizer-model", help="Proposal/critic model; defaults to target model")
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--eval-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--seed-prompt", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracker", choices=["local", "mlflow", "wandb"], default="local")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--backend", choices=["dspy", "textgrad"], default="dspy")
    args = parser.parse_args()
    if min(args.train_per_class, args.eval_per_class, args.batch_size) < 1:
        parser.error("dataset sizes and batch size must be positive")
    if args.steps < 0 or args.num_candidates < 1 or args.max_demos < 0:
        parser.error("steps, candidates and max-demos must be nonnegative")
    args.output = args.output or Path("runs") / (
        name + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    )
    rows, digest = load_dataset(Path("runs/datasets/sms.zip"))
    train, validation, test = make_splits(
        rows, args.train_per_class, args.eval_per_class, args.seed
    )
    metadata = {
        "dataset": "UCI SMS Spam Collection",
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "license": "CC BY 4.0",
        "citation": "Almeida & Hidalgo (2011), https://doi.org/10.24432/C5CC84",
        "archive_sha256": digest,
        "balanced_classes": True,
        "split_sha256": {
            name: hashlib.sha256(
                json.dumps([[row.input, row.target] for row in split]).encode()
            ).hexdigest()
            for name, split in (("train", train), ("validation", validation), ("test", test))
        },
    }
    return args, (train, validation, test), metadata


def trackers_for(args):
    trackers = [ConsoleTracker()]
    if args.tracker == "mlflow":
        trackers.append(MLflowTracker("sms-prompts", tracking_uri=args.tracking_uri))
    elif args.tracker == "wandb":
        trackers.append(WandbTracker("sms-prompts", mode=args.wandb_mode))
    return trackers


class NativeRun:
    """Only IO and reporting; optimization remains visible in each native script."""

    def __init__(self, args, metadata):
        self.args = args
        self.metadata = metadata
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise ValueError("Use a new or empty output directory")
        self.trackers = [JSONLTracker(args.output / "progress.jsonl"), *trackers_for(args)]
        self.step = 0

    def report(self, event):
        for tracker in self.trackers:
            tracker.log(event)

    def __enter__(self):
        config = {**vars(self.args), "output": str(self.args.output), "dataset": self.metadata}
        (self.args.output / "config.json").write_text(json.dumps(config, indent=2))
        try:
            self.report(Event("start", 0, {}, config))
        except BaseException:
            self.__exit__(RuntimeError, None, None)
            raise
        return self

    def progress(self, prompt, score):
        self.report(Event("candidate", self.step, {"validation/score": score}, {"prompt": prompt}))
        self.step += 1

    def finish(self, baseline, best, splits, prompt, metric):
        scores = {
            phase: {
                name: measure(predict, rows, metric)
                for name, rows in zip(("train", "validation", "test"), splits, strict=True)
            }
            for phase, predict in (("baseline", baseline), ("final", best))
        }
        result = {"prompt": prompt, "scores": scores, "dataset": self.metadata}
        (self.args.output / "result.json").write_text(json.dumps(result, indent=2))
        (self.args.output / "prompt.txt").write_text(prompt)
        self.report(Event("artifact", self.step, {}, {"path": str(self.args.output.resolve())}))
        metrics = {
            f"{phase}/{split}/score": value["score"]
            for phase, values in scores.items()
            for split, value in values.items()
        }
        self.report(Event("finish", self.step, metrics, result))
        print(json.dumps(scores, indent=2))
        print(f"Artifacts: {self.args.output}")

    def __exit__(self, exc_type, exc, traceback):
        for tracker in self.trackers:
            tracker.close("finished" if exc_type is None else "failed")


def target_model(args):
    return (
        VLLM(
            args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens, seed=args.seed,
        )
        if args.base_url
        else args.model
    )


def optimizer_model(args):
    if not args.optimizer_model:
        return None
    return (
        VLLM(
            args.optimizer_model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens, seed=args.seed,
        )
        if args.base_url
        else args.optimizer_model
    )
