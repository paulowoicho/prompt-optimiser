"""A complete optimiser backend in about thirty lines, plugged in with no changes to the library.

It asks the model for a few rewrites of the seed prompt, scores each on validation, and keeps
the best. Not a serious optimiser; it shows the whole contract: one ``fit`` method, a
``FitResult`` of two ``Prompt``s. Run it against a served model:

    python examples/custom_backend.py --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sms_common import PROBLEM  # noqa: E402
from sms_data import load_dataset, make_splits  # noqa: E402

from prompt_optimiser import VLLM, Event, FitResult, Prompt, optimize  # noqa: E402
from prompt_optimiser.experiment import measure  # noqa: E402


@dataclass
class RewriteSearch:
    """Try ``rewrites`` model-written variants of the seed prompt; select on validation.

    A dataclass so its settings land in the run's ``config.json`` automatically.
    """

    rewrites: int = 3

    def fit(self, *, model, seed_prompt, train, validation, metric, greater_is_better, report, **_):
        # Any callable model(system_prompt, text) -> str works; VLLM offers one.
        call = model.as_litellm() if isinstance(model, VLLM) else model
        sign = 1 if greater_is_better else -1
        examples = json.dumps([{"input": r.input, "target": r.target} for r in train[:5]])
        ask = (
            "Rewrite the following system prompt so a model follows it more reliably. "
            f"Labelled examples of the task, for context only: {examples}\n"
            "Return only the rewritten prompt."
        )
        candidates = [Prompt(seed_prompt, lambda x, p=seed_prompt: call(p, x))]
        for _ in range(self.rewrites):
            text = call(ask, seed_prompt).strip()
            candidates.append(Prompt(text, lambda x, p=text: call(p, x)))
        best, best_score = None, None
        for step, candidate in enumerate(candidates):
            score = measure(candidate.predict_one, validation, metric)["score"]
            report(
                Event("candidate", step, {"validation/score": score}, {"prompt": candidate.text})
            )
            if best is None or sign * score > sign * best_score:
                best, best_score = candidate, score
        return FitResult(baseline=candidates[0], best=best)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--rewrites", type=int, default=3)
    args = parser.parse_args()
    rows, _ = load_dataset(Path("runs/datasets/sms.zip"))
    train, validation, test = make_splits(rows, 10, 20, 42)
    result = optimize(
        problem=PROBLEM,
        model=VLLM(args.model, base_url=args.base_url, max_tokens=256),
        backend=RewriteSearch(args.rewrites),
        train_data=train,
        validation_data=validation,
        test_data=test,
    )
    print(result.prompt)
    print(json.dumps(result.scores, indent=2))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
