"""A complete optimiser backend in about thirty lines, plugged in with no changes to the library.

It asks the model for a few rewrites of the seed prompt, scores each on validation, and keeps the
best. Not a serious optimiser; it shows the whole contract: one ``fit`` method returning a
``FitResult`` of two ``Prompt``s. Self-contained: a tiny sentiment task is defined below.

    python examples/custom_backend/rewrite_search.py \
        --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from prompt_optimiser import VLLM, Event, Example, FitResult, Prompt, optimize
from prompt_optimiser.experiment import measure


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


SENTIMENT = [
    ("Absolutely loved it, would buy again.", "positive"),
    ("Broke after two days. Waste of money.", "negative"),
    ("Does what it says. Nothing special.", "neutral"),
    ("Customer service was rude and unhelpful.", "negative"),
    ("Exceeded every expectation I had.", "positive"),
    ("Arrived on time. Haven't used it much yet.", "neutral"),
    ("The best purchase I've made this year.", "positive"),
    ("Cheap plastic, smells odd, returning it.", "negative"),
    ("It's fine. Average quality for the price.", "neutral"),
    ("Five stars, my kids adore it.", "positive"),
    ("Stopped working after a week and no refund.", "negative"),
    ("Okay product, okay packaging, okay delivery.", "neutral"),
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--rewrites", type=int, default=3)
    args = parser.parse_args()
    rows = [Example(text, label) for text, label in SENTIMENT]
    result = optimize(
        problem="Classify the review's sentiment. Return only positive, negative or neutral.",
        model=VLLM(args.model, base_url=args.base_url, max_tokens=64),
        backend=RewriteSearch(args.rewrites),
        train_data=rows[:6],
        validation_data=rows[6:9],
        test_data=rows[9:],
    )
    print(result.prompt)
    print(json.dumps(result.scores, indent=2))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
