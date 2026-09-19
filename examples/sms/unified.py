"""One task, one dataset, any optimiser: the experimenter picks the tool and its algorithm.

Examples, against a served model:

  # DSPy, GEPA (the example default)
  python -m examples.sms.unified --backend dspy --optimizer GEPA \
      --optimizer-kwargs '{"max_metric_calls": 400, "reflection_minibatch_size": 3}' \
      --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1

  # DSPy, MIPROv2
  python -m examples.sms.unified --backend dspy --optimizer MIPROv2 \
      --optimizer-kwargs '{"auto": "light"}' \
      --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1

  # DSPy, COPRO
  python -m examples.sms.unified --backend dspy --optimizer COPRO \
      --optimizer-kwargs '{"breadth": 4, "depth": 2}' \
      --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1

  # DSPy, bootstrapped demonstrations with random search
  python -m examples.sms.unified --backend dspy --optimizer BootstrapFewShotWithRandomSearch \
      --optimizer-kwargs '{"max_bootstrapped_demos": 4, "num_candidate_programs": 4}' \
      --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1

  # TextGrad, textual gradient descent with gradient memory and a custom critique
  python -m examples.sms.unified --backend textgrad --optimizer TextualGradientDescent \
      --optimizer-kwargs '{"gradient_memory": 2}' \
      --steps 3 --batch-size 4 --loss "Judge only whether the label is correct." \
      --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1

Add --tracker mlflow --tracking-uri sqlite:///runs/mlflow.db to see it at http://127.0.0.1:5000.
"""

import json

from examples.sms.common import PROBLEM
from examples.sms.common import optimizer_model
from examples.sms.common import setup
from examples.sms.common import target_model
from examples.sms.common import trackers_for
from prompt_optimiser import DSPy
from prompt_optimiser import TextGrad
from prompt_optimiser import optimize

GEPA_DEFAULTS = {"max_metric_calls": 400, "reflection_minibatch_size": 3}


def build_backend(args):
    if args.backend == "dspy":
        optimizer = args.optimizer or "GEPA"
        options = json.loads(args.optimizer_kwargs)
        if not options and optimizer == "GEPA":
            options = dict(GEPA_DEFAULTS)
        return DSPy(
            optimizer=optimizer,
            optimizer_kwargs=options,
            compile_kwargs=json.loads(args.compile_kwargs),
            optimizer_model=optimizer_model(args),
        )
    return TextGrad(
        optimizer=args.optimizer or "TextualGradientDescent",
        optimizer_kwargs=json.loads(args.optimizer_kwargs),
        steps=args.steps,
        batch_size=args.batch_size,
        loss=args.loss,
        constraints=tuple(args.constraint),
        optimizer_model=optimizer_model(args),
    )


def main():
    args, (train, validation, test), metadata = setup(
        "unified",
        lambda parser: (
            parser.add_argument(
                "--optimizer", help="Native class (default: GEPA or TextGrad TGD)"
            ),
            parser.add_argument(
                "--optimizer-kwargs", default="{}", help="JSON passed to the optimiser constructor"
            ),
            parser.add_argument(
                "--compile-kwargs", default="{}", help="JSON passed to DSPy compile()"
            ),
            parser.add_argument("--loss", help="TextGrad critique instruction"),
            parser.add_argument(
                "--constraint",
                action="append",
                default=[],
                help="TextGrad constraint (repeatable)",
            ),
        ),
    )
    result = optimize(
        problem=PROBLEM,
        model=target_model(args),
        backend=build_backend(args),
        train_data=train,
        validation_data=validation,
        test_data=test,
        seed_prompt=args.seed_prompt,
        random_state=args.seed,
        output_dir=args.output,
        trackers=trackers_for(args),
        metadata=metadata,
    )
    print(result.prompt)
    print(json.dumps(result.scores, indent=2))
    print(f"Artifacts: {result.output_dir}")


if __name__ == "__main__":
    main()
