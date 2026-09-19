# prompt-optimiser

Optimise a prompt against data and a metric. You choose the tool (DSPy, TextGrad, or one of
your own) and, within it, the algorithm and its settings. The library runs the experiment:
it checks the data, calls the tool's own training loop, evaluates on held-out data, and
writes the artifacts and tracking.

## Install

Python 3.10+ and `uv`.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,dspy,textgrad,llm,mlflow,wandb]'
```

Every integration is optional; the core has no runtime dependencies.

## Use

```python
from prompt_optimiser import DSPy, VLLM, exact_match, optimize

model = VLLM("llama-3.1-8b-instruct", base_url="http://127.0.0.1:8124/v1")

result = optimize(
    problem="Classify an SMS message as ham or spam. Return exactly ham or spam.",
    model=model,
    backend=DSPy(optimizer="GEPA", optimizer_kwargs={"max_metric_calls": 400}),
    train_data=train,
    validation_data=validation,
    test_data=test,
    metric=exact_match,
    output_dir="runs/sms-gepa",
)
result.prompt
result.scores["final"]["test"]
result.predict(["WIN A FREE PRIZE"])
```

`train_data` is a list of `Example(input, target)`. If `validation_data` is omitted, a fifth
of the training inputs is held out. Test data never reaches the optimiser.

The backend is the only line that changes between experiments. `DSPy(optimizer=...)` takes
any DSPy teleprompter by name or class, with `optimizer_kwargs` for its constructor and
`compile_kwargs` for `compile()`. `TextGrad(...)` takes the optimiser, the loss (an
instruction string or a callable that builds a native loss) and constraints. Nothing about
the algorithm is fixed by the library.

`model` is any OpenAI-compatible endpoint through `VLLM(name, base_url=...)`, a LiteLLM
provider string, or a native DSPy or TextGrad model object. `metric` is any
`metric(expected, predicted) -> float`; the examples show AI judges written as ordinary
metrics. TextGrad trains on textual feedback from its loss; the numeric metric selects and
reports prompts. Trackers are optional: `MLflowTracker`, `WandbTracker`, `ConsoleTracker`,
`JSONLTracker`.

## What a run leaves behind

```
runs/<name>/
  config.json      task, model and backend settings, versions, split sizes
  progress.jsonl   the evaluations and updates the optimiser reported, with scores
  prompt.txt       the selected instructions
  baseline/ best/  native state (DSPy program.json, TextGrad prompt.txt) and rendered messages
  predictions/     per-row predictions and scores for every split
  result.json      baseline and final scores
```

A DSPy program's instruction text is not the whole prompt: demonstrations and formatting are
in `best/program.json`, and `result.predict()` runs that program.

## Adding an optimiser

Implement one method. No base class, no registry.

```python
from prompt_optimiser import FitResult, Prompt


class MyOptimizer:
    def fit(self, *, problem, model, seed_prompt, train, validation, metric,
            greater_is_better, random_state, report) -> FitResult:
        ...
        return FitResult(Prompt(seed_prompt, predict_seed), Prompt(best_text, predict_best))
```

`examples/custom_backend/rewrite_search.py` is a complete example.

## Examples

Each folder under `examples/` is self-contained and runs as a module from the repository
root, for instance `python -m examples.sms.unified --help`.

- `sms/` — spam classification with exact match; a sweep over several optimisers, with results
- `custom_backend/` — a third optimiser plugged in without touching the library
- `career_coaching/` — an AI judge as the metric, then the same judge calibrated on
  synthetic preference labels
- `preference_judge/` — a judge calibrated on MT-bench human votes, then used to optimise
  responses, with question-level disjoint splits

Experiment results and their limits are in the SMS, coaching and preference-judge READMEs.

## Repository structure

- `prompt_optimiser/experiment.py` — `optimize()` and the `OptimizerBackend` contract
- `prompt_optimiser/native/` — the DSPy and TextGrad adapters
- `prompt_optimiser/tracking.py` — trackers
- `prompt_optimiser/vllm.py` — one endpoint configuration for every tool
- `examples/` — experiments
- `tests/` — pytest suite

## Development

```bash
ruff format --check && ruff check
python -m pytest tests
```
