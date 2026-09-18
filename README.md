# prompt-optimiser

Optimise a prompt with data towards a metric. Pick the tool (DSPy, TextGrad, or your own)
and, within the tool, the algorithm. The library owns the experiment: data checks, held-out
evaluation, artifacts and tracking. It does not own the optimisation recipe. You do.

```python
from prompt_optimiser import DSPy, TextGrad, VLLM, optimize, exact_match

model = VLLM(model="llama-3.1-8b-instruct", base_url="http://127.0.0.1:8124/v1", max_tokens=1024)

result = optimize(
    problem="Classify an SMS message as ham or spam. Return exactly ham or spam.",
    model=model,
    backend=DSPy(optimizer="MIPROv2", optimizer_kwargs={"auto": "light"}),
    train_data=train,            # Example(input, target) or {"input": ..., "target": ...}
    validation_data=validation,  # optional; 20% of train is held out if omitted
    test_data=test,              # optional; never shown to the optimiser
    metric=exact_match,          # metric(expected, predicted) -> float, the objective
    seed_prompt=None,            # defaults to the problem description
    output_dir="runs/sms-mipro", # new or empty; a fresh runs/<Backend>-<time> dir if omitted
    trackers=[],                 # MLflowTracker(...), WandbTracker(...), ConsoleTracker()
)
result.prompt                        # readable instructions
result.scores["final"]["test"]       # {"score": ..., "n_examples": ...}; also "baseline", "train", "validation"
result.predict(["WIN A FREE PRIZE"]) # the exact native predictor that was scored
```

Swap `backend=` and nothing else changes (`import dspy` for the class-based forms):

```python
DSPy(optimizer="COPRO", optimizer_kwargs={"breadth": 4, "depth": 2})
DSPy(optimizer="BootstrapFewShotWithRandomSearch",
     optimizer_kwargs={"max_bootstrapped_demos": 4, "num_candidate_programs": 4})
DSPy(optimizer=dspy.GEPA, optimizer_kwargs={"auto": "light"}, optimizer_model=big_model)
DSPy(optimizer="MIPROv2", program=dspy.ChainOfThought)
TextGrad(optimizer="TextualGradientDescent", steps=3, batch_size=4)
TextGrad(steps=3, loss="Judge only whether the label is correct and how to fix the instructions.")
TextGrad(steps=3, constraints=("Keep the prompt under 60 words",), optimizer_model=big_model)
```

## Install

```bash
uv venv --python 3.12
uv pip install -e '.[dev,dspy,textgrad,llm,mlflow,wandb]'
```

Every integration is optional and imported only when used. The core has no runtime
dependencies. Flat package, no `src/`. The `mlflow` extra installs `mlflow-skinny`, which
logs but has no UI; for the browser UI add the full package: `uv pip install 'mlflow>=3,<4'`.

## The pieces

**`optimize(...)`** validates the data (no empty splits, no input overlap, duplicates kept
together when it splits validation off), calls the backend once with train and validation
only, then re-evaluates the baseline and the selected prompt on every split with your metric.
It writes the directory below, sends progress to trackers, and returns an `ExperimentResult`.

**`metric(expected, predicted) -> float`** is the objective. Default `exact_match` is a
trimmed, case-sensitive comparison. Pass `greater_is_better=False` for a loss. DSPy receives
this metric natively. TextGrad uses it to select and score prompts; its training signal is
its own textual critique (see `loss` below).

**`DSPy(...)`** compiles a `dspy.Predict("text -> answer")` program with whatever
teleprompter you name. `optimizer` is a class, an exported name (`"MIPROv2"`, `"COPRO"`,
`"GEPA"`, `"SIMBA"`, `"BootstrapFewShot"`, `"BootstrapFewShotWithRandomSearch"`,
`"LabeledFewShot"`, `"KNNFewShot"`, ...) or a preconfigured instance. `optimizer_kwargs`
go to the constructor and `compile_kwargs` to `compile()`, untouched. The harness fills in
only what the constructor actually declares: the metric, task and prompt models, seed, and
`valset` when `compile()` accepts one. `program` may be `dspy.ChainOfThought`, another
factory taking a signature, or a `dspy.Module` instance. `optimizer_model` sets a
different proposer.

**`TextGrad(...)`** runs the native loop: forward, textual loss, backward, optimiser step,
then validates and keeps the best prompt. `optimizer` is `"TextualGradientDescent"`, another
exported name, or a factory; `optimizer_kwargs` and `constraints` pass through. `loss` is
`None` for a reference-answer critique, a string with your own critique instruction, or a
callable `loss(response, example, engine) -> tg.Variable` that builds any native
differentiable loss. A malformed update from the model raises with an `error` event
rather than being skipped. Known upstream defect: the installed TextGrad's
`TextualGradientDescentwithMomentum` never returns its update prompt, so it cannot run
until fixed upstream. Standard TGD with gradient memory works.

**`VLLM(model, base_url, ...)`** is one endpoint configuration translated to each tool's
model interface. Any OpenAI-compatible server works. Plain provider strings
(`"openai/gpt-4o-mini"`) and native `dspy.LM` or TextGrad engines are accepted too.
Serving commands used here are in [docs/VLLM.md](docs/VLLM.md).

**Trackers.** `MLflowTracker(experiment, tracking_uri=...)`, `WandbTracker(project, ...)`,
`ConsoleTracker()`, `JSONLTracker(path)`. They receive the same `Event`s through `log()`:
`start` with the full config, one `candidate` per native evaluation with its prompt and score,
`error` when a native step fails, `artifact` with the run directory (also on failure, so the
partial run is diagnosable), and `finish` with all scores. `close(status)` is then called with
`finished` or `failed`. MLflow and W&B store the run directory, including native program state.

```bash
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db --port 5000   # then http://127.0.0.1:5000
wandb sync wandb/offline-run-*                                        # for offline W&B runs
```

## What a run leaves behind

```
runs/<name>/
  config.json        task, model, backend settings (nested, secrets redacted), versions, sizes
  progress.jsonl     every event the trackers saw
  prompt.txt         the selected readable instructions
  baseline/ best/    native state (DSPy program.json, TextGrad prompt.txt) and messages.json
  predictions/       <phase>-<split>.jsonl with input, target, prediction, score per row
  result.json        baseline and final scores for train, validation, test
  status.json        finished or failed
```

A DSPy program's instruction text is not its whole prompt: `best/messages.json` shows the
rendered template with signature and demonstrations, and `result.predict()` runs the saved
program itself. Baselines can differ between tools because their message formatting differs.

## Add another optimisation tool

Implement one method, no inheritance, no registry:

```python
class MyOptimizer:
    def fit(self, *, problem, model, seed_prompt, train, validation, metric,
            greater_is_better, random_state, report) -> FitResult:
        ...  # run the tool's own loop; call report(Event("candidate", step, {...}, {...})) as it goes
        return FitResult(baseline=Prompt(text, predict_one, save_native),
                         best=Prompt(text, predict_one, save_native))
```

The backend never receives test data. Each `Prompt` carries the readable instructions, the
exact native predictor, and optionally a function that saves native state into a directory
(default: write `prompt.txt`). Accept `**request` to ignore arguments you do not need. To call
the served model without knowing DSPy or TextGrad, use `model.as_litellm()` on a `VLLM`, which
gives `model(system_prompt, text) -> str`. [examples/custom_backend/rewrite_search.py](examples/custom_backend/rewrite_search.py)
is a complete working backend in about thirty lines and is exercised by the test suite. The
contract is in [prompt_optimiser/experiment.py](prompt_optimiser/experiment.py); the DSPy and
TextGrad adapters in [prompt_optimiser/native/](prompt_optimiser/native/) are the real examples.

## Examples

One folder per task under [examples/](examples/README.md): `sms/` (labels, exact match, the sweep),
`career_coaching/` (open-ended answers scored by a pairwise AI judge written inside the example),
`custom_backend/` (a third optimiser in thirty lines), and `preference_judge/` (train a judge
on MT-Bench human votes, then use it as the metric). Each is self-contained.

### Your own metric

A metric is any `metric(expected, predicted) -> float`. Nothing needs registering. If it needs more
than the target string, give it that context when you build it. The career-coaching example does
this with an AI judge: the seed prompt answers every question once, those answers become the
targets, and `PairwiseJudge` (defined in
[examples/career_coaching/judge.py](examples/career_coaching/judge.py), not in the library) compares
a candidate answer with the incumbent for the same question in both orderings and returns a
tie-adjusted preference score (1 win, 0 loss, 0.5 tie). The verdict is constrained to a JSON
schema by the server and checked by the example. Report the measured baseline: backend
formatting and serving variation can change the seed responses. Higher scores mean more
preference from this judge, rather than independently established response quality.

## The SMS experiment

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
Almeida & Hidalgo (2011), CC BY 4.0. `examples/sms/data.py` normalises whitespace,
removes duplicates and conflicting labels, and makes class-balanced disjoint splits, so
accuracy here does not reflect the natural spam rate. Default experiment: 10 train, 20
validation and 20 test messages per class, seed 42.

```bash
python examples/sms/unified.py --backend dspy --optimizer MIPROv2 \
    --optimizer-kwargs '{"auto": null, "num_candidates": 3}' --compile-kwargs '{"num_trials": 3, "minibatch": false}' \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --train-per-class 10 --eval-per-class 20 --tracker mlflow --tracking-uri sqlite:///runs/mlflow.db
python examples/sms/unified.py --backend textgrad --optimizer TextualGradientDescent --steps 3 --batch-size 4 \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 --train-per-class 10 --eval-per-class 20
```

`examples/sms/native_dspy.py` and `examples/sms/native_textgrad.py` are the plain
single-tool scripts the adapters were derived from; read them to see what each tool does
without the harness.

### Results so far

Same fixed splits (10 train, 20 validation, 20 test per class, seed 42), exact match, proposer
or critic is the target model unless stated. Selection is on validation; the baseline is kept
on ties. Run directories are under `runs/live-*` (Claude) and `runs/review-8b` (Codex); each
holds per-row predictions, native state and the MLflow run.

| Model | Tool | Configuration | Baseline val / test | Final val / test | Outcome |
|---|---|---|---|---|---|
| Llama-3.1-8B | DSPy | MIPROv2, 3 candidates, 3 trials, native proposer and demonstration defaults | 0.900 / 0.875 | 0.950 / 0.925 | improved |
| Llama-3.1-8B | DSPy | MIPROv2, small zero-shot preset: no demonstrations, data-aware proposer only, whole train set as data summary | 0.900 / 0.875 | 0.900 / 0.875 | kept seed |
| Llama-3.1-8B | DSPy | COPRO, breadth 3, depth 2 | 0.900 / 0.875 | 0.900 / 0.875 | kept seed; candidate 0.825 |
| Llama-3.1-8B | DSPy | LabeledFewShot, k=4 | 0.900 / 0.875 | 0.900 / 0.875 | tied, kept seed |
| Llama-3.1-8B | TextGrad | TGD, 3 steps, batch 2 | 0.900 / 0.900 | 0.900 / 0.900 | kept seed |
| Llama-3.1-8B | TextGrad | TGD, 3 steps, batch 4 | 0.900 / 0.900 | 0.925 / 0.875 | val up, test down |
| Llama-3.1-8B | TextGrad | TGD with gradient memory 2, 3 steps, batch 2 | 0.900 / 0.900 | 0.900 / 0.900 | kept seed; last candidate 0.250 |
| Qwen2.5-72B-AWQ | DSPy | MIPROv2, 3 candidates, 3 trials | 1.000 / 0.975 | 1.000 / 0.975 | no headroom |
| Qwen2.5-72B-AWQ | TextGrad | TGD, 3 steps, batch 4 | 0.975 / 1.000 | 0.975 / 1.000 | kept seed; candidates 0.925, 0.950, 0.975 (tie) |
| Qwen2.5-1.5B (earlier) | DSPy | MIPROv2 preset | 0.525 / 0.500 | 0.525 / 0.500 | predictions not saved |

Forty examples per split means one example is 0.025. None of these differences is
statistically strong; the table shows the harness working across tools, algorithms and models,
and that configuration choices within one tool change the result. A configuration that does
not help is a result about that configuration on that model, not about the tool. TextGrad's
baseline differs from DSPy's on the same model because their message formatting differs.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Status and open items: [PLAN.md](PLAN.md). Working log: [COLLABORATION.md](COLLABORATION.md).
