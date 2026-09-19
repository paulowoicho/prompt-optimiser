# Examples

One folder per example. Each is self-contained: its own data, its own metric if it needs one, and a
`run.py` or single script as the entry point. Nothing in here is imported by the library or by other
examples, so adding a new one cannot break an existing one. Run them from the repo root with the
`.venv` interpreter and a served model (see [../docs/VLLM.md](../docs/VLLM.md)).

| Folder | Task | Metric | Shows |
|---|---|---|---|
| [sms/](sms/) | UCI SMS spam, ham/spam labels | exact match | DSPy and TextGrad through one interface; the native scripts they were derived from; a multi-configuration sweep |
| [career_coaching/](career_coaching/) | open-ended coaching answers, no gold labels | part one: a hand-written pairwise AI judge; part two: the same judge calibrated on synthetic preference labels | writing your own metric; why an uncalibrated judge stalls an optimiser; synthetic preference data with explicit label provenance; calibrate, then optimise |
| [custom_backend/](custom_backend/) | tiny inline sentiment task | exact match | a complete third optimiser backend in about thirty lines |
| [preference_judge/](preference_judge/) | MT-bench human preference votes (CC-BY-4.0) | stage 1: agreement with human labels; stage 2: the optimised judge | calibrating an AI judge on human data before using it as a metric; question-level disjoint splits between the two stages |

DSPy examples default to GEPA with an explicit metric-call budget. Select another native
algorithm with `--optimizer` and its constructor options with `--optimizer-kwargs`; TextGrad
remains available through `--backend textgrad`. The native SMS reference scripts show their
original single-tool algorithms.

## Adding an example

Make a folder, put the data and any task-specific metric in it, and call `optimize(...)` from a
`run.py`. Import sibling modules by qualified name (`from examples.<folder>.data import ...`)
after putting the repo root on `sys.path`, as the existing scripts do; plain names like `data`
would collide when two examples are imported in one process. A metric is any
`metric(expected, predicted) -> float`; if it
needs more context than the target string (a judge that wants the question, say), give it that
context when you construct it, as `career_coaching/judge.py` does. Add a row to the table above.
