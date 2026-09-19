# Examples

One folder per experiment. Each is self-contained, with its own data and, where it needs one,
its own metric; nothing here is imported by the library. Run them as modules from the
repository root against any OpenAI-compatible model endpoint:

```bash
python -m examples.sms.unified --help
```

- `sms/` — UCI SMS spam with exact match. `unified.py` runs any DSPy or TextGrad optimiser
  through one interface, `sweep.py` runs several in a row, and the two `native_*.py` scripts
  are the single-tool originals the adapters were derived from.
- `custom_backend/` — a third optimiser in thirty lines, on a tiny inline sentiment task.
- `career_coaching/` — an AI judge as the metric on open-ended answers, then the same judge
  calibrated on synthetic preference labels.
- `preference_judge/` — a judge calibrated on MT-bench human votes, then used to optimise
  responses, with question-level disjoint splits.

DSPy scripts default to GEPA with an explicit metric-call budget; `--optimizer` and
`--optimizer-kwargs` select any other native algorithm, and `--backend textgrad` switches tool.

To add one: make a folder, put the data and any task-specific metric in it, call
`optimize(...)` from a `run.py`, and import siblings by full name
(`from examples.<folder>.data import ...`). A metric that needs more than the target string,
such as a judge that wants the question, gets that context when it is constructed, as
`career_coaching/judge.py` does.
