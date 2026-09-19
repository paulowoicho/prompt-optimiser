# Status

The design discussion and work log live in [COLLABORATION.md](COLLABORATION.md).
This file is the one-screen summary.

## Goal

Optimise a prompt with data towards a metric, choosing freely between optimisation
tools (DSPy, TextGrad, or your own) and, within each tool, between its optimisers.
The library provides the interface and the experiment lifecycle. It does not choose
the algorithm; the experimenter does.

## Done (2026-09-18)

- One entry point, `optimize(...)`, with one extension contract, `OptimizerBackend.fit(...)`.
- The earlier candidate-search prototype (`PromptOptimizer`, `SearchContext`, textual
  criteria, `ModelJudge`) is deleted. Its source is preserved in `runs/backups/prototype-distribution-20260917.tar.gz`.
- Public API reduced from 27 names to 17.
- Fresh default output directory per run; `status.json` is final before trackers upload.
- Local vLLM serving of Llama-3.1-8B-Instruct and Qwen2.5-72B-Instruct-AWQ; see [docs/VLLM.md](docs/VLLM.md).

- Native adapters take the optimiser, program, loss and kwargs from the experimenter;
  one adapter per tool (`DSPy`, `TextGrad`), no presets.
- README rewritten around the generic adapters, with the SMS results table.
- Nine optimiser runs on the 8B and 72B models plus the earlier 1.5B run, tabulated in the
  README; both agents cross-reviewed each other's code and docs.
- A third backend written as a thirty-line example (`examples/custom_backend/rewrite_search.py`), under test.
- First commits `5feb93d`, `de49e04` (2026-09-18; history rewritten to exclude coordination records).

- Career-coaching example: example-local AI judge and TextGrad feedback; MIPROv2, GEPA and
  TextGrad runs recorded. Examples organised one folder per task.

- MT-Bench preference-judge example: train the judge on human labels, then use its native
  predictor as a response-optimisation metric. Both stages and live runs recorded. Stage 2
  inherits stage-1 splits and judge settings; regression tests cover native rendering,
  demonstration retention and reports uploaded before trackers close.

- GEPA leads the examples; the core remains optimiser-agnostic. SMS GEPA and both
  preference-judge stages ran on local vLLM models, with results and limitations documented.
  Score-only GEPA beat the 400-call feedback variant on judge-validation; the two stage-2
  response runs and their cross-scoring were reviewed against saved artifacts.

## In progress

- Coaching judge calibration on 160 synthetic questions: data construction, seeded committee
  labels, disjoint judge/response splits, saved judge reload and tracker reports are implemented
  and reviewed. Live committee labelling, calibration and response comparisons are in progress.
- Private GitHub repository created and the reviewed commits pushed to the user's account.
- Shared-board coordination with Claude remains active. The earlier MT-Bench 1500-call feedback
  run tied validation agreement and lost the consistency tie-break; its selected judge is unchanged.

## Later

- Hard call and token budgets, cancellation, resumability.
- Repeated seeds and uncertainty on the reported scores.
- Batch metrics such as F1 and richer input/output structures.
