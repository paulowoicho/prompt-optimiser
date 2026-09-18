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
- Seven optimiser configurations run on the 8B and 72B models; both agents cross-reviewed
  each other's code and docs.

## In progress

- Remaining 72B TextGrad run; final results table.

## Later

- Hard call and token budgets, cancellation, resumability.
- Repeated seeds and uncertainty on the reported scores.
- Batch metrics such as F1 and richer input/output structures.
