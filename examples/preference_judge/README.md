# Calibrate the judge first, then optimise with it

The [career-coaching example](../career_coaching/) showed the failure mode of an uncalibrated AI
judge: a hand-written rubric called most pairs BOTH_GOOD and rarely distinguished the answers.
This example tests whether training the judge on human votes provides a more useful metric.
It uses one open dataset with disjoint question splits across two stages.

**Dataset**: [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments),
CC-BY-4.0. 3,355 votes by 65 expert human judges over the 80 MT-bench questions (8 categories),
covering both conversation turns and six models, with ties. We use the 1,689 first-turn votes.
Votes on the same pair are aggregated by plurality (the most votes); equal top counts become TIE. [data.py](data.py) does the download, aggregation and splits.

## Splits, by question id

| Split | Questions | Used for |
|---|---|---|
| judge_train | 24 | stage 1 training |
| judge_validation | 8 | stage 1 candidate selection |
| judge_test | 8 | stage 1 held-out report: agreement with humans, position consistency |
| response_train | 16 | stage 2 training |
| response_validation | 8 | stage 2 candidate selection |
| response_test | 16 | stage 2 held-out report |

Stratified by category, 10 questions each. A question is in exactly one split. The judge is
never scored on a question it trained on, and the response optimiser never sees a judge-test
question. Stage 2 reads the split assignment saved by stage 1; changing its optimisation seed
does not change that assignment. Every pair appears in both orderings with the label swapped, so a judge with a
position preference loses points in training and is measured in evaluation.

## Stage 1: optimise the judge ([stage1_judge.py](stage1_judge.py))

Install the example dataset dependencies with `pip install pandas pyarrow huggingface_hub`.

Proposals and TextGrad updates use a separate 2,048-token budget; the judge still uses
`--max-tokens` (16 by default) for labels. `--optimizer-model` optionally changes the proposer.

The judge is an ordinary prompt: JSON `{question, response_a, response_b}` in, one of
`A_BETTER` / `B_BETTER` / `TIE` out, metric = agreement with the human label. Any optimiser works.

```bash
# GEPA (the default): evolves the instructions by reflecting on per-example feedback.
python examples/preference_judge/stage1_judge.py \
    --model qwen2.5-72b-instruct-awq --base-url http://127.0.0.1:8123/v1 \
    --backend dspy --optimizer GEPA \
    --optimizer-kwargs '{"max_metric_calls": 400, "reflection_minibatch_size": 3, "num_threads": 4}' \
    --tracker mlflow --output runs/preference_judge/stage1-gepa

# Add --gepa-feedback with a fresh --output directory to compare textual feedback.

# MIPROv2 for comparison: searches instructions and demonstrations by Bayesian optimisation.
python examples/preference_judge/stage1_judge.py ... --optimizer MIPROv2 \
    --optimizer-kwargs '{"auto": null, "num_candidates": 3, "max_errors": 1}' \
    --compile-kwargs '{"num_trials": 3, "minibatch": false}' --output runs/preference_judge/stage1-miprov2
```

The GEPA feedback metric lives in the example (`gepa_feedback_metric`) and reaches GEPA through
`DSPy(optimizer_kwargs={"metric": ...})`; the harness still selects and reports with plain
agreement. That is the pass-through the adapter is built for: the experimenter decides what the
optimiser sees.

`report.json` in the run directory gives baseline and optimised agreement with humans on
judge-test, plus position consistency (does the judge flip its answer when A and B swap) and the
predicted label distribution, computed from the saved per-row predictions.

## Stage 2: optimise responses with the calibrated judge ([stage2_responses.py](stage2_responses.py))

Reloads the exact judge that stage 1 scored (DSPy `best/program.json` or TextGrad
`best/prompt.txt`), wraps it in a pairwise metric with position swap and 1 / 0 / 0.5 scores, and
optimises a response prompt for a smaller model on the response questions, with the seed
prompt's answers as the incumbent. `--judge baseline` reloads the same run's unoptimised judge
instead, so the two metrics can be compared on identical data. The judge model and sampling
settings come from stage 1; `--judge-base-url` can point to its new endpoint. Invalid judge
output (a non-label) scores a tie and is counted in `judge_counts.json`; pass
`--on-invalid error` to make it fatal instead.

```bash
python examples/preference_judge/stage2_responses.py \
    --judge-run runs/preference_judge/stage1-gepa --judge best \
    --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --backend dspy --optimizer GEPA \
    --optimizer-kwargs '{"max_metric_calls": 400, "reflection_minibatch_size": 3, "num_threads": 4}' \
    --tracker mlflow --output runs/preference_judge/stage2-gepa-best-judge
```

Swap `--optimizer MIPROv2` (with its own kwargs) or `--backend textgrad` to compare optimisers on
the same judge and questions.

## Results

### Stage 1: Qwen2.5-72B-Instruct-AWQ as judge, DSPy MIPROv2 (3 candidates, 3 trials)

Agreement with the aggregated human label, 6 pairs per question, both orderings (96 rows on
validation and on test; one row is 0.010).

| Split | Baseline | Optimised |
|---|---|---|
| judge_train (288 rows) | 0.628 | 0.663 |
| judge_validation (96) | 0.510 | 0.531 |
| judge_test (96, held out) | 0.521 | 0.656 |

On judge-test the optimised judge agrees with humans on 63 of 96 rows against 50 for the seed
prompt. Position consistency (does the verdict flip when A and B swap) went from 0.77 to 0.83 over
the 48 pairs. The seed judge almost never said TIE (6 of 96 predictions) while humans tied 40 of
96; the optimised judge says TIE 23 times. Uniform random guessing would score 0.33; always
choosing TIE would score 40/96 (0.417) on this test set. In the full dataset, 105 of 910 pairs
have equal top vote counts; another 30 have TIE as the unique top label despite some disagreement.
The original run metadata grouped all 135 as `majority_overridden_to_tie`; the corrected summary
separates tied vote counts from non-unanimous TIE labels. Aggregated labels and run scores are unchanged.

Three things worth reading before trusting this:

- **The instructions did not change.** `prompt.txt` is identical to the seed. MIPROv2's selected
  program differs from the baseline by four bootstrapped demonstrations from judge-train
  questions (`best/program.json`, rendered in `best/messages.json` as a ten-message template).
  This is the README's standing caveat made concrete: a DSPy program is more than its instruction
  text, and `result.predict()` runs the real thing. Stage 2 must reload the program, not the text.
- **One proposal scored 0.0.** The recorded run shared the 16-token judge budget with its
  proposer; the runner now gives the proposer a separate budget. The proposer emitted an instruction truncated to "You are an
  impartial judge of answers to a user", and with it the judge stopped producing bare labels on
  every row. The metric gave it zero, as it should; it was never a candidate for selection.
- **One invalid output on test.** The optimised judge answered one row with the start of a JSON
  object instead of a label, scored 0. `report.json` counts these.

Validation (8 questions) and test (8 questions) disagree on the size of the gain, 0.02 versus
0.13. The 96 rows are correlated: they contain both orderings of 48 pairs from only eight
questions. An independent-row binomial standard error would understate uncertainty. These are
observed gains on this split, not an established general improvement; repeated seeds and
question-level uncertainty estimates are needed for that claim.

### Stage 1 with GEPA (the example default)

Same splits and rows. GEPA rewrites the instructions by reflecting on per-example results; it
attached no demonstrations.

| Judge | Validation | Test | Test position consistency | What changed |
|---|---|---|---|---|
| seed | 0.510 | 0.521 | 0.77 | |
| MIPROv2, 3×3 | 0.531 | 0.656 | 0.83 | 4 demonstrations, instructions unchanged |
| GEPA, score only, 400 calls | 0.531 | 0.562 | 0.71 | instructions rewritten: five spelled-out criteria plus worked examples from judge-train questions |
| GEPA, `--gepa-feedback`, 400 calls | 0.510 | 0.521 | 0.77 | kept the seed: 3 new candidates received full-validation checks, each costing 96 calls |
| GEPA, `--gepa-feedback`, 1500 calls | running | | | |

**Selection was on validation only.** MIPROv2 and score-only GEPA tie at 0.531, so a tie-break was
declared before being applied: position consistency on the *validation* predictions. GEPA 0.854
with no invalid outputs, MIPROv2 0.708 with three. The GEPA judge therefore goes to stage 2. Its
test agreement is lower than MIPROv2's; that is reported, not acted on, because acting on it
would make judge-test a selection split. The feedback variant's budget note matters: GEPA scores
every accepted candidate on the whole validation set, so 400 calls allows few full-validation checks; the
budget is the experimenter's knob, not a verdict on textual feedback.

### Stage 2: Llama-3.1-8B response prompt, DSPy MIPROv2 (3 candidates, 3 trials)

Tie-adjusted preference against the seed prompt's own answers on 16 / 8 / 16 held-out questions
(one question contributes at most 0.125 on validation and 0.0625 on test). The incumbent is rendered through the same DSPy seed
program as the candidates. Every baseline in these recorded runs scored 0.5; serving
nondeterminism can still produce differences on reruns.

| Judge used as the metric | Final val / test (own judge) | Selected program |
|---|---|---|
| calibrated (stage 1 `best`) | 0.656 / **0.547** | new instructions + 4 demonstrations |
| uncalibrated (stage 1 `baseline`) | 0.688 / **0.438** | seed instructions + 4 demonstrations |

The uncalibrated judge reported the larger validation gain and selected a program that scores
*below* the seed on test by its own judgement. The calibrated judge's selection scores slightly
above the seed in this test run.
Cross-scoring both selected programs' saved test answers with both judges
(`runs/preference_judge/stage2-v2-crosscheck.json`):

| Selected by | Scored by calibrated judge | Scored by uncalibrated judge |
|---|---|---|
| calibrated judge | 0.547 | 0.516 |
| uncalibrated judge | 0.484 | 0.438 |

Both judges score the program selected by the trained judge above the seed and the other
program below it on these sixteen test questions. These small observed gaps do not establish
that judge training improves response optimisation generally, and neither scorer supplies
human preference labels for the new responses.

### Stage 2 with GEPA (the example default), judge from the GEPA stage 1

GEPA as the response optimiser, 400 metric calls, reflection minibatch 3, same incumbents.

| Judge used as the metric | Final val / test (own judge) | Selected program |
|---|---|---|
| calibrated (`stage1-gepa/best`) | 0.750 / **0.656** | rewritten instructions, no demonstrations |
| uncalibrated (`stage1-gepa/baseline`) | 0.688 / **0.547** | rewritten instructions, no demonstrations |

Cross-scoring both selected programs' saved test answers with both judges
(`runs/preference_judge/stage2-gepa-crosscheck.json`), independently rechecked from the saved
responses; per-question scores, verdicts and artifact hashes are in
`runs/preference_judge/stage2-gepa-crosscheck-reviewed.json`:

| Selected by | Scored by calibrated judge | Scored by uncalibrated judge |
|---|---|---|
| calibrated judge | 0.656 | 0.672 |
| uncalibrated judge | 0.594 | 0.547 |

Both judges assign higher mean scores to the program selected using the trained judge in this
GEPA comparison. The trained judge differs from the one used in the MIPROv2 comparison, so
these runs do not isolate the effect of the response optimiser. The sample-size and lack-of-human-
evaluation caveats above still apply. One more to read in the artifacts: both GEPA prompts open with a
list of task domains taken from the training questions ("designing a seismically resilient
bridge", "highest common ancestor of two nodes"). They still improved answers on the disjoint
test questions, but a prompt that enumerates its training set is a prompt to watch as sets grow.

**Attempt 1 measured formatting, not prompts.** The first stage-2 runs generated the incumbent
answers with a plain chat call while DSPy renders every candidate, the unchanged seed included,
through a signature with field markers. Under that rendering the 8B answered about 40% shorter
(mean 1,029 vs 1,719 characters) and both judges preferred the longer chat answers, so the
"baseline" scored 0.25 and 0.16 on test instead of 0.5. `seed_responder` now renders the incumbent
the way the backend will render candidates. The attempt-1 directories
(`runs/preference_judge/stage2-miprov2-*`) are kept; in them the uncalibrated judge also selected
the proposer's own meta-text as the instruction, a failure the calibrated judge did not make.

Both stages write `report.json` before MLflow/W&B upload their artifacts. Judge verdict counts
combine training, selection and final evaluation; use the per-split prediction files for held-out scores.
