# Career coaching with an AI judge as the metric

Which system prompt makes a model give better career-coaching answers? There is no gold answer
to a question like "I've been passed over for promotion twice, what's holding me back?", so the
metric is a pairwise judge: a stronger model reads the question and two answers and says which
helps more.

## How it is set up

- **Data**: forty synthetic questions in [data.py](data.py), split 16 / 12 / 12 with a seed.
- **Incumbent**: the seed prompt answers every question once with the model being optimised.
  Those answers are cached (`runs/coaching/incumbent_responses.json`) and become `Example.target`.
- **Metric**: [`PairwiseJudge`](judge.py), defined here, not in the library. It is an ordinary
  `metric(expected, predicted) -> float`. `expected` is the incumbent answer, `predicted` the
  candidate. The judge sees the question plus both answers as a JSON object (data, not
  instructions), in both orderings, and returns one of `A_BETTER`, `B_BETTER`, `BOTH_GOOD`,
  `BOTH_BAD` plus a one-sentence reason. The output is constrained by the server with a JSON
  schema (`VERDICT_SCHEMA` passed as `response_format`). The returned JSON and verdict are still
  validated; truncation or transport errors fail the run. Scores are 1 / 0 / 0.5 / 0.5 by
  default, a tie-adjusted preference score against the seed prompt, where 0.5 covers equal
  quality, both bad, and the judge disagreeing with itself across orderings. The baseline
  compares the seed prompt with itself and should sit near 0.5. It is not exactly 0.5: the backend
  renders the seed in its own format (DSPy wraps it in a signature with field markers, TextGrad
  sends it as a plain system prompt) while the incumbents came from a plain chat call, and serving
  is not perfectly deterministic. The distance from 0.5 measures that effect and is worth
  reading before the final score. A position-biased judge that always says A scores a tie. The
  enum check stays in code so a transport without constrained decoding fails loudly rather than
  scoring a guess; before the schema was added, a live Qwen2.5-72B run failed on exactly that,
  appending prose after its label. One more live failure shaped this: with the schema alone, the
  judge hit the reason's length cap and then emitted a hundred blank lines until the token budget
  ran out, so the run also sets vLLM's `guided_whitespace_pattern` to allow at most one space
  between JSON tokens. Both failures were caught because the metric raises instead of guessing.
- **Judge model**: a different and larger model than the one being optimised. Here Qwen2.5-72B
  judges Llama-3.1-8B. Judging with the same model measures self-preference.
- **Why the judge takes a lookup**: the library's metric never sees the input, only the target and
  the prediction. The judge gets a `{incumbent answer: question}` mapping when it is built. That is
  the whole pattern for "my metric needs context": give it the context at construction time.

Nothing was added to the library for this. The judge, data and runner are three files in this folder.

## Run

```bash
# GEPA (the default): reflective prompt evolution, budgeted by metric calls
python examples/career_coaching/run.py \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \
    --backend dspy --optimizer GEPA \
    --optimizer-kwargs '{"max_metric_calls": 120, "reflection_minibatch_size": 3}' \
    --tracker mlflow --output runs/coaching/dspy-gepa

# MIPROv2 for comparison
python examples/career_coaching/run.py ... --backend dspy --optimizer MIPROv2 \
    --optimizer-kwargs '{"auto": null, "num_candidates": 3, "max_errors": 1}' \
    --compile-kwargs '{"num_trials": 3, "minibatch": false}' \
    --output runs/coaching/dspy-miprov2

python examples/career_coaching/run.py ... --backend textgrad --steps 3 --batch-size 4 \
    --output runs/coaching/textgrad-tgd
```

Any DSPy teleprompter works the same way: name it and pass its constructor arguments. The adapter
fills in only what the constructor declares (metric, task and prompt models, GEPA's
`reflection_lm`, seed). GEPA is a natural partner for a judge: it reflects on per-example
feedback, and its metric may return a score with feedback text; here it gets the score only.
The TextGrad path uses the same criteria as the judge for its textual critique
([loss.py](loss.py), contributed by Codex) and a constraint that the learned prompt stay
reusable, after a smoke run learned a prompt that embedded an answer to one training question.

The run directory holds per-row predictions for every split, `judge_verdicts.jsonl` with every
comparison the judge made and its one-sentence reason, and `judge_counts.json` with the tally.
They are written before the MLflow/W&B upload, so the trackers carry them too. The incumbent
cache is keyed on the target model configuration and seed prompt and regenerates when either
changes.

## Results

**What the judge actually said** (MIPROv2 run, 232 verdicts across all evaluations):
`BOTH_GOOD` 208, `BOTH_BAD` 21, `A_BETTER` 2, `B_BETTER` 1. Ninety-nine percent ties. The judge
finds the 8B's answers to any two coaching prompts interchangeable, so every candidate scores 0.5
and the optimiser keeps the seed. In a direct probe the same judge scored a deliberately bad
answer 0.0 and an off-topic one 0.25, so it can tell bad from worse; what it cannot do is rank
two competent-looking answers. That is a property of this hand-written rubric, and it is the
motivation for the [preference_judge](../preference_judge/) example, which calibrates the judge
on human votes before using it as a metric.

GEPA spent its 120-call budget the same way: 464 verdicts, 439 `BOTH_GOOD`, 12 `BOTH_BAD`, 13
decisive; training minibatch scores never left {0.417, 0.5, 0.583}; seed kept.

The TextGrad run illustrates a second hazard. Its selected prompt edged the seed on validation
by one decisive verdict (0.521) and tied on test, which does not establish a general improvement. Worse, despite the
reusability constraint the learned text reads as a coaching *reply* ("Can you tell me more about
your current role...") and mentions AI's impact on the person's role, lifted from a training
question. Its 280 verdicts were 224 `BOTH_GOOD`, 54 `BOTH_BAD`, 2 `A_BETTER`. With a judge this
indifferent, a textual-gradient optimiser has no signal to steer by and drifts.

These twelve validation and twelve test questions are a small illustrative sample. Judge
disagreement and question sampling add uncertainty; these runs do not establish statistical
significance. One question can move a split's score by at most 0.083.

| Optimised model | Judge | Configuration | Baseline val / test | Final val / test |
|---|---|---|---|---|
| Llama-3.1-8B | Qwen2.5-72B-AWQ | DSPy MIPROv2, 3 candidates, 3 trials | 0.500 / 0.500 | 0.500 / 0.500, kept seed |
| Llama-3.1-8B | Qwen2.5-72B-AWQ | DSPy GEPA, 120 metric calls, reflection minibatch 3 | 0.500 / 0.500 | 0.500 / 0.500, kept seed |
| Llama-3.1-8B | Qwen2.5-72B-AWQ | TextGrad TGD, 3 steps, batch 4, rubric loss + reusability constraint | 0.500 / 0.500 | 0.521 / 0.500 |

## Part two: calibrate the coaching judge, then optimise with it

The judge above was never checked against anyone. Part two applies the
[preference_judge](../preference_judge/) method to coaching. There are no human coaching votes,
so [preference_data.py](preference_data.py) builds a synthetic stand-in and is explicit about
where every label comes from:

- **Questions**: the forty above plus twelve per scenario across ten scenarios (promotion,
  career change, layoff, negotiation, conflict, burnout, returning, early career, late career,
  going independent), generated by the 72B and deduplicated: 160 in total.
- **Constructed pairs, intended-degradation labels**: a strong 72B answer versus a degraded copy
  of itself (specifics stripped into platitudes, an answer to a different question from the same
  split, cut off mid-way, padded with filler, or with an unethical suggestion added). Strong versus
  itself is a tie. The label records the intent of the construction, not a verified ordering: a
  rewrite can fail to degrade, and shorter or longer is not always worse. These pairs test whether
  the judge shares the intended ordering, kind by kind.
- **Natural pairs, committee-labelled**: 8B seed answer versus 72B strong answer, and 8B under two
  prompts, labelled by the 72B reasoning step by step, three independently seeded samples in both
  orderings, strict majority of valid votes, tie otherwise. Invalid votes remain in the saved
  tallies. This is a proxy for human votes, and every table says so.
- **Splits by question id**, stratified by scenario: judge 55 / 25 / 25, response 27 / 14 / 14.
  Judge questions never appear in response optimisation and vice versa.

`calibrate_judge.py` is stage 1 (GEPA by default, `--gepa-feedback` optional) and reports
agreement separately by label source and pair kind. `optimize_with_judge.py` is stage 2: it
reloads the exact calibrated judge, renders the incumbent the way the backend renders
candidates, and optimises the coaching prompt for the 8B on the response questions;
`--judge baseline` is the uncalibrated contrast.

```bash
python examples/career_coaching/calibrate_judge.py \
    --model qwen2.5-72b-instruct-awq --base-url http://127.0.0.1:8123/v1 \
    --small-model llama-3.1-8b-instruct --small-base-url http://127.0.0.1:8124/v1 \
    --tracker mlflow --output runs/coaching_preference/stage1-gepa

python examples/career_coaching/optimize_with_judge.py \
    --judge-run runs/coaching_preference/stage1-gepa --judge best \
    --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --tracker mlflow --output runs/coaching_preference/stage2-gepa-best-judge
```

### Results

#### Stage 1: Qwen2.5-72B judge calibrated with GEPA (400 calls) on the synthetic labels

440 training, 200 validation and 200 test rows (4 pairs per question, both orderings; one test
row is 0.005). Labels are synthetic, as described above.

| | Train | Validation | Test | Test position consistency |
|---|---|---|---|---|
| seed judge | 0.664 | 0.665 | 0.630 | 0.63 |
| GEPA-calibrated judge | 0.664 | 0.695 | 0.630 | 0.81 |

Agreement on the test rows by label source and by pair kind, seed → calibrated:

| Source / kind | Seed | Calibrated |
|---|---|---|
| committee pairs (all) | 0.659 | **0.750** |
| 8B seed answer vs 72B strong answer | 0.679 | 0.679 |
| 8B seed answer vs 8B bare-prompt answer | 0.625 | 0.875 |
| constructed pairs (all) | 0.622 | **0.596** |
| strong vs truncated | 1.00 | 1.00 |
| strong vs off-topic | 0.95 | 1.00 |
| strong vs itself (tie) | 1.00 | 1.00 |
| strong vs generic rewrite | 0.176 | **0.088** |
| strong vs padded | 0.464 | **0.321** |
| strong vs unethical rewrite | 0.318 | 0.409 |

What happened: GEPA rewrote the judge's instructions (no demonstrations). The calibrated judge is
far more self-consistent across orderings and agrees more with the committee, and it less often
distinguishes the original from two of the intended degradations. On the held-out rows neither
judge ever picks the padded or the generic rewrite over the original; what changes is ties. For
the 28 padded rows the original wins 13 → 9 and ties go 15 → 19; for the 34 generic rows the
original wins 6 → 3 and ties go 28 → 31. The generic rewrite is subtle (same structure and length,
only the specifics removed: "accounting → your new field"), which may help explain the high
tie rate. Overall test agreement is flat because the committee gain (29 → 33 of 44 rows) and the
constructed loss (97 → 93 of 156) cancel exactly at 126 of 200.

Two limitations matter here. The committee preferred the 8B's seed-prompt answers
over the 72B's concise "strong" answers 117 to 29, and in 159 of 160 such pairs the 8B answer is
the longer one (about 2,900 versus 1,500 characters), so a length preference is plausible; but
length is confounded with model, prompt and content, and the calibrated judge did not start
preferring padded text, it stopped separating it. Second, the committee itself produced 236
invalid votes out of 1,920 (a reasoning answer that did not end in a label), so its majorities
are over the valid votes only: 155 pairs had no invalid vote, 110 had one, 43 two, 9 three, 2
four and 1 five, so 12 pairs were decided on three or fewer valid votes and one on a single vote.
The calibrated judges produced no invalid outputs.

Read the numbers as 25 held-out question groups, not 200 independent observations. The
conclusion is narrow: better committee agreement and position consistency did not improve
overall held-out agreement and do not establish better coaching judgement. Degradation labels
are heuristic; independent human labels would give a target that is not another model's opinion,
which is different from guaranteeing an unbiased judge. Reporting agreement by label source is
what made any of this visible, and it is the part worth copying.

#### Stage 2

Pending: GEPA response optimisation with the calibrated and the baseline judge.
