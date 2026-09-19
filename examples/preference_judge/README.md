# Preference judge

Calibrate an AI judge against human votes, then use it as the metric for optimising
responses, on one dataset with splits arranged so nothing leaks between the two stages.

Data is `lmsys/mt_bench_human_judgments` (CC-BY-4.0): 3,355 votes by 65 expert judges over the
80 MT-bench questions on answers from six models, with ties; the 1,689 first-turn votes are
used, giving 910 distinct pairs. Votes on the same pair are merged by plurality, and a tied top
vote becomes a tie. `data.py` downloads, aggregates and splits by question id, stratified over
the eight categories: 24 / 8 / 8 questions for training, selecting and testing the judge (288 /
96 / 96 rows at six pairs per question) and 16 / 8 / 16 for optimising responses. Every pair
appears in both orderings with the label swapped.

## Stage 1: optimise the judge

The judge is a prompt like any other: JSON `{question, response_a, response_b}` in, one of
`A_BETTER` / `B_BETTER` / `TIE` out, metric is agreement with the human label.

```bash
python -m examples.preference_judge.stage1_judge \
    --model qwen2.5-72b-instruct-awq --base-url http://127.0.0.1:8123/v1 \
    --optimizer GEPA --output runs/preference_judge/stage1-gepa
```

`--gepa-feedback` hands GEPA a native metric that also says "humans said X, you said Y";
`--optimizer MIPROv2` with its own kwargs is the comparison. `report.json` records agreement
and position consistency on the held-out questions.

Qwen2.5-72B-Instruct-AWQ as the judge, 96 rows in each evaluation split:

| Judge | Validation | Test | Test position consistency | What changed |
|---|---|---|---|---|
| seed | 0.510 | 0.521 | 0.77 | |
| MIPROv2 | 0.531 | 0.656 | 0.83 | four demonstrations, instructions unchanged |
| GEPA, 400 calls | 0.531 | 0.562 | 0.71 | instructions rewritten with criteria and examples |
| GEPA with feedback, 400 calls | 0.510 | 0.521 | 0.77 | kept the seed; the budget allowed three fully validated candidates |
| GEPA with feedback, 1,500 calls | 0.531 | 0.573 | 0.73 | instructions rewritten over 47 proposals |

Selection is on validation only. MIPROv2 and GEPA tie there, so the tie-break declared in
advance was position consistency on validation: GEPA 0.85, MIPROv2 0.71. The GEPA judge goes
to stage 2; MIPROv2's higher test agreement is reported and not acted on. The seed judge said
tie on 6 of 96 test rows where the human label is tie on 40; MIPROv2's judge says it 23 times
and GEPA's 9. Of the 910 pairs, 105 had a tied top vote and became ties by rule.

## Stage 2: optimise responses with the calibrated judge

`stage2_responses.py` reloads the exact judge stage 1 scored (`best/program.json`, so the
demonstrations come with it), wraps it in a pairwise metric with position swap and 1 / 0 / 0.5
scores, and optimises a response prompt for Llama-3.1-8B on the response questions. The
incumbent is the seed prompt's own answer rendered the way the backend renders candidates, so
every baseline is 0.5. `--judge baseline` uses the same run's unoptimised judge for contrast.

```bash
python -m examples.preference_judge.stage2_responses \
    --judge-run runs/preference_judge/stage1-gepa --judge best \
    --judge-model qwen2.5-72b-instruct-awq --judge-base-url http://127.0.0.1:8123/v1 \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --optimizer GEPA --output runs/preference_judge/stage2-gepa-best-judge
```

GEPA, 400 calls, 16 held-out test questions. Each cell is a selected prompt's answers scored
against the same seed-prompt incumbents by one judge, not a head-to-head:

| Selected with | Scored by calibrated judge | Scored by seed judge |
|---|---|---|
| calibrated judge | 0.656 | 0.672 |
| seed judge | 0.594 | 0.547 |

Whichever judge does the scoring, the program chosen with the calibrated judge is preferred
more often. An earlier pair of runs with MIPROv2 as the response optimiser and its own
calibrated judge showed the same ordering at smaller gains (0.547 against 0.484); the two
optimisers used different judges, so that is not a ranking of optimisers. Both GEPA prompts
open with a list of task domains taken from the training questions; they still helped on the
disjoint test questions. Sixteen questions and one seed: an experiment, not a theorem.

A first attempt generated the incumbents with a plain chat call while DSPy renders candidates
through its signature format, under which the 8B answers about 40% shorter; the "baseline"
scored 0.25 instead of 0.5 and measured formatting rather than prompts. Render the incumbent
the way the backend will render candidates.
