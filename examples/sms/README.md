# SMS spam

The UCI SMS Spam Collection (Almeida & Hidalgo 2011, CC BY 4.0), used to check that the
harness works on a labelled task with a plain metric. `data.py` downloads the archive,
normalises whitespace, drops duplicate and conflicting messages, and makes class-balanced
disjoint splits: ten training, twenty validation and twenty test messages per class at seed 42.
Balanced classes mean accuracy here does not estimate the natural spam rate.

```bash
python -m examples.sms.unified --optimizer GEPA \
    --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --train-per-class 10 --eval-per-class 20
python -m examples.sms.sweep --model llama-3.1-8b-instruct --base-url http://127.0.0.1:8124/v1 \
    --output runs/sweep-8b
```

`native_dspy.py` and `native_textgrad.py` are the single-tool scripts the adapters were derived
from, kept so the native workflow can be read without the harness.

Exact match, same splits, proposer or critic is the target model unless stated. The baseline is
kept on ties.

| Model | Configuration | Baseline val / test | Final val / test |
|---|---|---|---|
| Llama-3.1-8B | DSPy GEPA, 400 calls | 0.900 / 0.875 | 0.925 / 0.900 |
| Llama-3.1-8B | DSPy MIPROv2, 3 candidates, 3 trials, native defaults | 0.900 / 0.875 | 0.950 / 0.925 |
| Llama-3.1-8B | DSPy MIPROv2, zero-shot preset, data-aware proposer only | 0.900 / 0.875 | 0.900 / 0.875 |
| Llama-3.1-8B | DSPy COPRO, breadth 3, depth 2 | 0.900 / 0.875 | kept seed; candidate 0.825 |
| Llama-3.1-8B | DSPy LabeledFewShot, k=4 | 0.900 / 0.875 | tied, kept seed |
| Llama-3.1-8B | TextGrad TGD, 3 steps, batch 2 | 0.900 / 0.900 | kept seed |
| Llama-3.1-8B | TextGrad TGD, 3 steps, batch 4 | 0.900 / 0.900 | 0.925 / 0.875 |
| Llama-3.1-8B | TextGrad TGD, gradient memory 2, batch 2 | 0.900 / 0.900 | kept seed |
| Qwen2.5-72B-AWQ | DSPy MIPROv2, 3 candidates, 3 trials | 1.000 / 0.975 | kept seed |
| Qwen2.5-72B-AWQ | TextGrad TGD, 3 steps, batch 4 | 0.975 / 1.000 | kept seed |
| Qwen2.5-1.5B | DSPy MIPROv2 preset | 0.525 / 0.500 | kept seed |

Forty messages per split, so one message is 0.025 and none of these differences is
statistically strong. The table shows the harness running across tools, algorithms and
models, and that a configuration choice within one tool changes the outcome. TextGrad and
DSPy baselines differ on the same model because their message formatting differs.
