"""MT-bench human judgments as pairwise preference data, split by question so nothing leaks.

Source: ``lmsys/mt_bench_human_judgments`` (CC-BY-4.0), split ``human``: 3,355 votes by 65 expert
judges over 80 questions in 8 categories, comparing answers from six 2023-era models. Only the
first turn is used. Several judges vote on the same (question, model A, model B) pair; votes are
aggregated by plurality into one label per pair, TIE when the top vote counts are equal.

Two things matter for "proper ML practice" here and both live in this file:

1. Splits are by **question id**. Questions used to train, validate and test the judge are
   disjoint from the questions used later to optimise responses, so the judge is never scored on
   a question it learned from, and the response optimiser never touches judge-test questions.
2. Every pair appears in **both orderings** with the label swapped, so a judge that prefers
   position A is penalised in training and caught in evaluation.
"""

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from prompt_optimiser import Example

REPO = "lmsys/mt_bench_human_judgments"
FILE = "data/human-00000-of-00001-25f4910818759289.parquet"
LABELS = ("A_BETTER", "B_BETTER", "TIE")
SWAP = {"A_BETTER": "B_BETTER", "B_BETTER": "A_BETTER", "TIE": "TIE"}
CATEGORIES = (
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
)
# Questions per category in each split; 3+1+1+2+1+2 = 10 = every MT-bench category.
SPLIT_SIZES = {
    "judge_train": 3,
    "judge_validation": 1,
    "judge_test": 1,
    "response_train": 2,
    "response_validation": 1,
    "response_test": 2,
}


def category(question_id: int) -> str:
    return CATEGORIES[(question_id - 81) // 10]


@dataclass(frozen=True)
class Pair:
    question_id: int
    category: str
    question: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    label: str
    votes: int
    unanimous: bool
    tied_vote: bool = False


def load_pairs(cache_dir: Path = Path("runs/datasets/mt_bench")) -> list[Pair]:
    """Download once, then aggregate. See ``pairs_from_frame`` for the rules."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(REPO, FILE, repo_type="dataset", cache_dir=str(cache_dir))
    return pairs_from_frame(pd.read_parquet(path))


def pairs_from_frame(frame) -> list[Pair]:
    """Keep first turns; merge orderings; choose the most votes, TIE for equal top counts."""
    frame = frame[frame["turn"] == 1]
    votes: dict[tuple, Counter] = defaultdict(Counter)
    content: dict[tuple, tuple[str, str, str]] = {}
    for row in frame.itertuples():
        a, b = row.model_a, row.model_b
        conv_a, conv_b = list(row.conversation_a), list(row.conversation_b)
        answer_a, answer_b = conv_a[1]["content"], conv_b[1]["content"]
        label = {"model_a": "A_BETTER", "model_b": "B_BETTER", "tie": "TIE"}[row.winner]
        if a > b:  # canonical ordering by model name so both orderings of a vote merge
            a, b, answer_a, answer_b, label = b, a, answer_b, answer_a, SWAP[label]
        key = (int(row.question_id), a, b)
        votes[key][label] += 1
        content[key] = (conv_a[0]["content"], answer_a, answer_b)
    pairs = []
    for key in sorted(votes):
        counts = votes[key]
        (top, top_n), *rest = counts.most_common()
        tied_vote = bool(rest and top_n == rest[0][1])
        label = "TIE" if tied_vote else top
        question, answer_a, answer_b = content[key]
        pairs.append(
            Pair(
                question_id=key[0],
                category=category(key[0]),
                question=question,
                model_a=key[1],
                model_b=key[2],
                response_a=answer_a,
                response_b=answer_b,
                label=label,
                votes=sum(counts.values()),
                unanimous=len(counts) == 1,
                tied_vote=tied_vote,
            )
        )
    return pairs


def split_questions(pairs: list[Pair], seed: int = 42) -> dict[str, list[int]]:
    """Assign every question id to exactly one split, stratified by category."""
    by_category: dict[str, list[int]] = defaultdict(list)
    for question_id in sorted({pair.question_id for pair in pairs}):
        by_category[category(question_id)].append(question_id)
    rng = random.Random(seed)
    splits: dict[str, list[int]] = {name: [] for name in SPLIT_SIZES}
    for name in CATEGORIES:
        ids = by_category[name]
        if len(ids) != sum(SPLIT_SIZES.values()):
            expected = sum(SPLIT_SIZES.values())
            raise ValueError(f"Expected {expected} questions in {name}, got {len(ids)}")
        rng.shuffle(ids)
        for split, size in SPLIT_SIZES.items():
            splits[split].extend(ids[:size])
            ids = ids[size:]
    return {name: sorted(ids) for name, ids in splits.items()}


def judge_examples(
    pairs: list[Pair], question_ids: list[int], pairs_per_question: int = 6, seed: int = 42
) -> list[Example]:
    """Judge training rows: JSON in, label out, each pair in both orderings back to back."""
    rng = random.Random(seed)
    wanted = set(question_ids)
    per_question: dict[int, list[Pair]] = defaultdict(list)
    for pair in pairs:
        if pair.question_id in wanted:
            per_question[pair.question_id].append(pair)
    examples = []
    for question_id in sorted(per_question):
        chosen = per_question[question_id]
        if len(chosen) > pairs_per_question:
            chosen = rng.sample(chosen, pairs_per_question)
        for pair in chosen:
            forward = {
                "question": pair.question,
                "response_a": pair.response_a,
                "response_b": pair.response_b,
            }
            backward = {**forward, "response_a": pair.response_b, "response_b": pair.response_a}
            examples.append(Example(json.dumps(forward), pair.label))
            examples.append(Example(json.dumps(backward), SWAP[pair.label]))
    return examples


def questions(pairs: list[Pair], question_ids: list[int]) -> list[str]:
    """Unique question texts for the response-optimisation stage, in question-id order."""
    seen: dict[int, str] = {}
    for pair in pairs:
        if pair.question_id in question_ids:
            seen.setdefault(pair.question_id, pair.question)
    return [seen[question_id] for question_id in sorted(seen)]


def summary(pairs: list[Pair]) -> dict:
    return {
        "pairs": len(pairs),
        "questions": len({pair.question_id for pair in pairs}),
        "labels": dict(Counter(pair.label for pair in pairs)),
        "unanimous_pairs": sum(pair.unanimous for pair in pairs),
        "tied_vote_pairs": sum(pair.tied_vote for pair in pairs),
        "non_unanimous_ties": sum(
            1 for pair in pairs if pair.label == "TIE" and not pair.unanimous and pair.votes > 1
        ),
    }
