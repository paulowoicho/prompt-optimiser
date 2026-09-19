"""Synthetic coaching preference data: label provenance, splits, orderings, committee votes."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pdm = importlib.import_module("examples.career_coaching.preference_data")
stage2 = importlib.import_module("examples.career_coaching.optimize_with_judge")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pdm, "CACHE", tmp_path / "cache")
    return tmp_path / "cache"


def fake_questions(n_per_scenario=2):
    items = [{"scenario": "original", "question": q} for q in pdm.ORIGINAL_QUESTIONS[:4]]
    for s in pdm.SCENARIOS:
        items += [
            {"scenario": s, "question": f"{s} question {k} with enough words to count as real"}
            for k in range(n_per_scenario)
        ]
    return [{"question_id": i, **it} for i, it in enumerate(items)]


def test_generate_questions_parses_json_arrays_dedupes_and_caches(cache):
    calls = []

    def call(system, text):
        calls.append(text)
        topic = text.split("about: ")[1][:25]
        messages = [
            f"message {i} about {topic} with plenty of specific detail here" for i in range(3)
        ]
        return 'Sure:\n["' + '", "'.join(messages) + '", "short one"]'

    qs = pdm.generate_questions(call, {"model": "fake"}, per_scenario=3, seed=1)
    assert len(calls) == len(pdm.SCENARIOS)
    assert len(qs) == len(pdm.ORIGINAL_QUESTIONS) + 3 * len(pdm.SCENARIOS)  # "short one" dropped
    assert {q["scenario"] for q in qs} == {"original", *pdm.SCENARIOS}
    again = pdm.generate_questions(call, {"model": "fake"}, per_scenario=3, seed=1)
    assert again == qs and len(calls) == len(pdm.SCENARIOS)  # served from cache
    assert (cache / "questions.json").exists()


def test_generate_answers_builds_every_degradation_kind(cache):
    qs = fake_questions(1)[:3]

    def strong(system, text):
        if system.startswith("Rewrite the coaching reply so it keeps"):
            return "generic platitudes"
        if system.startswith("Rewrite the coaching reply so that one"):
            return "strong but with a dishonest tip"
        return f"STRONG answer for {text[:20]} " + "detail " * 60

    small = lambda system, text: f"small answer under {system[:12]} for {text[:10]}"  # noqa: E731
    answers = pdm.generate_answers(strong, small, {"m": 1}, qs)
    a = answers[qs[0]["question"]]
    assert set(a) == {"strong", "seed8b", "alt8b", "generic", "truncated", "padded", "unethical"}
    assert a["truncated"] != a["strong"] and a["strong"].startswith(a["truncated"][:30])
    assert a["padded"].startswith(a["strong"]) and len(a["padded"]) > len(a["strong"])
    assert a["seed8b"] != a["alt8b"]


def test_committee_label_strict_majority_swap_seeds_and_invalid():
    def biased(system, text):
        payload = json.loads(text)
        return (
            "reasoning...\nA_BETTER"
            if payload["response_a"] == "good"
            else "reasoning...\nB_BETTER"
        )

    seen = []
    calls = pdm.committee_calls(lambda seed: seen.append(seed) or biased, seeds=(7, 8))
    assert seen == [7, 8]  # one transport per seed
    label, tally = pdm.committee_label(calls, "q", "good", "bad")
    assert label == "A_BETTER" and "A_BETTER=4" in tally  # consistent across orderings
    label, _ = pdm.committee_label([lambda s, t: "thinking\nA_BETTER"], "q", "x", "y")
    assert label == "TIE"  # always says A: one vote each after swapping -> no majority
    label, tally = pdm.committee_label([lambda s, t: "no label here"], "q", "x", "y")
    assert label == "TIE" and "INVALID=2" in tally
    raw = iter(
        ["A_BETTER", "B_BETTER", "A_BETTER", "TIE"]
    )  # after swap: A, A, A?, no: A, A, A, TIE
    label, tally = pdm.committee_label([lambda s, t: next(raw)] * 2, "q", "x", "y")
    assert tally == "A_BETTER=3 TIE=1" and label == "A_BETTER"  # 3 of 4 valid is a majority
    raw = iter(["A_BETTER", "A_BETTER", "TIE", "TIE"])  # after swap: A, B, TIE, TIE -> no majority
    label, tally = pdm.committee_label([lambda s, t: next(raw)] * 2, "q", "x", "y")
    assert label == "TIE" and tally == "A_BETTER=1 B_BETTER=1 TIE=2"


def test_build_pairs_marks_sources_kinds_and_keeps_donors_within_split(cache):
    qs = fake_questions(12)
    kinds = ("strong", "seed8b", "alt8b", "generic", "truncated", "padded", "unethical")
    answers = {q["question"]: {k: f"{k}-{q['question_id']}" for k in kinds} for q in qs}
    splits = pdm.split_questions(qs, seed=5)
    pairs = pdm.build_pairs(
        [lambda s, t: "reasoning\nTIE"], {"c": 1}, qs, answers, splits, seeds=(0,)
    )
    assert len(pairs) == len(qs) * (6 + 2)
    # a degradation that failed to change the text is skipped rather than labelled A_BETTER
    answers[qs[0]["question"]]["generic"] = answers[qs[0]["question"]]["strong"]
    fewer = pdm.build_pairs([lambda s, t: "TIE"], {"c": 2}, qs, answers, splits, seeds=(0,))
    assert len(fewer) == len(pairs) - 1
    split_of = {qid: name for name, ids in splits.items() for qid in ids}
    owner = {answers[q["question"]]["strong"]: q["question_id"] for q in qs}
    for p in pairs:
        if p.kind == "strong_vs_offtopic":
            assert owner[p.response_b] != p.question_id
            assert split_of[owner[p.response_b]] == split_of[p.question_id]  # never crosses
    constructed = [p for p in pairs if p.source == "constructed"]
    assert all(p.label == "A_BETTER" for p in constructed if p.kind != "strong_vs_strong")
    assert all(p.label == "TIE" for p in constructed if p.kind == "strong_vs_strong")
    committee = [p for p in pairs if p.source == "committee"]
    assert {p.kind for p in committee} == {"seed8b_vs_strong", "seed8b_vs_alt8b"}
    assert all(p.label == "TIE" and p.votes for p in committee)
    assert pdm.summary(pairs)["by_source"] == {"constructed": 6 * len(qs), "committee": 2 * len(qs)}
    stored = json.loads((cache / "pairs.json").read_text())["key"]
    assert stored["answers"] == pdm.fingerprint(answers) and stored["seeds"] == [0]


def test_splits_disjoint_and_examples_match_meta_order():
    qs = fake_questions(12)
    splits = pdm.split_questions(qs, seed=2)
    ids = [i for s in splits.values() for i in s]
    assert len(ids) == len(set(ids)) == len(qs)
    assert all(splits[name] for name in pdm.SPLIT_FRACTIONS)
    pairs = [
        pdm.Pair(
            q["question_id"],
            q["scenario"],
            q["question"],
            "a",
            "b",
            "A_BETTER",
            "constructed",
            "strong_vs_generic",
        )
        for q in qs
    ] + [
        pdm.Pair(
            q["question_id"],
            q["scenario"],
            q["question"],
            "a",
            "c",
            "TIE",
            "committee",
            "seed8b_vs_strong",
        )
        for q in qs
    ]
    rows = pdm.judge_examples(pairs, splits["judge_test"], per_question=1, seed=3)
    meta = pdm.judge_examples_meta(pairs, splits["judge_test"], per_question=1, seed=3)
    assert len(rows) == len(meta) == 2 * len(splits["judge_test"])
    assert rows[1].target == pdm.SWAP[rows[0].target]
    assert meta[0] == meta[1]


def test_preference_metric_and_judge_reload(tmp_path):
    quality = {"better": 2, "seed": 1, "worse": 0}

    def judge(text):
        payload = json.loads(text)
        a, b = quality[payload["response_a"]], quality[payload["response_b"]]
        return "TIE" if a == b else ("A_BETTER" if a > b else "B_BETTER")

    metric = stage2.PreferenceMetric(judge, question_for={"seed": "q"})
    assert metric("seed", "better") == 1.0 and metric("seed", "worse") == 0.0
    run = tmp_path / "stage1"
    (run / "best").mkdir(parents=True)
    (run / "best" / "prompt.txt").write_text("JUDGE")

    class FakeVLLM:
        def as_litellm(self):
            return lambda system, text: "TIE" if system == "JUDGE" else "no"

    assert stage2.load_judge(run, "best", FakeVLLM())("{}") == "TIE"
