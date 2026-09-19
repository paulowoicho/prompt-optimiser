"""UCI SMS Spam Collection loader and class-balanced disjoint splits."""

import hashlib
import io
from pathlib import Path
import random
import urllib.request
import zipfile

from prompt_optimiser import Example

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_dataset(cache: Path) -> tuple[list[Example], str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        # Check the expected member before writing a downloaded response.
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            archive.getinfo("SMSSpamCollection")
        cache.write_bytes(content)
    content = cache.read_bytes()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        lines = archive.read("SMSSpamCollection").decode("utf-8").splitlines()
    unique: dict[str, Example] = {}
    conflicts: set[str] = set()
    for line in lines:
        label, text = line.split("\t", 1)
        text = " ".join(text.split())
        key = text.casefold()
        if label not in {"ham", "spam"}:
            raise ValueError(f"Unexpected dataset label: {label}")
        if key in unique and unique[key].target != label:
            conflicts.add(key)
        unique.setdefault(key, Example(text, label))
    return [row for key, row in unique.items() if key not in conflicts], hashlib.sha256(
        content
    ).hexdigest()


def make_splits(rows: list[Example], train_size: int, eval_size: int, seed: int):
    rng = random.Random(seed)
    splits: list[list[Example]] = [[], [], []]
    for label in ("ham", "spam"):
        group = [row for row in rows if row.target == label]
        rng.shuffle(group)
        if len(group) < train_size + 2 * eval_size:
            raise ValueError(f"Not enough distinct {label} messages for requested split sizes")
        splits[0].extend(group[:train_size])
        splits[1].extend(group[train_size : train_size + eval_size])
        splits[2].extend(group[train_size + eval_size : train_size + 2 * eval_size])
    for split in splits:
        rng.shuffle(split)
    return splits
