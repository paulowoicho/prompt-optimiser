from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

from .types import Example

Data = Iterable[Example | Mapping[str, str]]


def examples(data: Data, name: str) -> tuple[Example, ...]:
    rows = tuple(
        row if isinstance(row, Example) else Example(input=row["input"], target=row["target"])
        for row in data
    )
    if not rows:
        raise ValueError(f"{name} must not be empty")
    return rows


def split_validation(
    rows: tuple[Example, ...], fraction: float, seed: int
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    # Split by input, keeping duplicate inputs together to prevent leakage.
    inputs = list(dict.fromkeys(row.input for row in rows))
    if len(inputs) < 2:
        raise ValueError("Provide validation_data or at least two distinct training inputs")
    random.Random(seed).shuffle(inputs)
    size = min(len(inputs) - 1, max(1, math.ceil(len(inputs) * fraction)))
    held_out = set(inputs[:size])
    return (
        tuple(row for row in rows if row.input not in held_out),
        tuple(row for row in rows if row.input in held_out),
    )


def check_disjoint(**splits: tuple[Example, ...]) -> None:
    seen: set[str] = set()
    for name, rows in splits.items():
        inputs = {row.input for row in rows}
        if seen & inputs:
            raise ValueError(f"{name} contains inputs also present in another split")
        seen.update(inputs)
