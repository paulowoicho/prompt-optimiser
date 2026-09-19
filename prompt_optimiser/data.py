"""Normalise examples and keep data splits disjoint."""

from collections.abc import Iterable, Mapping
import math
import random

from prompt_optimiser.types import Example

Data = Iterable[Example | Mapping[str, str]]


def examples(data: Data, name: str) -> tuple[Example, ...]:
    """Convert input mappings to validated examples.

    Args:
        data: Examples or mappings with input and target text.
        name: Split name used in error messages.

    Returns:
        Nonempty tuple of examples.

    Raises:
        ValueError: The split is empty.
        TypeError: An input or target is not text.
        KeyError: A mapping lacks input or target.
    """
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
    """Split by input so duplicates cannot leak into validation.

    Args:
        rows: Training examples to split.
        fraction: Desired fraction of distinct inputs reserved for validation.
        seed: Local shuffle seed.

    Returns:
        Training and validation tuples, preserving their original row order.

    Raises:
        ValueError: Fewer than two distinct inputs are available.
    """
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
    """Reject inputs shared between splits.

    Args:
        **splits: Named example tuples.

    Raises:
        ValueError: Two splits contain the same input.
    """
    seen: set[str] = set()
    for name, rows in splits.items():
        inputs = {row.input for row in rows}
        if seen & inputs:
            raise ValueError(f"{name} contains inputs also present in another split")
        seen.update(inputs)
