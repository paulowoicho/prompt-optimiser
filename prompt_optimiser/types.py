"""Small, dependency-free public contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

Metric = Callable[[str, str], float]
"""metric(expected, predicted) -> finite float; the objective used for selection and reporting."""


@dataclass(frozen=True)
class Example:
    input: str
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.input, str) or not isinstance(self.target, str):
            raise TypeError("Example input and target must be strings")


@dataclass(frozen=True)
class Event:
    """One progress record: start, candidate, artifact, finish or close."""

    kind: str
    step: int
    metrics: dict[str, float]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tracker(Protocol):
    def log(self, event: Event) -> None: ...

    def close(self, status: str) -> None: ...
