"""Small, dependency-free public contracts."""

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

Metric = Callable[[str, str], float]
"""metric(expected, predicted) -> finite float; the objective used for selection and reporting."""


@dataclass(frozen=True)
class Example:
    """One supervised input and reference.

    Attributes:
        input: Text passed to the predictor.
        target: Reference passed to the evaluation metric.
    """

    input: str
    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.input, str) or not isinstance(self.target, str):
            raise TypeError("Example input and target must be strings")


@dataclass(frozen=True)
class Event:
    """One progress record emitted by an experiment or backend.

    Attributes:
        kind: Event name, such as start, candidate, artifact, finish or close.
        step: Backend step or candidate index.
        metrics: Named numeric measurements.
        data: JSON-serialisable settings, prompt state or artifact paths.
    """

    kind: str
    step: int
    metrics: dict[str, float]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Tracker(Protocol):
    """Receive experiment events and release resources when the run ends."""

    def log(self, event: Event) -> None:
        """Record an event.

        Args:
            event: Progress, metrics or artifact location to retain.
        """
        ...

    def close(self, status: str) -> None:
        """Finish the run and release tracker resources.

        Args:
            status: Either "finished" or "failed".
        """
        ...
