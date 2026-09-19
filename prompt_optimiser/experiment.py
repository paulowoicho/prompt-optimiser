"""A thin experiment lifecycle around an optimizer's own training loop."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from prompt_optimiser.data import Data
from prompt_optimiser.data import check_disjoint
from prompt_optimiser.data import examples
from prompt_optimiser.data import split_validation
from prompt_optimiser.metrics import exact_match
from prompt_optimiser.types import Event
from prompt_optimiser.types import Example
from prompt_optimiser.types import Metric
from prompt_optimiser.types import Tracker


@dataclass
class Prompt:
    """Readable instructions plus the exact native predictor that was evaluated.

    Attributes:
        text: Human-readable instructions.
        predict_one: Native predictor accepting one text input.
        save_native: Artifact writer. None installs a writer for prompt.txt.
    """

    text: str
    predict_one: Callable[[str], str]
    save_native: Callable[[Path], None] | None = None

    def __post_init__(self) -> None:
        if self.save_native is None:
            text = self.text

            def save_text(directory: Path) -> None:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "prompt.txt").write_text(text, encoding="utf-8")

            self.save_native = save_text


@dataclass
class FitResult:
    """Predictors returned by an optimisation backend.

    Attributes:
        baseline: Predictor before optimisation.
        best: Validation-selected predictor; may be the baseline itself.
    """

    baseline: Prompt
    best: Prompt


@runtime_checkable
class OptimizerBackend(Protocol):
    """Anything with this one method is a backend. Accept ``**request`` to ignore arguments."""

    def fit(
        self,
        *,
        problem: str,
        model: Any,
        seed_prompt: str,
        train: tuple[Example, ...],
        validation: tuple[Example, ...],
        metric: Metric,
        greater_is_better: bool,
        random_state: int,
        report: Callable[[Event], None],
    ) -> FitResult:
        """Train and select a prompt without access to test data.

        Args:
            problem: Task description.
            model: Model configuration understood by the backend.
            seed_prompt: Initial instructions.
            train: Examples available for optimisation.
            validation: Examples available for prompt selection.
            metric: Numeric selection and reporting metric.
            greater_is_better: Whether higher metric values are preferred.
            random_state: Seed for reproducible sampling where supported.
            report: Callback receiving training events.

        Returns:
            The baseline and selected native predictors.
        """
        ...


def measure(
    predict: Callable[[str], str],
    data: Iterable[Example],
    metric: Metric,
    record: Callable[[Example, str, float], None] | None = None,
) -> dict[str, float | int]:
    """Evaluate a predictor using the mean per-example metric.

    Args:
        predict: Predictor accepting one text input.
        data: Examples to evaluate.
        metric: Numeric score for a reference and prediction.
        record: Optional callback receiving each example, prediction and score.

    Returns:
        Mean score and number of examples.

    Raises:
        TypeError: A prediction is not text.
        ValueError: Data is empty or a metric value is nonfinite.
    """
    scores = []
    for row in data:
        prediction = predict(row.input)
        if not isinstance(prediction, str):
            raise TypeError("Predictors must return text")
        score = float(metric(row.target, prediction))
        if not math.isfinite(score):
            raise ValueError("Evaluation metrics must be finite numbers")
        scores.append(score)
        if record is not None:
            record(row, prediction, score)
    if not scores:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"score": sum(scores) / len(scores), "n_examples": len(scores)}


@dataclass
class ExperimentResult:
    """The selected prompt, predictor and experiment artifacts.

    Attributes:
        prompt: Selected instructions.
        scores: Baseline and final measurements, grouped by data split.
        history: Native candidate events in reporting order.
        output_dir: Directory containing configuration, predictions and native artifacts.
    """

    prompt: str
    scores: dict[str, dict[str, dict[str, float | int]]]
    history: list[dict[str, Any]]
    output_dir: Path
    _predict: Callable[[str], str] = field(repr=False)

    def predict(self, inputs: Iterable[str]) -> list[str]:
        """Generate predictions with the selected native predictor.

        Args:
            inputs: Text inputs in prediction order; not a single string.

        Returns:
            Response text for each input.

        Raises:
            TypeError: Inputs or predictions violate the text-only contract.
        """
        if isinstance(inputs, str):
            raise TypeError("Pass a sequence of input strings")
        result = []
        for text in inputs:
            if not isinstance(text, str):
                raise TypeError("Inputs must be strings")
            prediction = self._predict(text)
            if not isinstance(prediction, str):
                raise TypeError("Predictors must return text")
            result.append(prediction)
        return result


def _looks_secret(name: str) -> bool:
    """Credential-like setting names. Tuning options such as max_tokens are not secrets."""
    lowered = name.lower()
    if any(word in lowered for word in ("api_key", "apikey", "secret", "password")):
        return True
    return lowered == "token" or lowered.endswith("_token")


def _public_value(value: Any) -> Any:
    # Nested experiment settings are kept; anything that looks like a credential is not.
    if hasattr(value, "public_config"):
        return value.public_config()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if _looks_secret(str(key)) else _public_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_public_value(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return getattr(value, "model", type(value).__name__)


def optimize(
    *,
    problem: str,
    model: Any,
    backend: OptimizerBackend,
    train_data: Data,
    validation_data: Data | None = None,
    test_data: Data | None = None,
    seed_prompt: str | None = None,
    metric: Metric = exact_match,
    greater_is_better: bool = True,
    random_state: int = 42,
    output_dir: str | Path | None = None,
    trackers: Iterable[Tracker] = (),
    metadata: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Train natively, evaluate on held-out data, and save portable experiment artifacts.

    Textual losses, critics, candidate generation and search stay inside the backend.
    The common metric measures task success; it does not replace native training loss.

    Args:
        problem: Nonempty task description, also the seed when seed_prompt is omitted.
        model: Model identifier or configuration understood by the backend.
        backend: Optimiser implementing fit().
        train_data: Training examples or input/target mappings.
        validation_data: Selection examples. None reserves 20% of distinct training inputs.
        test_data: Optional held-out examples, never passed to the backend.
        seed_prompt: Initial instructions, or None to use problem.
        metric: Finite numeric metric for selection and final evaluation.
        greater_is_better: Whether higher metric values are preferred.
        random_state: Seed for splitting and native training.
        output_dir: Empty artifact directory. None creates a fresh directory under runs/.
        trackers: Event sinks, closed on success and failure.
        metadata: JSON-serialisable experiment details saved with the run.

    Returns:
        Selected instructions, native predictor, split scores and artifact location.

    Raises:
        ValueError: Instructions, data splits, output directory or metric values are invalid.
        TypeError: Data or predictions violate the text-only contract.
    """
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty task description")
    seed_prompt = problem if seed_prompt is None else seed_prompt
    if not isinstance(seed_prompt, str) or not seed_prompt.strip():
        raise ValueError("seed_prompt must be a non-empty string")
    train = examples(train_data, "train_data")
    if validation_data is None:
        train, validation = split_validation(train, 0.2, random_state)
    else:
        validation = examples(validation_data, "validation_data")
    test = examples(test_data, "test_data") if test_data is not None else ()
    check_disjoint(train=train, validation=validation, test=test)
    if output_dir is None:
        from datetime import datetime
        from datetime import timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        output_dir = Path("runs") / f"{type(backend).__name__}-{stamp}"
    destination = Path(output_dir)
    # Prevent accidental mixing of two experiments' artifacts.
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"Use a new or empty output_dir: {destination}")
    trackers = tuple(trackers)
    history: list[dict[str, Any]] = []
    model_name = model if isinstance(model, str) else getattr(model, "model", type(model).__name__)
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    versions = {}
    for package in ("prompt-optimiser", "dspy", "textgrad", "litellm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    backend_config = {}
    if is_dataclass(backend):
        backend_config = _public_value(
            {item.name: getattr(backend, item.name) for item in fields(backend)}
        )
    config = {
        "model_config": _public_value(model),
        "backend_config": backend_config,
        "versions": versions,
        "problem": problem,
        "model": model_name,
        "backend": type(backend).__name__,
        "seed_prompt": seed_prompt,
        "random_state": random_state,
        "metric": getattr(metric, "__name__", type(metric).__name__),
        "greater_is_better": greater_is_better,
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "metadata": metadata or {},
    }

    def report(event: Event) -> None:
        with (destination / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), allow_nan=False) + "\n")
        if event.kind == "candidate":
            history.append(event.to_dict())
        for tracker in trackers:
            tracker.log(event)

    status = "failed"
    try:
        (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        report(Event("start", 0, {}, config))
        trained = backend.fit(
            problem=problem,
            model=model,
            seed_prompt=seed_prompt,
            train=train,
            validation=validation,
            metric=metric,
            greater_is_better=greater_is_better,
            random_state=random_state,
            report=report,
        )
        scores = {}
        (destination / "predictions").mkdir()
        for name, prompt in (("baseline", trained.baseline), ("final", trained.best)):
            # This is the final evaluation pass. Native validation history is kept separately.
            if name == "final" and prompt is trained.baseline:
                scores[name] = scores["baseline"]
                continue
            scores[name] = {}
            for split, rows in (("train", train), ("validation", validation), ("test", test)):
                if not rows:
                    continue
                # Per-row predictions make a score diagnosable after the run.
                with (destination / "predictions" / f"{name}-{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:

                    def record(
                        row: Example, prediction: str, score: float, handle: TextIO = handle
                    ) -> None:
                        handle.write(
                            json.dumps(
                                {
                                    "input": row.input,
                                    "target": row.target,
                                    "prediction": prediction,
                                    "score": score,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )

                    scores[name][split] = measure(prompt.predict_one, rows, metric, record)
        (destination / "prompt.txt").write_text(trained.best.text, encoding="utf-8")
        trained.baseline.save_native(destination / "baseline")
        trained.best.save_native(destination / "best")
        payload = {
            "schema_version": 1,
            "prompt": trained.best.text,
            "scores": scores,
            "history": history,
            "config": config,
        }
        (destination / "result.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        # All local files exist now; record success before trackers upload the directory.
        status = "finished"
        (destination / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        # Artifact event carries a directory so trackers can retain the native program too.
        report(Event("artifact", len(history), {}, {"path": str(destination.resolve())}))
        metrics = {}
        for phase, split_scores in scores.items():
            for split, values in split_scores.items():
                metrics[f"{phase}/{split}/score"] = values["score"]
        report(Event("finish", len(history), metrics, payload))
        return ExperimentResult(
            trained.best.text, scores, history, destination, trained.best.predict_one
        )
    except BaseException:
        import warnings

        status = "failed"
        (destination / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        # Ship the partial directory (progress, error events) so failures are diagnosable remotely.
        try:
            report(Event("artifact", len(history), {}, {"path": str(destination.resolve())}))
        except Exception as exc:
            warnings.warn(
                f"Failure diagnostics were not uploaded: {exc}", RuntimeWarning, stacklevel=2
            )
        raise
    finally:
        # Close every tracker, including on native training failures.
        import warnings

        (destination / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        for tracker in trackers:
            try:
                tracker.close(status)
            except Exception as exc:
                warnings.warn(f"Tracker cleanup failed: {exc}", RuntimeWarning, stacklevel=2)
