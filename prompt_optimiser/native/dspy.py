"""Run a native DSPy optimiser without replacing its training loop."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import inspect
import json
import math
from pathlib import Path
from typing import Any

from prompt_optimiser.experiment import FitResult
from prompt_optimiser.experiment import Prompt
from prompt_optimiser.experiment import measure
from prompt_optimiser.models import optional_import
from prompt_optimiser.types import Event
from prompt_optimiser.types import Example
from prompt_optimiser.types import Metric
from prompt_optimiser.vllm import VLLM


@dataclass
class DSPy:
    """A configurable adapter to DSPy's native compile loop.

    Attributes:
        optimizer: Exported optimiser name, class, factory or configured instance.
        optimizer_kwargs: Constructor options, including an optional native metric.
        compile_kwargs: Compile options; task data comes from the harness.
        optimizer_model: Proposal model; defaults to the task model.
        program: Native module or signature factory accepting text and returning answer.
            A supplied module preserves its own instructions instead of the seed prompt.
    """

    optimizer: Any = "MIPROv2"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    compile_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer_model: Any = None
    program: Any = None

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
        """Optimise native instructions using training and validation examples.

        Args:
            problem: Task description used by the backend.
            model: Model identifier, VLLM configuration or native model object.
            seed_prompt: Initial instructions.
            train: Examples available to the optimiser.
            validation: Examples used to select the final prompt.
            metric: Numeric selection metric, separate from a native loss.
            greater_is_better: Whether higher metric values are preferred.
            random_state: Seed for sampling and supported native optimisers.
            report: Progress callback supplied by the experiment harness.

        Returns:
            The baseline and the best validation-selected native predictor.

        Raises:
            ImportError: The optional optimisation library is missing.
            ValueError: Backend options are invalid or a metric is nonfinite.
        """
        dspy = optional_import("dspy", "dspy")
        from dspy.utils.callback import BaseCallback

        def lm(value: Any) -> Any:
            if isinstance(value, VLLM):
                return value.as_dspy()
            return dspy.LM(value) if isinstance(value, str) else value

        task_lm = lm(model)
        proposer = lm(self.optimizer_model) if self.optimizer_model is not None else task_lm
        optimizer_source = (
            getattr(dspy, self.optimizer) if isinstance(self.optimizer, str) else self.optimizer
        )
        signature = dspy.Signature("text -> answer", instructions=seed_prompt)
        program = (
            self.program.deepcopy()
            if isinstance(self.program, dspy.Module)
            else (self.program or dspy.Predict)(signature)
        )

        def instructions(native: Any) -> str:
            predictors = native.named_predictors()
            if len(predictors) == 1:
                return predictors[0][1].signature.instructions
            return "\n\n".join(
                f"[{name}]\n{pred.signature.instructions}" for name, pred in predictors
            )

        trainset = [
            dspy.Example(text=row.input, answer=row.target).with_inputs("text") for row in train
        ]
        valset = [
            dspy.Example(text=row.input, answer=row.target).with_inputs("text")
            for row in validation
        ]
        sign = 1 if greater_is_better else -1

        def objective(
            example: Any,
            prediction: Any,
            trace: Any = None,
            pred_name: str | None = None,
            pred_trace: Any = None,
            program_trace: Any = None,
        ) -> float:
            score = float(metric(example.answer, prediction.answer))
            if not math.isfinite(score):
                raise ValueError("Evaluation metrics must be finite numbers")
            return sign * score

        kwargs = dict(self.optimizer_kwargs)
        supplied_instance = hasattr(optimizer_source, "compile") and not inspect.isclass(
            optimizer_source
        )
        constructor_parameters = (
            {} if supplied_instance else inspect.signature(optimizer_source).parameters
        )
        # Only explicit native parameters get defaults. Do not invent kwargs for
        # optimisers whose constructor accepts **kwargs but ignores unknown fields.
        for key, value in {
            "metric": objective,
            "task_model": task_lm,
            "prompt_model": proposer,
            "reflection_lm": proposer,
            "seed": random_state,
            "num_threads": 1,
        }.items():
            if key in constructor_parameters:
                kwargs.setdefault(key, value)
        if supplied_instance and (kwargs or self.optimizer_model is not None):
            raise ValueError(
                "Configure an optimizer instance directly, without constructor options"
            )
        optimizer = optimizer_source if supplied_instance else optimizer_source(**kwargs)
        compile_options = dict(self.compile_kwargs)
        if {"student", "trainset", "valset"} & compile_options.keys():
            raise ValueError("Pass task data to optimize(), not through compile_kwargs")
        compile_parameters = inspect.signature(optimizer.compile).parameters
        if "valset" in compile_parameters:
            compile_options["valset"] = valset
        if "seed" in compile_parameters:
            compile_options.setdefault("seed", random_state)

        def wrap(native: Any) -> Prompt:
            def predict(text: str) -> str:
                with dspy.context(lm=task_lm):
                    return native(text=text).answer

            def save(directory: Path) -> None:
                directory.mkdir(parents=True, exist_ok=True)
                native.save(directory / "program.json")
                adapter = dspy.settings.adapter or dspy.ChatAdapter()
                templates = {
                    name: adapter.format(
                        pred.signature,
                        pred.demos,
                        {key: "{" + key + "}" for key in pred.signature.input_fields},
                    )
                    for name, pred in native.named_predictors()
                }
                # Multiple stages retain separate templates; program.json retains
                # native predictor state. Recreate the same module class to load it.
                messages = next(iter(templates.values())) if len(templates) == 1 else templates
                (directory / "messages.json").write_text(
                    json.dumps(messages, indent=2), encoding="utf-8"
                )

            return Prompt(instructions(native), predict, save)

        seed_source = "native_program" if isinstance(self.program, dspy.Module) else "seed_prompt"
        baseline = wrap(program.deepcopy())
        baseline_score = measure(baseline.predict_one, validation, metric)["score"]
        report(
            Event(
                "candidate",
                0,
                {"validation/score": baseline_score},
                {
                    "prompt": baseline.text,
                    "phase": "baseline",
                    "seed_source": seed_source,
                },
            )
        )
        validation_inputs = {row.input for row in validation}
        training_inputs = {row.input for row in train}
        native_metric = supplied_instance or "metric" in self.optimizer_kwargs

        class Progress(BaseCallback):
            def __init__(self) -> None:
                self.pending = {}
                self.step = 1
                self.error = None

            def on_evaluate_start(
                self, call_id: str, instance: Any, inputs: dict[str, Any]
            ) -> None:
                rows = inputs.get("devset") or getattr(instance, "devset", ())
                observed = {getattr(row, "text", None) for row in rows}
                if observed and observed <= validation_inputs:
                    split = "validation"
                elif observed and observed <= training_inputs:
                    split = "train"
                else:
                    split = "native"
                self.pending[call_id] = (inputs["program"].deepcopy(), split)

            def on_evaluate_end(
                self, call_id: str, outputs: Any, exception: Exception | None = None
            ) -> None:
                pending = self.pending.pop(call_id, None)
                if exception is not None or pending is None:
                    return
                candidate, split = pending
                try:
                    # With a custom native metric, its scale/direction may differ
                    # from the common reporting metric. Label that explicitly.
                    score_key = "native/score" if native_metric else f"{split}/score"
                    score = outputs.score / 100
                    if not native_metric:
                        score *= sign
                    report(
                        Event(
                            "candidate",
                            self.step,
                            {score_key: score},
                            {
                                "prompt": instructions(candidate),
                                "native_state": candidate.dump_state(),
                                "evaluated_split": split,
                            },
                        )
                    )
                    self.step += 1
                except Exception as exc:
                    # DSPy swallows callback failures, so surface them after compile.
                    self.error = exc

        progress = Progress()
        with dspy.context(lm=task_lm, callbacks=[*(dspy.settings.callbacks or []), progress]):
            compiled = optimizer.compile(program, trainset=trainset, **compile_options)
        if progress.error is not None:
            raise progress.error
        candidate = wrap(compiled)
        score = measure(candidate.predict_one, validation, metric)["score"]
        improved = sign * score > sign * baseline_score
        report(
            Event(
                "candidate",
                progress.step,
                {"validation/score": score},
                {
                    "prompt": candidate.text,
                    "phase": "compiled",
                    "selected": improved,
                    "native_state": compiled.dump_state(),
                },
            )
        )
        return FitResult(baseline, candidate if improved else baseline)
