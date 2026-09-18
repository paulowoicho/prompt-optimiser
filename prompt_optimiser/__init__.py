"""Optimise a prompt with data towards a metric, using DSPy, TextGrad or your own optimizer."""

from .experiment import ExperimentResult, FitResult, OptimizerBackend, Prompt, optimize
from .metrics import exact_match
from .native import DSPy, TextGrad
from .tracking import ConsoleTracker, JSONLTracker, MLflowTracker, WandbTracker
from .types import Event, Example, Metric, Tracker
from .vllm import VLLM

__all__ = [
    # entry point and result
    "optimize",
    "ExperimentResult",
    # data and objective
    "Example",
    "Metric",
    "exact_match",
    # models
    "VLLM",
    # optimizers
    "DSPy",
    "TextGrad",
    # extension contract for another optimizer
    "OptimizerBackend",
    "FitResult",
    "Prompt",
    # progress
    "Event",
    "Tracker",
    "ConsoleTracker",
    "JSONLTracker",
    "MLflowTracker",
    "WandbTracker",
]
