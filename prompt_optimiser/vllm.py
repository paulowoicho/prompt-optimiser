"""One vLLM endpoint configuration translated to each library's model interface."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import LiteLLMModel, optional_import


@dataclass(frozen=True)
class VLLM:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 120.0
    seed: int = 42

    def __post_init__(self):
        if not self.model or not self.base_url.rstrip("/").endswith("/v1"):
            raise ValueError("Provide a served model name and a base_url ending in /v1")
        if self.max_tokens < 1 or self.timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")

    def _kwargs(self):
        return {
            "api_base": self.base_url,
            "api_key": os.environ.get(self.api_key_env, "EMPTY"),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "seed": self.seed,
        }

    def as_dspy(self):
        dspy = optional_import("dspy", "dspy")
        # The openai/ prefix selects the wire protocol; the destination is our vLLM URL.
        return dspy.LM(f"openai/{self.model}", **self._kwargs())

    def as_litellm(self) -> LiteLLMModel:
        """A plain callable ``model(system_prompt, text) -> str`` for backends of your own."""
        return LiteLLMModel(f"openai/{self.model}", **self._kwargs())

    def as_textgrad(self):
        optional_import("textgrad", "textgrad")
        from textgrad.engine import EngineLM

        transport = LiteLLMModel(f"openai/{self.model}", **self._kwargs())
        model_name = self.model

        class EndpointEngine(EngineLM):
            model_string = model_name

            def generate(self, prompt, system_prompt=None, **kwargs):
                # Sampling settings are fixed by VLLM configuration for this engine.
                return transport(
                    self.system_prompt if system_prompt is None else system_prompt, prompt
                )

            def __call__(self, *args, **kwargs):
                return self.generate(*args, **kwargs)

        return EndpointEngine()

    def public_config(self):
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "seed": self.seed,
            "api_key_env": self.api_key_env,
        }
