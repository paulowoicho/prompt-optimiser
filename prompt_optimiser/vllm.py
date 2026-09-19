"""One vLLM endpoint configuration translated to each library's model interface."""

from dataclasses import dataclass
import os
from typing import Any

from prompt_optimiser.models import LiteLLMModel
from prompt_optimiser.models import optional_import


@dataclass(frozen=True)
class VLLM:
    """One endpoint configuration shared by the supported model transports.

    Attributes:
        model: Served model name.
        base_url: OpenAI-compatible endpoint ending in /v1.
        api_key_env: Environment variable containing the endpoint credential.
        max_tokens: Maximum response length.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        seed: Sampling seed.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 120.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model or not self.base_url.rstrip("/").endswith("/v1"):
            raise ValueError("Provide a served model name and a base_url ending in /v1")
        if self.max_tokens < 1 or self.timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")

    def _kwargs(self) -> dict[str, Any]:
        return {
            "api_base": self.base_url,
            "api_key": os.environ.get(self.api_key_env, "EMPTY"),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "seed": self.seed,
        }

    def as_dspy(self) -> Any:
        """Build a native DSPy LM; requires the dspy installation extra."""
        dspy = optional_import("dspy", "dspy")
        # The openai/ prefix selects the wire protocol; the destination is our vLLM URL.
        return dspy.LM(f"openai/{self.model}", **self._kwargs())

    def as_litellm(self) -> LiteLLMModel:
        """A plain callable ``model(system_prompt, text) -> str`` for backends of your own."""
        return LiteLLMModel(f"openai/{self.model}", **self._kwargs())

    def as_textgrad(self) -> Any:
        """Build a native TextGrad engine using this endpoint and fixed sampling settings."""
        optional_import("textgrad", "textgrad")
        from textgrad.engine import EngineLM

        transport = LiteLLMModel(f"openai/{self.model}", **self._kwargs())
        model_name = self.model

        class EndpointEngine(EngineLM):
            model_string = model_name

            def generate(
                self, prompt: str, system_prompt: str | None = None, **kwargs: Any
            ) -> str:
                # Sampling settings are fixed by VLLM configuration for this engine.
                return transport(
                    self.system_prompt if system_prompt is None else system_prompt, prompt
                )

            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return self.generate(*args, **kwargs)

        return EndpointEngine()

    def public_config(self) -> dict[str, Any]:
        """Return serialisable settings with the credential environment name, never its value."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "seed": self.seed,
            "api_key_env": self.api_key_env,
        }
