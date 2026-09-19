"""Model transports and lazy imports for optional integrations."""

import importlib
from types import ModuleType
from typing import Any


def optional_import(module: str, extra: str) -> ModuleType:
    """Import an integration without loading it during core package import.

    Args:
        module: Module to import.
        extra: Installation extra to suggest when the module is missing.

    Returns:
        Imported module.

    Raises:
        ImportError: The requested integration or one of its dependencies is missing.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise ImportError(
            f"Install this integration with: pip install 'prompt-optimiser[{extra}]'"
        ) from exc


class LiteLLMModel:
    """A non-streaming chat completion transport backed by LiteLLM.

    Attributes:
        name: Provider-prefixed model identifier.
        kwargs: Native completion options, which may include credentials.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Configure a text-only model transport.

        Args:
            model: LiteLLM model identifier.
            **kwargs: Completion options passed through to LiteLLM.

        Raises:
            ValueError: Options override messages or enable streaming.
        """
        if "messages" in kwargs or kwargs.get("stream"):
            raise ValueError("LiteLLMModel requires non-streaming text completions")
        self.name = model
        self.kwargs = kwargs

    def __call__(self, prompt: str, text: str) -> str:
        """Generate a text response.

        Args:
            prompt: System instructions.
            text: User input.

        Returns:
            The model's response text.

        Raises:
            ValueError: The response contains no text.
            ImportError: LiteLLM is not installed.
        """
        litellm = optional_import("litellm", "llm")
        response = litellm.completion(
            model=self.name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            **self.kwargs,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Model returned no text; tool-only responses are not supported")
        return content
