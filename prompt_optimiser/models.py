from __future__ import annotations

import importlib
from typing import Any


def optional_import(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise ImportError(
            f"Install this integration with: pip install 'prompt-optimiser[{extra}]'"
        ) from exc


class LiteLLMModel:
    """Provider-independent model. Credentials are read by LiteLLM, never logged here."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        if "messages" in kwargs or kwargs.get("stream"):
            raise ValueError("LiteLLMModel requires non-streaming text completions")
        self.name = model
        self.kwargs = kwargs

    def __call__(self, prompt: str, text: str) -> str:
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
