from __future__ import annotations

from collections.abc import Iterator
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Optional provider for OpenAI-compatible services such as Groq or OpenRouter."""

    def __init__(self, *, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.1, timeout_seconds: int = 120) -> None:
        load_dotenv()
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is missing; add the provider's API key to .env.")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds)
        self.model = model
        self.temperature = temperature

    def _messages(self, prompt: str, system: str):
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def generate_text(self, prompt: str, *, system: str = "") -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=self.temperature,
        )
        return result.choices[0].message.content or ""

    def stream_text(self, prompt: str, *, system: str = "") -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=self.temperature,
            stream=True,
        )
        for part in stream:
            if part.choices and part.choices[0].delta.content:
                yield part.choices[0].delta.content

    def _generate_json_text(self, prompt: str, schema: type[BaseModel], *, system: str = "") -> str:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": schema.model_json_schema()},
        }
        result = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=self.temperature,
            response_format=response_format,
        )
        return result.choices[0].message.content or ""
