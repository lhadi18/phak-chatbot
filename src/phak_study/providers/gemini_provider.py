from __future__ import annotations

from collections.abc import Iterator
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Optional Google Gen AI provider; import only when selected in config."""

    def __init__(self, *, model: str, temperature: float = 0.1, timeout_seconds: int = 120) -> None:
        load_dotenv()
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env to use the Gemini provider.")
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
        self.model = model
        self.temperature = temperature

    def generate_text(self, prompt: str, *, system: str = "") -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"system_instruction": system, "temperature": self.temperature},
        )
        return response.text or ""

    def stream_text(self, prompt: str, *, system: str = "") -> Iterator[str]:
        for response in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config={"system_instruction": system, "temperature": self.temperature},
        ):
            if response.text:
                yield response.text

    def _generate_json_text(self, prompt: str, schema: type[BaseModel], *, system: str = "") -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return response.text or ""
