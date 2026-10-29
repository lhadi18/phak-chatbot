from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError


SchemaModel = TypeVar("SchemaModel", bound=BaseModel)


class LLMProvider(ABC):
    """Small provider contract shared by local and optional hosted LLMs."""

    @abstractmethod
    def generate_text(self, prompt: str, *, system: str = "") -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_text(self, prompt: str, *, system: str = ""):
        """Yield generated text fragments for interactive CLI output."""
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        schema: type[SchemaModel],
        *,
        system: str = "",
        retries: int = 1,
    ) -> SchemaModel:
        """Generate JSON matching a Pydantic schema; repair malformed output once."""
        raw = ""
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            request = prompt
            if attempt:
                request = (
                    f"{prompt}\n\nYour previous output did not validate: {last_error}. "
                    "Repair it and return only valid JSON matching this schema:\n"
                    f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n"
                    f"Previous output:\n{raw}"
                )
            raw = self._generate_json_text(request, schema, system=system)
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Model returned invalid {schema.__name__} JSON after {retries + 1} attempts: {last_error}")

    @abstractmethod
    def _generate_json_text(self, prompt: str, schema: type[BaseModel], *, system: str = "") -> str:
        raise NotImplementedError
