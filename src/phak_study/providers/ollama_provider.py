from __future__ import annotations

from collections.abc import Iterator

from ollama import Client, ResponseError

from pydantic import BaseModel

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_seconds: int,
        temperature: float = 0.1,
        context_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.options = {"temperature": temperature, "num_ctx": context_tokens}
        self.client = Client(host=self.base_url, timeout=timeout_seconds)
        self._available = False

    def _check_ready(self) -> None:
        if self._available:
            return
        try:
            response = self.client.list()
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not responding. Start the Ollama app or run `ollama serve`, "
                "then retry. Check llm_base_url in config.yaml."
            ) from exc
        models = getattr(response, "models", [])
        names = {getattr(item, "model", "") for item in models}
        if self.model not in names:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not pulled. Fix it with: `ollama pull {self.model}`"
            )
        self._available = True

    def _chat(self, prompt: str, *, system: str, stream: bool, format_schema=None):
        self._check_ready()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                format=format_schema,
                options=self.options,
            )
        except ResponseError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise RuntimeError(
                    f"Ollama model '{self.model}' is unavailable. Fix it with: `ollama pull {self.model}`"
                ) from exc
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

    def generate_text(self, prompt: str, *, system: str = "") -> str:
        response = self._chat(prompt, system=system, stream=False)
        return response.message.content

    def stream_text(self, prompt: str, *, system: str = "") -> Iterator[str]:
        try:
            for part in self._chat(prompt, system=system, stream=True):
                fragment = part.message.content
                if fragment:
                    yield fragment
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Ollama streaming failed at {self.base_url}: {exc}") from exc

    def _generate_json_text(self, prompt: str, schema: type[BaseModel], *, system: str = "") -> str:
        response = self._chat(
            prompt,
            system=system,
            stream=False,
            format_schema=schema.model_json_schema(),
        )
        return response.message.content
