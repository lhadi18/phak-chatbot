from __future__ import annotations

from ..config import Settings
from .base import LLMProvider


def create_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            context_tokens=settings.llm_context_tokens,
        )
    if provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(model=settings.llm_model, temperature=settings.llm_temperature, timeout_seconds=settings.llm_timeout_seconds)
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key_env=settings.llm_api_key_env,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unsupported llm_provider {settings.llm_provider!r}; use ollama, gemini, or openai_compatible.")
