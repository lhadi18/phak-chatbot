from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str
    embedding_device: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_base_url: str
    llm_timeout_seconds: int
    llm_context_tokens: int
    llm_api_key_env: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    embedding_batch_size: int
    retrieval_top_k: int
    database_path: Path
    chapters: dict[int, str]
    chapter_section_ranges: dict[int, tuple[int, int]]


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or ROOT / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Settings(
        data_dir=ROOT / raw.get("data_dir", "data"),
        chroma_dir=ROOT / raw.get("chroma_dir", ".chroma"),
        collection_name=raw.get("collection_name", "phak"),
        embedding_model=raw.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_device=raw.get("embedding_device", "cpu"),
        llm_provider=raw.get("llm_provider", "ollama"),
        llm_model=raw.get("llm_model", "qwen3.5:2b"),
        llm_temperature=float(raw.get("llm_temperature", 0.1)),
        llm_base_url=raw.get("llm_base_url", "http://localhost:11434"),
        llm_timeout_seconds=int(raw.get("llm_timeout_seconds", 600)),
        llm_context_tokens=int(raw.get("llm_context_tokens", 4096)),
        llm_api_key_env=raw.get("llm_api_key_env", "OPENAI_API_KEY"),
        chunk_tokens=int(raw.get("chunk_tokens", 650)),
        chunk_overlap_tokens=int(raw.get("chunk_overlap_tokens", 100)),
        embedding_batch_size=int(raw.get("embedding_batch_size", 32)),
        retrieval_top_k=int(raw.get("retrieval_top_k", 6)),
        database_path=ROOT / raw.get("database_path", "study.sqlite3"),
        chapters={int(k): str(v) for k, v in raw.get("chapters", {}).items()},
        chapter_section_ranges={
            int(k): (int(v[0]), int(v[1]))
            for k, v in raw.get("chapter_section_ranges", {}).items()
        },
    )
