from __future__ import annotations

from functools import lru_cache

from .config import Settings


@lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is missing. Install the project requirements first.") from exc
    return SentenceTransformer(model_name, device=device)


def encode_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    model = _load(settings.embedding_model, settings.embedding_device)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
