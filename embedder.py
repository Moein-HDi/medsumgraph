"""SentenceTransformer wrapper (all-MiniLM-L6-v2) with a lazy singleton."""
import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 matrix of normalized embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
    """Cosine sim between one query embedding and many candidate embeddings."""
    if candidate_vecs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return (candidate_vecs @ query_vec).astype(np.float32)
