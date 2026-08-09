"""Re-ranking: embed triples and query, keep top-k by cosine similarity."""
import numpy as np

import config
from embedder import embed, cosine_similarity


def rerank_triples(
    query: str, triples: list[tuple[str, str, str]], top_k: int = config.RERANK_TOP_K
) -> list[tuple[str, str, str]]:
    if not triples:
        return []
    strings = [f"{s} {p} {o}" for s, p, o in triples]
    vecs = embed(strings)
    qvec = embed([query])[0]
    sims = cosine_similarity(qvec, vecs)
    order = np.argsort(-sims)
    return [triples[i] for i in order[:top_k]]
