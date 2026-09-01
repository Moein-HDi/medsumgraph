"""Re-ranking: embed triples and query, keep top-k by cosine similarity,
with per-subject diversity so a single entity cannot crowd out the context."""
import numpy as np

import config
from embedder import cosine_similarity, embed


def rerank_triples(
    query: str,
    triples: list[tuple[str, str, str]],
    top_k: int = config.RERANK_TOP_K,
    max_per_subject: int = 4,
) -> list[tuple[str, str, str]]:
    """Return the top-k triples, at most max_per_subject sharing a subject.

    Without the per-subject cap, the most semantically similar entity (e.g.
    "sensorineural hearing loss" for a tinnitus question) floods the top-k
    with near-duplicates and pushes out the few triples that matter.
    """
    if not triples:
        return []
    strings = [f"{s} {p} {o}" for s, p, o in triples]
    vecs = embed(strings)
    qvec = embed([query])[0]
    sims = cosine_similarity(qvec, vecs)
    order = np.argsort(-sims)

    chosen: list[tuple[str, str, str]] = []
    seen_subjects: dict[str, int] = {}
    for i in order:
        s, p, o = triples[i]
        if seen_subjects.get(s.lower(), 0) >= max_per_subject:
            continue
        chosen.append((s, p, o))
        seen_subjects[s.lower()] = seen_subjects.get(s.lower(), 0) + 1
        if len(chosen) >= top_k:
            break
    return chosen
