"""Phase 2: hybrid retrieval — global search + local search, then re-ranking."""
import json
import re

import prompts
from data.kg_builder import _parse_json_list
from graph import reranker
from graph.kg_store import KnowledgeGraph
from llm_client import LLMClient


def _entities_from_json(resp: str) -> list[str]:
    data = _parse_json_list(resp)
    return [str(e).strip() for e in data if str(e).strip()]


def global_search(llm: LLMClient, question: str) -> list[str]:
    """Global: LLM summarizes the question into its key entities."""
    resp = llm.chat(
        prompts.QUESTION_SUMMARY_SYSTEM,
        prompts.QUESTION_SUMMARY_USER.format(question=question),
    )
    return _entities_from_json(resp)


def local_search(llm: LLMClient, question: str) -> list[str]:
    """Local: extract medical entity names mentioned in the question."""
    resp = llm.chat(
        prompts.ENTITY_EXTRACTION_SYSTEM,
        prompts.ENTITY_EXTRACTION_USER.format(question=question),
    )
    return _entities_from_json(resp)


def hybrid_retrieve(
    llm: LLMClient, kg: KnowledgeGraph, question: str, top_k: int = None
) -> list[tuple[str, str, str]]:
    """Combine global + local entity sets, pull subgraph triples, rerank."""
    global_entities = global_search(llm, question)
    local_entities = local_search(llm, question)
    entities = list(dict.fromkeys(global_entities + local_entities))

    triples = kg.subgraph(entities)
    deduped = list(dict.fromkeys(triples))
    return reranker.rerank_triples(question, deduped, top_k=top_k or 20)
