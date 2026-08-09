"""Phase 1: medical knowledge graph construction.

Per entity: C_e = UMLS definition + Wikipedia summary
            C_sum = L_sum(C_e)   (summarization)
            triples = L_rel(C_sum) (relationship extraction)

Results are cached per-CUI on disk so builds can resume.
"""
import json
import re
from pathlib import Path

from tqdm import tqdm

import config
import prompts
from data import umls_loader, wikipedia_retriever
from graph.kg_store import KnowledgeGraph
from llm_client import LLMClient


def _parse_json_list(text: str) -> list:
    """Best-effort extraction of a JSON list from an LLM response."""
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return data


def summarize_entity(llm: LLMClient, context: str) -> str:
    return llm.chat(prompts.SUMMARIZE_SYSTEM, prompts.SUMMARIZE_USER.format(context=context))


def extract_triples(llm: LLMClient, summary: str) -> list[list[str]]:
    resp = llm.chat(prompts.RELATION_SYSTEM, prompts.RELATION_USER.format(summary=summary))
    data = _parse_json_list(resp)
    return [t for t in data if isinstance(t, list) and len(t) >= 3]


def build_knowledge_graph(
    entities: list[dict],
    llm: LLMClient,
    kg: KnowledgeGraph,
    limit: int | None = config.KG_ENTITY_LIMIT,
    cache_dir: Path = config.KG_CACHE_DIR,
) -> KnowledgeGraph:
    """Process entities (skipping cached CUIs) and populate the graph."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        entities = entities[:limit]

    for ent in tqdm(entities, desc="Building KG"):
        cui = ent["cui"]
        name = ent["name"]
        cache_path = cache_dir / f"{cui}.json"

        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                triples = json.load(f)
        else:
            context = ent["definition"]
            wiki = wikipedia_retriever.fetch_summary(name)
            if wiki:
                context = (context + "\n" + wiki).strip() if context else wiki
            if not context.strip():
                continue
            summary = summarize_entity(llm, context)
            triples = extract_triples(llm, summary)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(triples, f, ensure_ascii=False)

        kg.add_node(name, cui=cui, entity_type=ent["type"])
        for subj, pred, obj in triples:
            kg.add_triple(subj.strip(), pred.strip(), obj.strip())

    kg.save(config.GRAPH_PATH)
    return kg


def load_or_build() -> KnowledgeGraph:
    if config.GRAPH_PATH.exists():
        return KnowledgeGraph.load(config.GRAPH_PATH)
    entities = umls_loader.load_entities()
    kg = build_knowledge_graph(entities, LLMClient(), KnowledgeGraph())
    return kg
