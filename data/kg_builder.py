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
from logging_utils import log


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
    """No-op stub: we skip the LLM summarize step (option 3 optimization).

    The original pipeline called an LLM to compress the raw UMLS+Wikipedia
    text into a JSON shape before extracting triples, but that extra hop
    isn't necessary — the triple-extraction prompt can operate on the raw
    context directly. Returning the context unchanged preserves the
    call signature so existing code paths work without modification.
    """
    return context


def extract_triples(llm: LLMClient, context_or_summary: str) -> list[list[str]]:
    """Extract triples from the raw medical context (UMLS definition + Wikipedia).

    Skips the summarize hop and goes straight to the relation prompt. Uses
    temperature=0 to keep the output structured and deterministic, since this
    is a constrained JSON-generation task with no creative component.
    """
    resp = llm.chat(
        prompts.RELATION_SYSTEM,
        prompts.RELATION_USER.format(context=context_or_summary),
        model=config.KG_LLM_MODEL,
        temperature=0.0,
    )
    data = _parse_json_list(resp)
    out = []
    for t in data:
        if not isinstance(t, list) or len(t) < 3:
            continue
        s, p, o = (_clean_token(x) for x in t[:3])
        if not (_is_valid_entity(s) and _is_valid_entity(o)):
            continue
        if p.lower() not in ALLOWED_PREDICATES:
            continue
        if s.lower() == o.lower():
            continue
        if p.lower() == "treated_with" and s.lower() == o.lower():
            continue
        out.append([s, p, o])
    return out


# Predicates that are medically meaningful and safe for a QA knowledge graph.
ALLOWED_PREDICATES = {
    "causes",
    "treated_with",
    "risk_factor",
    "symptoms",
    "diagnosed_by",
    "complication",
    "medication",
    "prevents",
    "contraindicated_with",
    "associated_with",
    "indicates",
    "defined_as",
}

# Generic tokens that are not medical entities (reject as subject/object).
GENERIC_ENTITIES = {
    "medications",
    "treatment",
    "treatments",
    "patients",
    "patient",
    "none",
    "disease",
    "diseases",
    "symptoms",
    "definition",
    "diagnosis",
    "therapy",
    "drugs",
    "drug",
    "medication",
    "complications",
    "risk factors",
    "n/a",
    "",
}


def _clean_token(tok: str) -> str:
    return tok.strip().strip('"').strip("'")


def _is_valid_entity(tok: str) -> bool:
    tok = _clean_token(tok).lower()
    if not tok or tok in GENERIC_ENTITIES:
        return False
    # must contain at least one alphabetic character
    return any(ch.isalpha() for ch in tok)


# Bump this when the extraction prompts/filters change so stale cached
# triples are re-extracted rather than reused.
KG_CACHE_VERSION = 2


def build_knowledge_graph_from_cache(
    entities: list[dict],
    kg: KnowledgeGraph,
    cache_dir: Path = config.KG_CACHE_DIR,
) -> KnowledgeGraph:
    """Assemble the graph ONLY from already-cached entities (no LLM calls).

    Used to rebuild the graph JSON from the per-entity cache after an
    interrupted build — no new scope extraction or entity processing.
    """
    for ent in tqdm(entities, desc="Building KG from cache"):
        cui = ent["cui"]
        name = ent["name"]
        cache_path = cache_dir / f"{cui}.v{KG_CACHE_VERSION}.json"
        if not cache_path.exists():
            continue
        with open(cache_path, encoding="utf-8") as f:
            triples = json.load(f)
        kg.add_node(name, cui=cui, entity_type=ent["type"])
        for subj, pred, obj in triples:
            kg.add_triple(subj.strip(), pred.strip(), obj.strip())

    kg.save(config.GRAPH_PATH)
    return kg


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
        cache_path = cache_dir / f"{cui}.v{KG_CACHE_VERSION}.json"

        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                triples = json.load(f)
        else:
            try:
                log("KG: %s (%s) — fetching Wikipedia", name, cui)
                context = ent["definition"]
                wiki = wikipedia_retriever.fetch_summary(name)
                if wiki:
                    context = (context + "\n" + wiki).strip() if context else wiki
                if not context.strip():
                    log("KG: %s (%s) — no context, skipping", name, cui)
                    continue
                log(
                    "KG: %s (%s) — extracting triples (%d chars, model=%s)",
                    name,
                    cui,
                    len(context),
                    config.KG_LLM_MODEL,
                )
                triples = extract_triples(llm, context)
            except Exception as e:
                # Leave no cache file on failure so the next run retries this
                # entity. This keeps long builds failure-tolerant.
                tqdm.write(f"[build-kg] skipped {name} ({cui}): {e}")
                continue

            #TODO OFF for now
            # Don't write empty caches — the next run would silently skip this
            # entity even if the LLM works. A failure leaves no cache, a
            # non-empty success does. Also defensive: if the LLM returned
            # pure noise (no parseable triples), treat it as a failure.
            # if triples:
            with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(triples, f, ensure_ascii=False)
            # else:
            #     log("KG: %s (%s) — got 0 triples, NOT caching (will retry)", name, cui)

        kg.add_node(name, cui=cui, entity_type=ent["type"])
        for subj, pred, obj in triples:
            kg.add_triple(subj.strip(), pred.strip(), obj.strip())

    kg.save(config.GRAPH_PATH)
    return kg


def load_or_build() -> KnowledgeGraph:
    if config.GRAPH_PATH.exists():
        return KnowledgeGraph.load(config.GRAPH_PATH)
    raise FileNotFoundError(
        "No knowledge graph found at "
        f"{config.GRAPH_PATH}.\n\n"
        "Build one first, e.g.:\n"
        "  python run_pipeline.py build-kg --scope medqa\n"
        "Building the full ~3.5M-concept UMLS subset without a scope filter "
        "would take weeks; --scope medqa restricts it to MedQA-relevant "
        "concepts and is the recommended default."
    )
