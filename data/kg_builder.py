"""Phase 1: medical knowledge graph construction.

Per entity: C_e = UMLS definition + Wikipedia summary
            C_sum = L_sum(C_e)   (summarization)
            triples = L_rel(C_sum) (relationship extraction)

Results are cached per-CUI on disk so builds can resume.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
import prompts
from data import umls_loader, wikipedia_retriever
from graph.kg_store import KnowledgeGraph
from llm_client import LLMClient
from logging_utils import log


def _parse_json_list(text: str) -> list:
    """Best-effort extraction of a JSON array from an LLM response.

    Handles truncation (capped output) by finding the last complete ']'
    before the truncation point, trimming back to a valid array boundary,
    and parsing from there.
    """
    text = text.strip()
    if not text:
        return []

    # Find the opening bracket of the top-level array.
    start = text.find("[")
    if start == -1:
        return []

    # Find the LAST closing bracket that completes a valid JSON array.
    # Start from the end and work backwards.
    best: list | None = None
    for end in range(len(text) - 1, start, -1):
        if text[end] != "]":
            continue
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                # Validate: each element should be a 3-element list (triple)
                valid = [t for t in data if isinstance(t, list) and len(t) >= 3]
                if len(valid) >= len(best or []):
                    best = valid
                    break  # first valid from the end is good enough
        except json.JSONDecodeError:
            continue

    return best or []


def summarize_entity(llm: LLMClient, context: str) -> str:
    return llm.chat(
        prompts.SUMMARIZE_SYSTEM, prompts.SUMMARIZE_USER.format(context=context)
    )


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
        # temperature=0.0,
    )
    data = _parse_json_list(resp)
    out = []
    for t in data:
        if not isinstance(t, list) or len(t) < 3:
            continue
        s, p, o = (_clean_token(x) for x in t[:3])
        if not (_is_valid_entity(s) and _is_valid_entity(o)):
            continue
        # TODO ALLOW EVERYTHING TEMPORARILY
        # if p.lower() not in ALLOWED_PREDICATES:
        #     continue
        if s.lower() == o.lower():
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
KG_CACHE_VERSION = 3


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


def _process_one_entity(
    ent: dict, llm: LLMClient, cache_dir: Path
) -> tuple[str, str, str, list] | None:
    """Process a single entity: fetch Wikipedia if needed, extract triples.

    Returns (cui, name, type, triples) on success, None on skip/failure.
    This function is called from the parallel thread pool.
    """
    cui = ent["cui"]
    name = ent["name"]
    cache_path = cache_dir / f"{cui}.v{KG_CACHE_VERSION}.json"

    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            triples = json.load(f)
        return cui, name, ent["type"], triples

    context = ent["definition"]

    # TODO OFF for now
    # Only fetch Wikipedia if UMLS definition is short or missing
    # if len(context.split()) < 30:
    try:
        log("KG: %s (%s) — fetching Wikipedia", name, cui)
        wiki = wikipedia_retriever.fetch_summary(name)
        if wiki:
            context = (
                (name + "\n" + context + "\n" + wiki).strip()
                if context
                else (name + "\n" + wiki).strip()
            )
    except Exception:
        pass  # Wikipedia failure is non-fatal; use UMLS definition only

    if not context.strip():
        log("KG: %s (%s) — no context, skipping", name, cui)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False)
        return None

    # log("KG: %s (%s) — summarizing (%d chars)", name, cui, len(context))
    # summary = summarize_entity(llm, context)
    summary = context
    log(
        "KG: %s (%s) — extracting triples (%d chars, model=%s)",
        name,
        cui,
        len(summary),
        config.KG_LLM_MODEL,
    )
    try:
        triples = extract_triples(llm, summary)
    except Exception as e:
        tqdm.write(f"[build-kg] skipped {name} ({cui}): {e}")
        return None

    # Write cache (including empty — some entities genuinely have no triples)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False)

    return cui, name, ent["type"], triples


# Configurable: how many parallel LLM requests to fire at once.
# OpenRouter's free tier allows ~20 RPM; each call takes 2-5s,
# so 5 workers keeps well within limits while giving ~5x speedup.
KG_WORKERS = int(os.getenv("KG_WORKERS", "5"))


def build_knowledge_graph(
    entities: list[dict],
    llm: LLMClient,
    kg: KnowledgeGraph,
    limit: int | None = config.KG_ENTITY_LIMIT,
    cache_dir: Path = config.KG_CACHE_DIR,
) -> KnowledgeGraph:
    """Process entities in parallel (skipping cached CUIs) and populate the graph."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        entities = entities[:limit]

    pbar = tqdm(total=len(entities), desc="Building KG")
    completed = 0
    with ThreadPoolExecutor(max_workers=KG_WORKERS) as pool:
        futures = {
            pool.submit(_process_one_entity, ent, llm, cache_dir): ent
            for ent in entities
        }
        for future in as_completed(futures):
            completed += 1
            pbar.update(1)
            result = future.result()
            if result is None:
                continue
            cui, name, etype, triples = result
            kg.add_node(name, cui=cui, entity_type=etype)
            for subj, pred, obj in triples:
                kg.add_triple(subj.strip(), pred.strip(), obj.strip())
            # if completed % 100 == 0:
            #     kg.save(config.GRAPH_PATH)  # periodic checkpoint

    kg.save(config.GRAPH_PATH)
    pbar.close()
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
