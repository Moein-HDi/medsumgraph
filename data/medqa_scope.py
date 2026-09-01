"""MedQA-relevant entity scope extraction (resumable, failure-tolerant).

Strategy (user-chosen): LLM-based medical entity extraction per question
(1 Groq call per question), then resolve each extracted entity to a UMLS CUI
via the name index.

Resumability: every question's extraction is cached to
cache/medqa_scope/<id>.json immediately after it completes. A failed call
writes nothing, so a re-run only re-processes the questions that never
finished. The final CUI list is cached at cache/medqa_scope/scope_entities.json.
"""
import json
import re
from pathlib import Path

from tqdm import tqdm

import config
import prompts
from data import medqa_loader, umls_loader
from llm_client import LLMClient
from logging_utils import log

_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_id(qid: str) -> str:
    return _ID_RE.sub("_", qid) or "unknown"


class UMLSMatcher:
    """Resolve entity mentions to CUIs using the UMLS synonym index."""

    def __init__(self, synonym_index: dict[str, str]):
        self.synonym_index = synonym_index

    def resolve(self, mention: str) -> str | None:
        """Best-effort mention -> CUI.

        Matching order: exact normalized match -> whole-word containment in a
        synonym -> token-overlap (>=2 shared tokens) with longest-overlap
        winner among synonyms.
        """
        norm = re.sub(r"[^a-z0-9 ]", "", mention.lower()).strip()
        if not norm:
            return None
        exact = self.synonym_index.get(norm)
        if exact:
            return exact

        tokens = set(norm.split())
        if not tokens:
            return None

        best: tuple[int, str] | None = None
        seen: set[str] = set()
        for name, cui in self.synonym_index.items():
            if cui in seen:
                continue
            key_tokens = set(name.split())
            overlap = len(tokens & key_tokens)
            if overlap == 0:
                continue
            # accept if the mention is fully contained in the synonym or vice
            # versa, or if they share >= 2 tokens
            covered = (
                norm in name
                or name in norm
                or (overlap >= 2 and (tokens <= key_tokens or key_tokens <= tokens))
            )
            if covered:
                if best is None or overlap > best[0]:
                    seen.add(cui)
                    best = (overlap, cui)
        return best[1] if best else None


def _extract_entities(llm: LLMClient, question: str) -> list[str]:
    resp = llm.chat(
        prompts.ENTITY_EXTRACTION_SYSTEM,
        prompts.ENTITY_EXTRACTION_USER.format(question=question),
    )
    from data.kg_builder import _parse_json_list

    data = _parse_json_list(resp)
    return [str(e).strip() for e in data if str(e).strip()]


def extract_scope(
    llm: LLMClient | None = None,
    limit_questions: int | None = None,
    split: str = "all",
    cache_dir: Path = config.SCOPE_CACHE_DIR,
) -> list[str]:
    """Return the list of CUIs relevant to MedQA questions.

    split: 'all' = train+test, 'train' = train only, 'test' = test only.
    Resumes from cache: already-processed question ids are skipped.
    """
    llm = llm or LLMClient()
    cache_dir.mkdir(parents=True, exist_ok=True)

    final_path = config.SCOPE_ENTITIES_PATH
    meta_path = final_path.with_suffix(".meta.json")

    # Check cached final list validity: count + split must match.
    cached_meta = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                cached_meta = json.load(f)
        except (json.JSONDecodeError, AttributeError):
            cached_meta = None

    if cached_meta is not None and final_path.exists():
        cached_count = cached_meta.get("questions")
        cached_split = cached_meta.get("split", "all")
        ok_count = (limit_questions is None or
                    cached_count is not None and cached_count >= limit_questions)
        ok_split = (cached_split == split)
        if ok_count and ok_split:
            with open(final_path, encoding="utf-8") as f:
                return json.load(f)

    if split == "test":
        questions = medqa_loader.load_test()
    elif split == "train":
        questions = medqa_loader.load_train()
    else:
        questions = medqa_loader.load_train() + medqa_loader.load_test()

    if limit_questions is not None:
        questions = questions[:limit_questions]

    synonym_index = umls_loader.build_synonym_index()
    matcher = UMLSMatcher(synonym_index)

    cuis: set[str] = set()
    for q in tqdm(questions, desc="Extracting MedQA scope"):
        qid = _safe_id(q["id"])
        cache_path = cache_dir / f"{qid}.json"
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                entities = json.load(f)
        else:
            text = q["question"] + "\n" + "\n".join(q["options"].values())
            log("Scope: extracting entities for %s", qid)
            try:
                entities = _extract_entities(llm, text)
            except Exception:
                entities = []  # leave no cache; will retry on next run
            if entities:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(entities, f, ensure_ascii=False)

        for ent in entities:
            cui = matcher.resolve(ent)
            if cui:
                cuis.add(cui)

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(sorted(cuis), f, ensure_ascii=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"questions": len(questions)}, f, ensure_ascii=False)
    return sorted(cuis)


def entities_in_scope(scope_cuis: set[str], entities: list[dict]) -> list[dict]:
    return [e for e in entities if e["cui"] in scope_cuis]
