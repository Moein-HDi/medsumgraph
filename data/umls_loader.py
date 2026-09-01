"""Load UMLS Metathesaurus entities (CUI, name, type, definition) from RRF files.

Requires the RRF files produced by the UMLS install script (any subset — the
'Active subset' is recommended for this project). Expected files under the
META directory:
  MRCONSO.RRF  -> preferred English concept names
  MRSTY.RRF    -> semantic types
  MRDEF.RRF    -> definitions
"""
import json
import re
from pathlib import Path

import config

# Field indices (0-based) from the actual MRCONSO.RRF rows (19 fields):
# CUI|LAT|TS|LUI|ISPREF|SUI|?|AUI|SAUI|SCUI|SAB|CODE|TTY|CODE2|STR|SRL|SUPPRESS|CVF|?
CUI_IDX, LAT_IDX, ISPREF_IDX, SAB_IDX, STR_IDX = 0, 1, 4, 10, 14
# MRDEF actual layout: CUI|AUI|ATUI|SATUI|SAB|DEF|SUPPRESS|CVF|?
MRDEF_CUI_IDX, MRDEF_DEF_IDX = 0, 5
# MRSTY: CUI|TUI|STN|STY|...
MRSTY_CUI_IDX, MRSTY_STY_IDX = 0, 3


def find_meta_dir() -> Path:
    """Locate the META directory containing the RRF files.

    Resolution order: explicit UMLS_META_DIR -> known install roots with the
    standard <release>/META or data/<release>/META layout.
    """
    if config.UMLS_META_DIR:
        p = Path(config.UMLS_META_DIR)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"UMLS_META_DIR does not exist: {p}")

    roots = [
        Path(r"D:\UMLS"),
        Path(r"D:\Moein\Uni\datasets\UMLS"),
        Path(r"D:\Moein\Uni\datasets\UMLS\data"),
        Path.home() / "umls",
    ]
    releases = ("2026AA", "2026AB", "2025AB", "2025AA", "2024AB", "2024AA", "2023AB")
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root, *(root / r for r in releases), *(root / r / "META" for r in releases)]
        for cand in candidates:
            if (cand / "MRCONSO.RRF").exists():
                return cand
    raise FileNotFoundError(
        "Could not find UMLS META directory. Set UMLS_META_DIR in .env "
        "or place RRF files under <umls-install>/<release>/META/"
    )


def load_entities(
    meta_dir: Path | None = None, allowed_types: set[str] | None = None
) -> list[dict]:
    """Return [{cui, name, type, definition}] for concepts in the subset.

    If allowed_types is given, only concepts whose semantic type is in the
    set are kept (see config.ALLOWED_SEMANTIC_TYPES).
    """
    meta = meta_dir or find_meta_dir()
    names = _load_names(meta / "MRCONSO.RRF")
    types = _load_types(meta / "MRSTY.RRF")
    defs = _load_definitions(meta / "MRDEF.RRF")

    entities = []
    for cui in names:
        if allowed_types is not None and types.get(cui, "") not in allowed_types:
            continue
        entities.append(
            {
                "cui": cui,
                "name": names[cui],
                "type": types.get(cui, ""),
                "definition": defs.get(cui, ""),
            }
        )
    return entities


NAME_INDEX_CACHE = config.CACHE_DIR / "umls_name_index.json"


def build_name_index(meta_dir: Path | None = None) -> dict[str, str]:
    """Return {cui: preferred English name} for all concepts (all types).

    Used to resolve MedQA-extracted entity mentions back to CUIs. The result
    is cached to disk (parsing the full MRCONSO takes minutes), and the cache
    is invalidated if the source file changes.
    """
    meta = meta_dir or find_meta_dir()
    src = meta / "MRCONSO.RRF"

    if NAME_INDEX_CACHE.exists():
        cached_mtime = None
        marker = NAME_INDEX_CACHE.with_suffix(".mtime")
        if marker.exists():
            try:
                cached_mtime = float(marker.read_text())
            except ValueError:
                cached_mtime = None
        try:
            src_mtime = src.stat().st_mtime
        except OSError:
            src_mtime = None
        if cached_mtime is not None and cached_mtime == src_mtime:
            import json

            with open(NAME_INDEX_CACHE, encoding="utf-8") as f:
                return json.load(f)

    index = _load_names(src)
    import json

    NAME_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(NAME_INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    try:
        NAME_INDEX_CACHE.with_suffix(".mtime").write_text(str(src.stat().st_mtime))
    except OSError:
        pass
    return index


def _load_names(path: Path) -> dict[str, str]:
    """cui -> preferred English name.

    UMLS marks the preferred term with ISPREF == "PF" (older releases used
    "Y"). A CUI can have many preferred names across vocabularies; we pick
    the one that appears most often (frequency), breaking ties by shortest
    length. This selects the common name ("ampicillin") over a one-off
    abbreviation ("AP").
    """
    counts: dict[str, dict[str, int]] = {}
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= STR_IDX:
                continue
            if parts[LAT_IDX] != "ENG":
                continue
            if parts[ISPREF_IDX] not in ("PF", "Y"):
                continue
            cui = parts[CUI_IDX]
            name = parts[STR_IDX]
            by_name = counts.setdefault(cui, {})
            by_name[name] = by_name.get(name, 0) + 1

    result: dict[str, str] = {}
    for cui, by_name in counts.items():
        best = max(
            by_name.items(), key=lambda kv: (kv[1], -len(kv[0]))
        )[0]
        result[cui] = best
    return result


SYNONYM_INDEX_CACHE = config.CACHE_DIR / "umls_synonym_index.json"


def build_synonym_index(meta_dir: Path | None = None, max_len: int = 40) -> dict[str, str]:
    """Return {normalized STR: CUI} for every English STR up to max_len chars.

    Used to resolve LLM-extracted entity mentions (e.g. "Ampicillin") to a
    CUI even when the CUI's preferred name differs. Cached to disk.
    """
    meta = meta_dir or find_meta_dir()
    src = meta / "MRCONSO.RRF"

    if SYNONYM_INDEX_CACHE.exists():
        marker = SYNONYM_INDEX_CACHE.with_suffix(".mtime")
        if marker.exists():
            try:
                cached_mtime = float(marker.read_text())
            except ValueError:
                cached_mtime = None
            try:
                src_mtime = src.stat().st_mtime
            except OSError:
                src_mtime = None
            if cached_mtime is not None and cached_mtime == src_mtime:
                with open(SYNONYM_INDEX_CACHE, encoding="utf-8") as f:
                    return json.load(f)

    result: dict[str, str] = {}
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= STR_IDX:
                continue
            if parts[LAT_IDX] != "ENG":
                continue
            name = parts[STR_IDX]
            if not name or len(name) > max_len:
                continue
            norm = _norm_name(name)
            if norm:
                result.setdefault(norm, parts[CUI_IDX])

    SYNONYM_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNONYM_INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    try:
        SYNONYM_INDEX_CACHE.with_suffix(".mtime").write_text(str(src.stat().st_mtime))
    except OSError:
        pass
    return result


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _load_types(path: Path) -> dict[str, str]:
    """cui -> first semantic type."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= MRSTY_STY_IDX:
                continue
            cui = parts[MRSTY_CUI_IDX]
            if cui not in result:
                result[cui] = parts[MRSTY_STY_IDX]
    return result


def _load_definitions(path: Path) -> dict[str, str]:
    """cui -> first definition."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= MRDEF_DEF_IDX:
                continue
            cui = parts[MRDEF_CUI_IDX]
            if cui not in result:
                result[cui] = parts[MRDEF_DEF_IDX]
    return result
