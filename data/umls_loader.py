"""Load UMLS Metathesaurus entities (CUI, name, type, definition) from RRF files.

Requires the RRF files produced by the UMLS install script (any subset — the
'Active subset' is recommended for this project). Expected files under the
META directory:
  MRCONSO.RRF  -> preferred English concept names
  MRSTY.RRF    -> semantic types
  MRDEF.RRF    -> definitions
"""
from pathlib import Path

import config

# Field indices (0-based) per UMLS RRF documentation:
# MRCONSO: CUI|LAT|TS|STT|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|...
CUI_IDX, LAT_IDX, ISPREF_IDX, SAB_IDX, STR_IDX = 0, 1, 4, 9, 12
# MRDEF: CUI|AUI|ATUI|SATUI|SAB|CODE|DEF|...
MRDEF_CUI_IDX, MRDEF_DEF_IDX = 0, 6
# MRSTY: CUI|TUI|STN|STY|...
MRSTY_CUI_IDX, MRSTY_STY_IDX = 0, 3


def find_meta_dir() -> Path:
    """Locate the META directory containing the RRF files."""
    if config.UMLS_META_DIR:
        p = Path(config.UMLS_META_DIR)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"UMLS_META_DIR does not exist: {p}")

    candidates = []
    for release in ("2026AB", "2025AB", "2024AB", "2024AA", "2023AB"):
        candidates += [
            Path(r"D:\UMLS") / release / "META",
            Path.home() / "umls" / release / "META",
            Path("D:/") / "umls" / release / "META",
        ]
    for cand in candidates:
        if (cand / "MRCONSO.RRF").exists():
            return cand
    raise FileNotFoundError(
        "Could not find UMLS META directory. Set UMLS_META_DIR in .env "
        "or place RRF files under <umls-install>/<release>/META/"
    )


def load_entities(meta_dir: Path | None = None) -> list[dict]:
    """Return [{cui, name, type, definition}] for all concepts in the subset."""
    meta = meta_dir or find_meta_dir()
    names = _load_names(meta / "MRCONSO.RRF")
    types = _load_types(meta / "MRSTY.RRF")
    defs = _load_definitions(meta / "MRDEF.RRF")

    entities = []
    for cui in names:
        entities.append(
            {
                "cui": cui,
                "name": names[cui],
                "type": types.get(cui, ""),
                "definition": defs.get(cui, ""),
            }
        )
    return entities


def _load_names(path: Path) -> dict[str, str]:
    """cui -> preferred English name."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) <= STR_IDX:
                continue
            if parts[LAT_IDX] != "ENG":
                continue
            cui = parts[CUI_IDX]
            name = parts[STR_IDX]
            if cui in result and parts[ISPREF_IDX] != "Y":
                continue
            result[cui] = name
    return result


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
