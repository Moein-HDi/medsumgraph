"""MedQA (USMLE) dataset loading, with local disk cache.

The first run downloads from HuggingFace and stores the normalized rows as
JSON files under cache/medqa/. Subsequent runs load from disk — no network.

Train split serves as the labeled few-shot example bank; test split is the
evaluation set. Format: {"question", "options": {A..}, "answer_idx": 0..3,
"answer": "A"|... }
"""
import json

from datasets import load_dataset

import config

# MedQA-USMLE 4-options HuggingFace port (paper's benchmark)
HF_DATASET = "GBaker/MedQA-USMLE-4-options-hf"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

CACHE_DIR = config.CACHE_DIR / "medqa"
TRAIN_CACHE = CACHE_DIR / "train.json"
TEST_CACHE = CACHE_DIR / "test.json"


def _normalize(row: dict) -> dict:
    options = {
        "A": row["ending0"],
        "B": row["ending1"],
        "C": row["ending2"],
        "D": row["ending3"],
    }
    answer_idx = int(row["label"])
    return {
        "id": row.get("id", ""),
        "question": row["sent1"],
        "options": options,
        "answer_idx": answer_idx,
        "answer": chr(ord("A") + answer_idx),
    }


def _from_hf(split: str) -> list[dict]:
    ds = load_dataset(HF_DATASET, split=split)
    return [_normalize(r) for r in ds]


def _load_or_build(cache_path, split: str) -> list[dict]:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    rows = _from_hf(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return rows


def load_train() -> list[dict]:
    return _load_or_build(TRAIN_CACHE, TRAIN_SPLIT)


def load_test(limit: int | None = None) -> list[dict]:
    rows = _load_or_build(TEST_CACHE, TEST_SPLIT)
    if limit is not None:
        rows = rows[:limit]
    return rows
