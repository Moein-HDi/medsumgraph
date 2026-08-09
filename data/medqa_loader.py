"""MedQA (USMLE) dataset loading from HuggingFace.

Train split serves as the labeled few-shot example bank; test split is the
evaluation set. Format: {"question", "options": {A..}, "answer_idx": 0..3,
"answer": "A"|... }
"""
from datasets import load_dataset

# MedQA-USMLE 4-options HuggingFace port (paper's benchmark)
HF_DATASET = "GBaker/MedQA-USMLE-4-options-hf"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def _normalize(row: dict) -> dict:
    options = {
        "A": row["ending0"],
        "B": row["ending1"],
        "C": row["ending2"],
        "D": row["ending3"],
    }
    answer_idx = int(row["label"])
    return {
        "question": row["sent1"],
        "options": options,
        "answer_idx": answer_idx,
        "answer": chr(ord("A") + answer_idx),
    }


def load_train() -> list[dict]:
    ds = load_dataset(HF_DATASET, split=TRAIN_SPLIT)
    return [_normalize(r) for r in ds]


def load_test(limit: int | None = None) -> list[dict]:
    ds = load_dataset(HF_DATASET, split=TEST_SPLIT)
    rows = [_normalize(r) for r in ds]
    if limit is not None:
        rows = rows[:limit]
    return rows
