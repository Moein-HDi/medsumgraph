"""MedQA evaluation: dynamic few-shot + hybrid retrieval + ensemble majority vote."""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
import prompts
from data import kg_builder, medqa_loader
from embedder import embed, cosine_similarity
from graph.kg_store import KnowledgeGraph
from llm_client import LLMClient
from retrieval import hybrid_retrieve

ANSWER_RE = re.compile(r"\b([A-D])\b")


def _answer_letter(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None


class FewShotBank:
    def __init__(self, examples: list[dict]):
        self.examples = examples
        self._vectors = None

    def _vectors_cache(self) -> np.ndarray:
        if self._vectors is None:
            cache_path = config.EMBED_CACHE_DIR / "medqa_train_embeddings.npy"
            if cache_path.exists():
                self._vectors = np.load(cache_path)
            else:
                self._vectors = embed([self._text(e) for e in self.examples])
                np.save(cache_path, self._vectors)
        return self._vectors

    @staticmethod
    def _text(e: dict) -> str:
        return e["question"] + " " + " ".join(e["options"].values())

    def retrieve(self, question: str, k: int = config.FEWSHOT_TOP_K) -> list[dict]:
        qvec = embed([question])[0]
        sims = cosine_similarity(qvec, self._vectors_cache())
        order = np.argsort(-sims)[:k]
        return [self.examples[i] for i in order]


def run_question(
    llm: LLMClient,
    kg: KnowledgeGraph,
    fewshot: FewShotBank,
    q: dict,
    baseline: bool = False,
) -> dict:
    if baseline:
        prompt = prompts.BASELINE_USER.format(
            question=q["question"], options=prompts.format_options(q["options"])
        )
        system = prompts.BASELINE_SYSTEM
        context = ""
    else:
        examples = fewshot.retrieve(q["question"])
        triples = hybrid_retrieve(llm, kg, q["question"])
        context = prompts.build_kg_context(triples)
        prompt = prompts.MEDSUMGRAPH_USER.format(
            fewshots=prompts.build_fewshots(examples),
            kg_context=context,
            question=q["question"],
            options=prompts.format_options(q["options"]),
        )
        system = prompts.MEDSUMGRAPH_SYSTEM

    votes: list[str] = []
    for _ in range(config.ENSEMBLE_SIZE):
        resp = llm.chat(system, prompt)
        votes.append(_answer_letter(resp) or "")

    counter = Counter(v for v in votes if v)
    if counter:
        answer, count = counter.most_common(1)[0]
        majority = answer if count >= (config.ENSEMBLE_SIZE + 1) // 2 else ""
    else:
        majority = ""

    return {
        "question": q["question"],
        "gold": q["answer"],
        "votes": votes,
        "predicted": majority or (votes[0] if votes else ""),
        "context": context,
    }


def evaluate(
    limit: int | None = None,
    baseline: bool = False,
    out_path: Path | None = None,
) -> float:
    llm = LLMClient()
    kg = kg_builder.load_or_build()
    train = medqa_loader.load_train()
    test = medqa_loader.load_test(limit)

    fewshot = FewShotBank(train)
    results = []
    for q in tqdm(test, desc="Evaluating"):
        results.append(run_question(llm, kg, fewshot, q, baseline=baseline))

    correct = sum(1 for r in results if r["predicted"] == r["gold"])
    accuracy = correct / len(results) if results else 0.0

    report = {
        "n": len(results),
        "accuracy": accuracy,
        "results": results,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    return accuracy
