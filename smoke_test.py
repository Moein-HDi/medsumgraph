"""Quick end-to-end smoke test of the full MedSumGraph pipeline.

Tests on a SINGLE question (from the test set) and prints:
  - How many entities were retrieved (global + local)
  - The reranked triples fed to the LLM
  - The few-shot examples selected
  - The raw LLM response
  - The final predicted answer vs gold

No full build or evaluation is required. Run this to verify everything
works before committing to a long overnight run.

Usage: python smoke_test.py [--question-index N] [--verbose]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VERBOSE", "1")
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("smoke_test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-index", type=int, default=10,
                        help="Index into the test set (0-1272). Default 10.")
    args = parser.parse_args()

    # --- Step 1: Load the graph ---
    import config
    from data.kg_builder import load_or_build
    from data.medqa_loader import load_test, load_train
    from graph.kg_store import KnowledgeGraph
    from retrieval import hybrid_retrieve
    from llm_client import LLMClient
    from evaluate import FewShotBank
    import prompts

    log("Loading graph...")
    t0 = time.time()
    kg = load_or_build()
    log("Graph loaded in %.1fs: %d nodes", time.time() - t0, len(kg.all_triples()))

    # --- Step 2: Pick a question ---
    test = load_test()
    q = test[args.question_index]
    log("Question #%d: %s", args.question_index, q["question"][:120])
    log("Gold answer: %s", q["answer"])
    log("Options:")
    for k, v in q["options"].items():
        log("  %s: %s", k, v[:80])

    # --- Step 3: Retrieval ---
    llm = LLMClient()
    log("Running hybrid retrieval...")
    t1 = time.time()
    triples = hybrid_retrieve(llm, kg, q["question"])
    log("Retrieved %d triples in %.1fs", len(triples), time.time() - t1)
    if triples:
        log("Top 5 triples:")
        for s, p, o in triples[:5]:
            log("  (%s, %s, %s)", s, p, o)

    # --- Step 4: Few-shot retrieval ---
    log("Building few-shot bank...")
    t2 = time.time()
    fewshot = FewShotBank(load_train())
    examples = fewshot.retrieve(q["question"])
    log("Selected %d few-shot examples in %.1fs", len(examples), time.time() - t2)
    for i, ex in enumerate(examples):
        log("  Example %d (answer %s): %s", i + 1, ex["answer"], ex["question"][:80])

    # --- Step 5: Build the prompt ---
    kg_context = prompts.build_kg_context(triples)
    prompt = prompts.MEDSUMGRAPH_USER.format(
        fewshots=prompts.build_fewshots(examples),
        kg_context=kg_context,
        question=q["question"],
        options=prompts.format_options(q["options"]),
    )
    log("Prompt length: %d chars", len(prompt))

    # --- Step 6: LLM call ---
    log("Calling LLM (model=%s)...", config.OPENROUTER_MODEL)
    t3 = time.time()
    try:
        response = llm.chat(
            prompts.MEDSUMGRAPH_SYSTEM, prompt, temperature=0.5
        )
        elapsed = time.time() - t3
        log("LLM response (%.1fs):", elapsed)
        log("  %s", response[:500])
    except Exception as e:
        log("LLM ERROR: %s", e)
        sys.exit(1)

    # --- Step 7: Extract answer ---
    import re
    m = re.search(r"\b([A-D])\b", response)
    predicted = m.group(1) if m else "NONE"
    log("Predicted answer: %s | Gold: %s | %s",
        predicted, q["answer"],
        "CORRECT" if predicted == q["answer"] else "WRONG")


if __name__ == "__main__":
    main()
