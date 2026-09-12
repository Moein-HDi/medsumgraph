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
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("VERBOSE", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-index", type=int, default=10,
                        help="Index into the test set (0-1272). Default 10.")
    args = parser.parse_args()

    # --- Step 1: Load the graph (uses existing cache, no LLM calls) ---
    import config
    from data.kg_builder import load_or_build
    from data.medqa_loader import load_test, load_train
    from retrieval import hybrid_retrieve
    from llm_client import LLMClient
    from evaluate import FewShotBank
    import prompts

    print("Loading graph...")
    t0 = time.time()
    kg = load_or_build()
    triples_count = sum(len(v) for v in kg._edges.values())
    print(f"  Graph loaded in {time.time()-t0:.1f}s: {len(kg._edges)} nodes, {triples_count} triples")

    # --- Step 2: Pick a question ---
    test = load_test()
    q = test[args.question_index]
    print(f"\nQuestion #{args.question_index}: {q['question'][:120]}")
    print(f"Gold answer: {q['answer']}")
    for k, v in q["options"].items():
        print(f"  {k}: {v[:80]}")

    # --- Step 3: Retrieval (1 LLM call for global + 1 for local) ---
    llm = LLMClient()
    print("\nRunning hybrid retrieval...")
    t1 = time.time()
    triples = hybrid_retrieve(llm, kg, q["question"])
    print(f"  Retrieved {len(triples)} triples in {time.time()-t1:.1f}s")
    if triples:
        print("  Top triples:")
        for s, p, o in triples[:5]:
            print(f"    ({s}, {p}, {o})")

    # --- Step 4: Few-shot retrieval (instant, from cache) ---
    print("\nRetrieving few-shot examples...")
    t2 = time.time()
    fewshot = FewShotBank(load_train())
    examples = fewshot.retrieve(q["question"])
    print(f"  Selected {len(examples)} examples in {time.time()-t2:.1f}s")
    for i, ex in enumerate(examples):
        print(f"    Example {i+1} (answer {ex['answer']}): {ex['question'][:80]}")

    # --- Step 5: Build the prompt ---
    kg_context = prompts.build_kg_context(triples)
    prompt = prompts.MEDSUMGRAPH_USER.format(
        fewshots=prompts.build_fewshots(examples),
        kg_context=kg_context,
        question=q["question"],
        options=prompts.format_options(q["options"]),
    )
    print(f"\nPrompt: {len(prompt)} chars")
    print(f"Prompt preview (first 300): {prompt[:300]}")

    # --- Step 6: LLM call ---
    print(f"\nCalling LLM ({config.OPENROUTER_MODEL})...")
    t3 = time.time()
    try:
        response = llm.chat(
            prompts.MEDSUMGRAPH_SYSTEM, prompt, temperature=0.5
        )
        elapsed = time.time() - t3
        print(f"\nLLM response ({elapsed:.1f}s):")
        print(f"  {response[:500]}")
    except Exception as e:
        print(f"\nLLM ERROR: {e}")
        sys.exit(1)

    # --- Step 7: Extract answer ---
    m = re.search(r"\b([A-D])\b", response)
    predicted = m.group(1) if m else "NONE"
    print(f"\nPredicted: {predicted} | Gold: {q['answer']} | {'CORRECT' if predicted == q['answer'] else 'WRONG'}")


if __name__ == "__main__":
    main()
