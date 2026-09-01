"""CLI entry point for the MedSumGraph pipeline.

Usage:
  python run_pipeline.py build-kg [--scope medqa] [--no-type-filter]
                                 [--limit-entities N] [--limit-questions N]
  python run_pipeline.py evaluate [--limit N] [--baseline] [--out results.json]
"""
import argparse

import config
from data import kg_builder, umls_loader
from data.kg_builder import KnowledgeGraph
from llm_client import LLMClient


def cmd_build_kg(args):
    if args.from_cache:
        # Rebuild the graph JSON purely from already-cached entities — no
        # scope extraction, no LLM calls, no Wikipedia.
        allowed_types = None if args.no_type_filter else config.ALLOWED_SEMANTIC_TYPES
        entities = umls_loader.load_entities(allowed_types=allowed_types)
        if args.scope == "medqa":
            from data import medqa_scope

            scope_cuis = set()
            try:
                scope_cuis = set(
                    medqa_scope.extract_scope(limit_questions=args.limit_questions,
                                             split=args.scope_split)
                )
            except Exception as e:
                print(f"[build-kg --from-cache] scope unavailable ({e}); using all cached entities")
            entities = medqa_scope.entities_in_scope(scope_cuis, entities) if scope_cuis else entities
        kg = kg_builder.build_knowledge_graph_from_cache(entities, KnowledgeGraph())
        print(f"Graph rebuilt from cache ({len(kg.all_triples())} triples) -> {config.GRAPH_PATH}")
        return

    allowed_types = None if args.no_type_filter else config.ALLOWED_SEMANTIC_TYPES
    entities = umls_loader.load_entities(allowed_types=allowed_types)
    print(f"Loaded {len(entities)} entities from UMLS"
          + ("" if allowed_types is None else " (after semantic-type filter)"))

    if args.scope == "medqa":
        from data import medqa_scope

        scope_cuis = set(
            medqa_scope.extract_scope(limit_questions=args.limit_questions,
                                     split=args.scope_split)
        )
        entities = medqa_scope.entities_in_scope(scope_cuis, entities)
        print(f"Restricted to {len(entities)} entities relevant to MedQA")

    if args.limit_entities is not None:
        entities = entities[: args.limit_entities]

    kg = kg_builder.build_knowledge_graph(entities, LLMClient(), KnowledgeGraph())
    print(f"Graph saved to {config.GRAPH_PATH} with {len(kg.all_triples())} triples")


def cmd_evaluate(args):
    import evaluate

    evaluate.evaluate(limit=args.limit, baseline=args.baseline, out_path=args.out)


def main():
    parser = argparse.ArgumentParser(description="MedSumGraph pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-kg", help="Phase 1: construct the knowledge graph")
    build.add_argument(
        "--scope",
        choices=["all", "medqa"],
        default="medqa",
        help="'medqa' (default) restricts the build to concepts relevant to the "
        "MedQA train/test questions via LLM entity extraction (resumable); "
        "'all' processes the entire filtered UMLS subset (very slow).",
    )
    build.add_argument(
        "--no-type-filter",
        action="store_true",
        help="Disable the semantic-type filter (config.ALLOWED_SEMANTIC_TYPES).",
    )
    build.add_argument("--limit-entities", type=int, default=None, help="Process only the first N entities")
    build.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="(scope=medqa) Extract entities from only the first N MedQA questions",
    )
    build.add_argument(
        "--scope-split",
        choices=["all", "train", "test"],
        default="all",
        help="(scope=medqa) Which MedQA split to extract entities from. "
        "'test' only processes 1,273 test questions (fast for accuracy "
        "testing); 'train' uses only the 10,178 train questions; "
        "'all' (default) uses both.",
    )
    build.add_argument(
        "--from-cache",
        action="store_true",
        help="Rebuild the graph JSON from already-cached entities only — no "
        "scope extraction or LLM processing. Uses the cached scope list if "
        "available, otherwise all type-filtered entities.",
    )
    build.set_defaults(func=cmd_build_kg)

    ev = sub.add_parser("evaluate", help="Phase 2: evaluate on MedQA")
    ev.add_argument("--limit", type=int, default=None, help="Evaluate on only N questions")
    ev.add_argument("--baseline", action="store_true", help="Use the baseline prompt (no KG/few-shot)")
    ev.add_argument("--out", default=str(config.RESULTS_DIR / "results.json"), help="Output JSON path")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
