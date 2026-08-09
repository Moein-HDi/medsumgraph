"""CLI entry point for the MedSumGraph pipeline.

Usage:
  python run_pipeline.py build-kg [--limit-entities N]
  python run_pipeline.py evaluate [--limit N] [--baseline] [--out results.json]
"""
import argparse

import config
from data import kg_builder, umls_loader
from data.kg_builder import KnowledgeGraph
from llm_client import LLMClient


def cmd_build_kg(args):
    entities = umls_loader.load_entities()
    print(f"Loaded {len(entities)} entities from UMLS")
    kg = kg_builder.build_knowledge_graph(
        entities, LLMClient(), KnowledgeGraph(), limit=args.limit_entities
    )
    print(f"Graph saved to {config.GRAPH_PATH} with {len(kg.all_triples())} triples")


def cmd_evaluate(args):
    import evaluate

    evaluate.evaluate(limit=args.limit, baseline=args.baseline, out_path=args.out)


def main():
    parser = argparse.ArgumentParser(description="MedSumGraph pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-kg", help="Phase 1: construct the knowledge graph")
    build.add_argument("--limit-entities", type=int, default=None, help="Process only N entities")
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
