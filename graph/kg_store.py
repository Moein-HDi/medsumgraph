"""In-memory knowledge graph with name-based lookup and subgraph extraction."""
from __future__ import annotations

import json
from typing import Iterator


class KnowledgeGraph:
    def __init__(self):
        # node -> list of (predicate, object)
        self._edges: dict[str, list[tuple[str, str]]] = {}
        # normalized name -> canonical node name
        self._index: dict[str, str] = {}
        # cui -> name (for entities sourced from UMLS)
        self._cui_to_name: dict[str, str] = {}
        self._names: dict[str, str] = {}

    @staticmethod
    def _norm(name: str) -> str:
        # strip punctuation/possessives/extra spaces for matching
        import re

        return re.sub(r"[^a-z0-9 ]", "", name.strip().lower()).strip()

    def add_node(self, name: str, cui: str = "", entity_type: str = ""):
        if not name:
            return
        if name not in self._edges:
            self._edges[name] = []
            self._names[name] = entity_type
        if cui and cui not in self._cui_to_name:
            self._cui_to_name[cui] = name
        self._index.setdefault(self._norm(name), name)

    def add_triple(self, subject: str, predicate: str, obj: str):
        self.add_node(subject)
        self.add_node(obj)
        self._edges[subject].append((predicate, obj))

    def resolve(self, name: str) -> str | None:
        """Canonical node name for a possibly-fuzzy entity mention."""
        exact = self._index.get(self._norm(name))
        if exact:
            return exact
        norm = self._norm(name)
        tokens = set(norm.split())
        if not tokens:
            return None
        best: tuple[int, str] | None = None
        for key, canonical in self._index.items():
            key_tokens = set(key.split())
            # require full containment of the smaller token set in the larger
            smaller, larger = (tokens, key_tokens) if len(tokens) <= len(key_tokens) else (key_tokens, tokens)
            if smaller and smaller <= larger:
                overlap = len(smaller)
                if best is None or overlap > best[0]:
                    best = (overlap, canonical)
        return best[1] if best else None

    def neighbors(self, node: str) -> list[tuple[str, str]]:
        return self._edges.get(node, [])

    def subgraph(self, entities: list[str], depth: int = 1) -> list[tuple[str, str, str]]:
        """Collect triples (subject, predicate, object) around the given entities."""
        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        frontier = set()

        for ent in entities:
            resolved = self.resolve(ent)
            if resolved:
                frontier.add(resolved)

        for node in frontier:
            for pred, obj in self.neighbors(node):
                t = (node, pred, obj)
                if t not in seen:
                    seen.add(t)
                    triples.append(t)
            # reverse edges (object -> subject) to catch "X treated_with Y" lookups from Y
            for src, out_edges in self._edges.items():
                for pred, obj in out_edges:
                    if obj == node or src == node:
                        t = (src, pred, obj)
                        if t not in seen:
                            seen.add(t)
                            triples.append(t)
        return triples

    def all_triples(self) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for src, out_edges in self._edges.items():
            for pred, obj in out_edges:
                triples.append((src, pred, obj))
        return triples

    def save(self, path):
        payload = {
            "edges": self._edges,
            "index": self._index,
            "cui_to_name": self._cui_to_name,
            "names": self._names,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path) -> "KnowledgeGraph":
        kg = cls()
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        kg._edges = payload["edges"]
        kg._index = payload["index"]
        kg._cui_to_name = payload.get("cui_to_name", {})
        kg._names = payload.get("names", {})
        return kg
