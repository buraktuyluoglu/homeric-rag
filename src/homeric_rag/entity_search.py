"""Entity-aware retrieval: resolve query entities, expand, fuse.

The variant this repo exists to demonstrate. On top of the plain hybrid
baseline (search.py), it:

1. Resolves character mentions in the *query* against the roster. Query
   matching is case-insensitive (people type "achilles") and, unlike the
   corpus-side matcher, also accepts conventional Greek names ("Odysseus",
   "Zeus") that never occur in Butler's Roman-name prose — exactly the
   vocabulary gap entity resolution is for.
2. Disambiguates within the query where possible: if two mentions'
   candidate sets intersect in exactly one character ("Ajax" x "son of
   Telamon"), both resolve to it. Queries are short, so the intersection
   runs over all mention pairs, not just adjacent ones.
3. Expands the query with each resolved character's canonical name,
   Greek name, and aliases, so BM25 and the dense retriever see the
   surface forms the corpus actually uses.
4. Fuses with RRF (k=60) over three rankings: bm25(expanded),
   dense(expanded), and an entity-match ranking — chunks ordered by how
   often the target characters are attributed in chunk_entities.jsonl.
   Rank-only fusion needs no score-scale tuning, which is why it is
   preferred here over an additive boost.

Honest behaviors, by design:

- A query surface that stays ambiguous ("Ajax" alone) contributes *all*
  candidates to expansion and to the entity ranking, and the result is
  flagged so the caller can say so. Guessing silently would be worse.
- Index-time ambiguous mentions (a bare "Ajax" in a chunk) were left
  unattributed by the entity layer and therefore do not feed the entity
  ranking. Only resolved attributions count.
- A query with no recognized entities falls back to the plain hybrid
  ranking unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from homeric_rag.entities import default_index
from homeric_rag.search import DATA_DIR, Ranking, SearchIndex, rrf_fuse


@dataclass(frozen=True)
class QueryEntities:
    """Entity mentions found in a query."""

    resolved: dict[str, str] = field(default_factory=dict)  # surface -> char_id
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    collective: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def target_ids(self) -> tuple[str, ...]:
        """Every character the query may be about, ambiguous candidates
        included, deduplicated, in stable order."""
        ids: list[str] = list(self.resolved.values())
        for candidates in (*self.ambiguous.values(), *self.collective.values()):
            ids.extend(candidates)
        return tuple(dict.fromkeys(ids))

    @property
    def is_empty(self) -> bool:
        return not (self.resolved or self.ambiguous or self.collective)


@dataclass(frozen=True)
class EntitySearchResult:
    query: str
    expanded_query: str
    entities: QueryEntities
    expansion_terms: tuple[str, ...]
    ranking: Ranking
    # Component dense ranking over expanded_query (== query when no
    # entities resolved), exposed so callers needing the top cosine do
    # not have to re-embed the query.
    dense_ranking: Ranking


class QueryResolver:
    """Case-insensitive surface matching over the roster, Greek names
    included. Corpus-side matching (entities.py) stays case-sensitive and
    corpus-verified; the query side meets users where they type."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        index = default_index(data_dir)
        self.characters = index.characters

        surface_to_ids: dict[str, set[str]] = {}
        self._display: dict[str, str] = {}  # folded -> original casing
        for char in self.characters.values():
            forms = char["aliases"] + char["epithets"] + char["patronymics"]
            if char.get("greek_name"):
                forms = forms + [char["greek_name"]]
            for form in forms:
                key = form.lower()
                surface_to_ids.setdefault(key, set()).add(char["id"])
                self._display.setdefault(key, form)
        self._entity_surfaces = {k: tuple(sorted(v)) for k, v in surface_to_ids.items()}

        self._collective_surfaces = {
            form.lower(): tuple(coll["members"])
            for coll in index.collectives.values()
            for form in coll["surface_forms"]
        }
        for coll in index.collectives.values():
            for form in coll["surface_forms"]:
                self._display.setdefault(form.lower(), form)

        all_forms = sorted(
            set(self._entity_surfaces) | set(self._collective_surfaces), key=len, reverse=True
        )
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(f) for f in all_forms) + r")(?!\w)",
            re.IGNORECASE,
        )

    def resolve(self, query: str) -> QueryEntities:
        mentions: list[tuple[str, set[str]]] = []  # (folded surface, candidates)
        collective: dict[str, tuple[str, ...]] = {}
        for match in self._pattern.finditer(query):
            key = match.group().lower()
            if key in self._collective_surfaces:
                collective[self._display[key]] = self._collective_surfaces[key]
            else:
                mentions.append((key, set(self._entity_surfaces[key])))

        # Pairwise disambiguation: a singleton intersection resolves both.
        changed = True
        while changed:
            changed = False
            for i, (_, a) in enumerate(mentions):
                for j, (_, b) in enumerate(mentions):
                    if i >= j:
                        continue
                    shared = a & b
                    if len(shared) == 1 and (len(a) > 1 or len(b) > 1):
                        mentions[i] = (mentions[i][0], set(shared))
                        mentions[j] = (mentions[j][0], set(shared))
                        changed = True

        resolved: dict[str, str] = {}
        ambiguous: dict[str, tuple[str, ...]] = {}
        for key, candidates in mentions:
            surface = self._display[key]
            if len(candidates) == 1:
                resolved[surface] = next(iter(candidates))
            else:
                ambiguous[surface] = tuple(sorted(candidates))
        return QueryEntities(resolved=resolved, ambiguous=ambiguous, collective=collective)

    def expansion_terms(self, entities: QueryEntities) -> tuple[str, ...]:
        """Canonical name, Greek name, and aliases for every target
        character, deduplicated case-insensitively."""
        terms: dict[str, str] = {}
        for char_id in entities.target_ids:
            char = self.characters[char_id]
            for form in [char["canonical_name"], char.get("greek_name"), *char["aliases"]]:
                if form:
                    terms.setdefault(form.lower(), form)
        return tuple(terms.values())


class EntitySearch:
    """Entity-aware search over a SearchIndex."""

    def __init__(self, index: SearchIndex | None = None, data_dir: Path = DATA_DIR) -> None:
        self.index = index or SearchIndex(data_dir)
        self.data_dir = self.index.data_dir
        self.resolver = QueryResolver(self.data_dir)

    @cached_property
    def chunk_entities(self) -> dict[str, dict[str, int]]:
        """chunk_id -> {char_id: resolved attribution count}, corpus order."""
        lines = (self.data_dir / "chunk_entities.jsonl").read_text(encoding="utf-8").splitlines()
        return {r["id"]: r["entities"] for r in map(json.loads, lines)}

    def entity_rank(self, target_ids: tuple[str, ...]) -> Ranking:
        """Chunks that mention any target character, ordered by total
        attribution count descending; ties keep corpus order. Covers only
        matching chunks — RRF treats absence as no contribution."""
        targets = set(target_ids)
        scored = [
            (chunk_id, float(sum(count for cid, count in ents.items() if cid in targets)))
            for chunk_id, ents in self.chunk_entities.items()
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def search(self, query: str) -> EntitySearchResult:
        entities = self.resolver.resolve(query)
        if entities.is_empty:
            dense = self.index.dense_rank(query)
            return EntitySearchResult(
                query=query,
                expanded_query=query,
                entities=entities,
                expansion_terms=(),
                ranking=rrf_fuse([self.index.bm25_rank(query), dense]),
                dense_ranking=dense,
            )

        terms = self.resolver.expansion_terms(entities)
        new_terms = [t for t in terms if t.lower() not in query.lower()]
        expanded = query + (" " + " ".join(new_terms) if new_terms else "")

        dense = self.index.dense_rank(expanded)
        ranking = rrf_fuse(
            [
                self.index.bm25_rank(expanded),
                dense,
                self.entity_rank(entities.target_ids),
            ]
        )
        return EntitySearchResult(
            query=query,
            expanded_query=expanded,
            entities=entities,
            expansion_terms=terms,
            ranking=ranking,
            dense_ranking=dense,
        )
