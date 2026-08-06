"""Behavior tests for entity-aware retrieval.

Note on the epithet case: Butler's prose drops the Greek epithets — the
roster records that "swift-footed" never occurs in the corpus — so the
epithet-style queries here use surfaces Butler actually wrote
(patronymics such as "son of Peleus") plus the Greek-name gap
("Odysseus" appears in queries but never in the text, which only uses
"Ulysses"). Dense query embedding is injected as a deterministic
stand-in so tests stay offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from homeric_rag.entity_search import EntitySearch, QueryResolver
from homeric_rag.search import SearchIndex


def fake_embed(query: str) -> np.ndarray:
    rng = np.random.default_rng(len(query))
    vec = rng.standard_normal(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="module")
def searcher() -> EntitySearch:
    return EntitySearch(SearchIndex(embed_fn=fake_embed))


@pytest.fixture(scope="module")
def resolver(searcher: EntitySearch) -> QueryResolver:
    return searcher.resolver


class TestQueryResolution:
    def test_greek_name_resolves_to_roman_character(self, resolver: QueryResolver):
        ents = resolver.resolve("how does Odysseus escape the Cyclops")
        assert ents.resolved.get("Odysseus") == "odysseus"

    def test_case_insensitive(self, resolver: QueryResolver):
        ents = resolver.resolve("the death of hector")
        assert ents.resolved.get("Hector") == "hector"

    def test_bare_ajax_stays_ambiguous(self, resolver: QueryResolver):
        ents = resolver.resolve("Ajax in battle")
        (candidates,) = ents.ambiguous.values()
        assert set(candidates) == {"ajax_oileus", "ajax_telamon"}
        assert not ents.resolved

    def test_intersection_disambiguates_within_query(self, resolver: QueryResolver):
        ents = resolver.resolve("Ajax son of Telamon defends the ships")
        assert "ajax_telamon" in ents.resolved.values()
        assert not ents.ambiguous

    def test_patronymic_epithet_resolves(self, resolver: QueryResolver):
        ents = resolver.resolve("the anger of the son of Peleus")
        assert ents.resolved.get("son of Peleus") == "achilles"

    def test_expansion_covers_corpus_surface_forms(self, resolver: QueryResolver):
        ents = resolver.resolve("Odysseus")
        terms = resolver.expansion_terms(ents)
        assert "Ulysses" in terms  # the form Butler actually uses

    def test_no_entities(self, resolver: QueryResolver):
        assert resolver.resolve("ships drawn up on the shore").is_empty


class TestEntityRanking:
    def test_epithet_query_expansion_changes_ranking(self, searcher: EntitySearch):
        """A patronymic-only query: entity mode must pull in chunks
        attributed to Achilles that the plain query text alone does not."""
        query = "the anger of the son of Peleus"
        entity_top = [cid for cid, _ in searcher.search(query).ranking[:10]]
        plain_top = [cid for cid, _ in searcher.index.hybrid_rank(query)[:10]]
        assert entity_top != plain_top

        def achilles_hits(chunk_ids: list[str]) -> int:
            return sum(1 for cid in chunk_ids if "achilles" in searcher.chunk_entities.get(cid, {}))

        assert achilles_hits(entity_top) > achilles_hits(plain_top)

    def test_greek_name_query_finds_ulysses_chunks(self, searcher: EntitySearch):
        """'Odysseus' never occurs in Butler's text; without entity
        expansion BM25 gets zero signal from the name."""
        result = searcher.search("Odysseus and the suitors")
        top = [cid for cid, _ in result.ranking[:10]]
        odysseus_hits = sum(1 for cid in top if "odysseus" in searcher.chunk_entities.get(cid, {}))
        assert odysseus_hits >= 5
        assert "Ulysses" in result.expanded_query

    def test_ambiguous_query_flagged_and_covers_both(self, searcher: EntitySearch):
        result = searcher.search("Ajax in battle")
        (candidates,) = result.entities.ambiguous.values()
        assert set(candidates) == {"ajax_oileus", "ajax_telamon"}
        # both candidates feed the entity ranking
        entity_ranking = dict(searcher.entity_rank(result.entities.target_ids))
        only_telamon = dict(searcher.entity_rank(("ajax_telamon",)))
        assert set(entity_ranking) > set(only_telamon)

    def test_entity_mode_deterministic(self, searcher: EntitySearch):
        q = "Achilles fights Hector"
        assert searcher.search(q).ranking == searcher.search(q).ranking

    def test_no_entities_falls_back_to_plain_hybrid(self, searcher: EntitySearch):
        q = "ships drawn up on the shore"
        result = searcher.search(q)
        assert result.entities.is_empty
        assert result.expanded_query == q
        assert result.ranking == searcher.index.hybrid_rank(q)

    def test_entity_rank_orders_by_attribution_count(self, searcher: EntitySearch):
        ranking = searcher.entity_rank(("achilles",))
        counts = [score for _, score in ranking]
        assert counts == sorted(counts, reverse=True)
        assert all(score > 0 for score in counts)
