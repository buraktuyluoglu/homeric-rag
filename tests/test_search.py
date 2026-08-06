"""Behavior tests for BM25, dense cosine, and RRF fusion.

Tests stay offline: dense query embedding is injected as a deterministic
stand-in; chunk vectors come from the committed data/embeddings.npz.
"""

from __future__ import annotations

import numpy as np
import pytest

from homeric_rag.search import (
    BM25Index,
    SearchIndex,
    rank_of,
    rrf_fuse,
    tokenize,
)


def fake_embed(query: str) -> np.ndarray:
    """Deterministic offline stand-in for the ONNX query embedder:
    seeded from the query length so repeated calls agree."""
    rng = np.random.default_rng(len(query))
    vec = rng.standard_normal(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="module")
def index() -> SearchIndex:
    return SearchIndex(embed_fn=fake_embed)


class TestTokenize:
    def test_stopwords_and_short_tokens_dropped(self):
        assert tokenize("the anger of Achilles is a b great") == ["anger", "achilles", "great"]

    def test_diacritics_folded(self):
        assert tokenize("Tüylüoğlu") == ["tuyluoglu"]


class TestBM25:
    def test_term_frequency_ranks_higher(self):
        idx = BM25Index(
            {
                "a": "ships ships ships in the harbour",
                "b": "ships in the harbour",
                "c": "horses on the plain",
            }
        )
        ranking = idx.rank("ships")
        assert [doc for doc, _ in ranking[:2]] == ["a", "b"]
        assert ranking[2][1] == 0.0

    def test_ties_keep_corpus_order(self):
        idx = BM25Index({"a": "same text", "b": "same text", "c": "same text"})
        assert [doc for doc, _ in idx.rank("text")] == ["a", "b", "c"]


class TestRRF:
    def test_agreement_wins(self):
        r1 = [("x", 9.0), ("y", 5.0), ("z", 1.0)]
        r2 = [("x", 0.9), ("z", 0.5), ("y", 0.1)]
        fused = rrf_fuse([r1, r2])
        assert fused[0][0] == "x"
        assert fused[0][1] == pytest.approx(2 / 61)

    def test_scores_ignored_only_positions_matter(self):
        big = [("x", 1000.0), ("y", 999.0)]
        small = [("y", 0.002), ("x", 0.001)]
        fused = rrf_fuse([big, small])
        # symmetric positions -> identical fused scores for both docs
        assert fused[0][1] == pytest.approx(fused[1][1])

    def test_subset_ranking_contributes_nothing_for_absent_docs(self):
        full = [("x", 2.0), ("y", 1.0)]
        subset = [("y", 5.0)]
        fused = dict(rrf_fuse([full, subset]))
        assert fused["y"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["x"] == pytest.approx(1 / 61)


class TestSearchIndex:
    def test_corpus_loaded(self, index: SearchIndex):
        assert len(index.chunks) == 719
        ids, matrix = index._embeddings
        assert matrix.shape == (719, 384)
        assert list(index.chunks) == ids  # row-aligned with corpus order

    def test_bm25_finds_the_wooden_horse(self, index: SearchIndex):
        top = [doc for doc, _ in index.bm25_rank("the wooden horse of Troy")[:10]]
        assert any(doc.startswith("odyssey") for doc in top)

    def test_hybrid_deterministic(self, index: SearchIndex):
        q = "the shield of Achilles"
        assert index.hybrid_rank(q) == index.hybrid_rank(q)

    def test_rank_of(self, index: SearchIndex):
        ranking = index.bm25_rank("Achilles")
        assert rank_of(ranking[0][0], ranking) == 1
        with pytest.raises(KeyError):
            rank_of("nope", ranking)
