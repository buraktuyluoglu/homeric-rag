"""Plain hybrid retrieval: Okapi BM25, dense cosine, and RRF fusion.

Constants mirror hybrid-retrieval-lab (and the production system that lab
grew out of): BM25 k1=1.2, b=0.75 with the same idf formula and tokenizer;
Reciprocal Rank Fusion with k=60. Every ranker returns (chunk_id, score)
pairs ordered best-first with ties keeping corpus order (stable sorts), so
identical inputs always produce identical rankings. BM25 covers only docs
with a query-term match; dense covers the whole corpus — RRF fuses subset
rankings by giving absent docs no contribution.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

K1 = 1.2
B = 0.75
RRF_K = 60

Ranking = list[tuple[str, float]]

# Same stopword union hybrid-retrieval-lab uses.
_STOPWORD_WORDS = """
    the and for with from under over into onto via using between across
    within without toward towards are was were this that these those its
    their our has have had can will not but all any each than then when
    where which who how why what been being such per based
    of in on at by as is it be an or if so do did does done you your
    yours my me we us he she him her they
"""
STOPWORDS = frozenset(_STOPWORD_WORDS.split())

_SPLIT = re.compile(r"[^a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase, strip combining marks, fold Turkish dotless i."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("ı", "i")


def tokenize(text: str) -> list[str]:
    """Duplicates and order preserved — term frequency matters to BM25."""
    return [t for t in _SPLIT.split(fold(text)) if len(t) >= 2 and t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: dict[str, str]):
        """docs: mapping of doc id -> raw text."""
        self.ids: list[str] = list(docs)
        self.tf: list[dict[str, int]] = []
        self.lengths: list[int] = []
        for text in docs.values():
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            self.lengths.append(len(tokens))
        self.n = len(self.ids)
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.df: dict[str, int] = {}
        for counts in self.tf:
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def score(self, query_terms: set[str], i: int) -> float:
        score = 0.0
        counts = self.tf[i]
        norm = 1 - B + B * (self.lengths[i] / self.avgdl)
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            score += self.idf(term) * (tf * (K1 + 1)) / (tf + K1 * norm)
        return score

    def rank(self, query: str) -> Ranking:
        """Docs with at least one matching query term, descending; ties
        keep corpus order. Zero-score docs are dropped (mirroring
        entity_rank) so RRF fusion never receives a corpus-order signal
        from docs BM25 knows nothing about."""
        terms = set(tokenize(query))
        scored = [(self.ids[i], self.score(terms, i)) for i in range(self.n)]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: -pair[1])
        return scored


def rrf_fuse(rankings: list[Ranking], k: int = RRF_K) -> Ranking:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of
    1 / (k + rank_d), ranks 1-based. Scores are ignored — only positions
    matter, which is what makes RRF safe to fuse across incomparable
    score scales. A ranking may cover only a subset of the corpus (the
    entity ranking does); absent docs simply get no contribution."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda pair: -pair[1])


def rank_of(chunk_id: str, ranking: Ranking) -> int:
    """1-based rank of a chunk in a ranking."""
    for position, (doc_id, _score) in enumerate(ranking, start=1):
        if doc_id == chunk_id:
            return position
    raise KeyError(chunk_id)


def _default_embed_query(text: str) -> np.ndarray:
    from homeric_rag.embed import embed_query

    return embed_query(text)


class SearchIndex:
    """Lazy-loading index over data/chunks.jsonl + data/embeddings.npz.

    ``embed_fn`` maps a query string to a (384,) vector; the default runs
    the bge-small ONNX model. Tests inject a deterministic stand-in so
    they stay offline.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.embed_fn = embed_fn or _default_embed_query

    @cached_property
    def chunks(self) -> dict[str, dict]:
        lines = (self.data_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        return {c["id"]: c for c in map(json.loads, lines)}

    @cached_property
    def bm25(self) -> BM25Index:
        return BM25Index({cid: c["text"] for cid, c in self.chunks.items()})

    @cached_property
    def _embeddings(self) -> tuple[list[str], np.ndarray]:
        path = self.data_dir / "embeddings.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `python -m homeric_rag.embed` first.")
        data = np.load(path)
        return [str(i) for i in data["ids"]], data["vectors"].astype(np.float32)

    def bm25_rank(self, query: str) -> Ranking:
        return self.bm25.rank(query)

    def dense_rank(self, query: str) -> Ranking:
        """Cosine similarity, descending. Vectors are L2-normalized, so
        the dot product is the cosine; stable argsort keeps corpus order
        on ties."""
        ids, matrix = self._embeddings
        scores = matrix @ np.asarray(self.embed_fn(query), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        return [(ids[i], float(scores[i])) for i in order]

    def hybrid_rank(self, query: str) -> Ranking:
        """The plain baseline: RRF over {bm25, dense}, no entity signal."""
        return rrf_fuse([self.bm25_rank(query), self.dense_rank(query)])
