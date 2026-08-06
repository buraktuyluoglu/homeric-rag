"""Retrieval evaluation: plain hybrid vs entity-aware, on gold passages.

Dataset: data/eval_questions.jsonl — 51 questions in five classes (alias,
homonym, cross-book, plain-factual, no-answer). Every gold passage is a
chunk id from data/chunks.jsonl, located by reading the corpus text, not
from memory.

Metrics:

- hit@k (k=1, 3, 5): a question counts as a hit when any of its gold
  chunks appears in the top k of the ranking. Reported overall and per
  question type, for both systems. No-answer questions are excluded from
  hit@k (they have no gold chunks) and measured separately.
- No-answer refusal: retrieval always returns *something*, so refusal has
  to come from a score floor. The relevance signal used is the top-1
  dense cosine similarity (bge-small query vs chunk embeddings) — BM25
  and RRF scores are corpus- and rank-scale dependent, cosine is not.
  The floor is *measured*, not hand-picked: for each system it is set to
  the minimum top-1 cosine observed on the answerable questions in the
  same run. A no-answer question counts as refused when its top-1 cosine
  falls below that floor. By construction the false-refusal rate on
  answerable questions is zero; the honest number to read is therefore
  the refusal rate on no-answer questions alone, and the floor itself is
  written into out/results.json.

The entity-aware system scores its cosine on the *expanded* query (the
query its dense retriever actually ran), so both systems are measured on
what they retrieved with.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections import Counter
from pathlib import Path
from typing import Any

from homeric_rag.search import DATA_DIR, Ranking, SearchIndex, rrf_fuse

QUESTIONS_PATH = DATA_DIR / "eval_questions.jsonl"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"
KS = (1, 3, 5)
TYPE_ORDER = ("alias", "homonym", "cross-book", "plain-factual", "no-answer")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    seen: set[str] = set()
    for q in questions:
        if q["id"] in seen:
            raise ValueError(f"duplicate question id {q['id']}")
        seen.add(q["id"])
    return questions


def hit_at_k(ranking: Ranking, gold: list[str], k: int) -> bool:
    """True when any gold chunk id is in the top k of the ranking."""
    top = {chunk_id for chunk_id, _score in ranking[:k]}
    return any(g in top for g in gold)


def class_histogram(questions: list[dict]) -> dict[str, int]:
    counts = Counter(q["type"] for q in questions)
    return {t: counts[t] for t in TYPE_ORDER if t in counts}


def _aggregate(records: list[dict], system: str) -> dict[str, Any]:
    """hit@k means over answerable records, overall and per type."""
    answerable = [r for r in records if r["answerable"]]

    def rates(subset: list[dict]) -> dict[str, float]:
        return {
            f"hit@{k}": round(sum(r[system][f"hit@{k}"] for r in subset) / len(subset), 4)
            for k in KS
        }

    per_type = {}
    for qtype in TYPE_ORDER:
        subset = [r for r in answerable if r["type"] == qtype]
        if subset:
            per_type[qtype] = {"n": len(subset), **rates(subset)}
    return {"overall": {"n": len(answerable), **rates(answerable)}, "per_type": per_type}


def _refusal(records: list[dict], system: str) -> dict[str, Any]:
    """Floor = min top-1 cosine on answerable questions; a no-answer
    question is refused when its top-1 cosine is below the floor."""
    answerable = [r for r in records if r["answerable"]]
    no_answer = [r for r in records if not r["answerable"]]
    floor = min(r[system]["top_cosine"] for r in answerable)
    refused = [r["id"] for r in no_answer if r[system]["top_cosine"] < floor]
    return {
        "floor": round(floor, 4),
        "floor_basis": "minimum top-1 dense cosine over answerable questions in this run",
        "n_no_answer": len(no_answer),
        "n_refused": len(refused),
        "refusal_rate": round(len(refused) / len(no_answer), 4) if no_answer else None,
        "refused_ids": refused,
        "false_refusal_rate_answerable": 0.0,
        "false_refusal_note": (
            "zero by construction — the floor is calibrated as the minimum "
            "answerable top score in the same run"
        ),
    }


def run_eval(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    from homeric_rag.embed import MODEL_NAME
    from homeric_rag.entity_search import EntitySearch

    index = SearchIndex(data_dir)
    entity_search = EntitySearch(index)
    questions = load_questions(data_dir / "eval_questions.jsonl")

    records: list[dict] = []
    for q in questions:
        query, gold = q["question"], q["gold_passages"]

        # Embed each query once per system: fuse the plain baseline from
        # its component rankings and take the entity-side cosine from the
        # dense ranking the entity search already computed.
        plain_dense = index.dense_rank(query)
        plain_ranking = rrf_fuse([index.bm25_rank(query), plain_dense])
        plain_cosine = plain_dense[0][1]

        result = entity_search.search(query)
        entity_cosine = result.dense_ranking[0][1]

        def measure(ranking: Ranking, cosine: float, gold: list[str] = gold) -> dict[str, Any]:
            return {
                "top5": [chunk_id for chunk_id, _ in ranking[:5]],
                "top_cosine": round(float(cosine), 4),
                **{f"hit@{k}": hit_at_k(ranking, gold, k) for k in KS},
            }

        records.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": query,
                "answerable": q["answerable"],
                "gold_passages": gold,
                "plain": measure(plain_ranking, plain_cosine),
                "entity": {
                    **measure(result.ranking, entity_cosine),
                    "resolved": result.entities.resolved,
                    "ambiguous": {s: list(c) for s, c in result.entities.ambiguous.items()},
                    "expanded_query": result.expanded_query,
                },
            }
        )

    results = {
        "dataset": "data/eval_questions.jsonl",
        "n_questions": len(questions),
        "class_histogram": class_histogram(questions),
        "embedding_model": MODEL_NAME,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "transformers", "numpy", "huggingface_hub")
        },
        "systems": {
            "plain_hybrid": "RRF(bm25, dense) over the raw query",
            "entity_aware": "RRF(bm25, dense, entity-match) over the entity-expanded query",
        },
        "metrics": {
            "plain": _aggregate(records, "plain"),
            "entity": _aggregate(records, "entity"),
        },
        "no_answer_refusal": {
            "plain": _refusal(records, "plain"),
            "entity": _refusal(records, "entity"),
        },
        "questions": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print_summary(results)
    return results


def print_summary(results: dict[str, Any]) -> None:
    hist = results["class_histogram"]
    classes = ", ".join(f"{t} {n}" for t, n in hist.items())
    print(f"\n{results['n_questions']} questions: {classes}")
    print(
        f"\n{'type':<14}{'n':>3}  "
        + "  ".join(f"{'plain@' + str(k):>8}" for k in KS)
        + "  "
        + "  ".join(f"{'ent@' + str(k):>8}" for k in KS)
    )
    plain, entity = results["metrics"]["plain"], results["metrics"]["entity"]
    rows = [
        *((t, plain["per_type"][t], entity["per_type"][t]) for t in plain["per_type"]),
        ("overall", plain["overall"], entity["overall"]),
    ]
    for name, p, e in rows:
        print(
            f"{name:<14}{p['n']:>3}  "
            + "  ".join(f"{p[f'hit@{k}']:>8.2f}" for k in KS)
            + "  "
            + "  ".join(f"{e[f'hit@{k}']:>8.2f}" for k in KS)
        )
    for system in ("plain", "entity"):
        ref = results["no_answer_refusal"][system]
        print(
            f"{system} refusal: {ref['n_refused']}/{ref['n_no_answer']} "
            f"no-answer questions below floor {ref['floor']}"
        )


if __name__ == "__main__":
    run_eval()
