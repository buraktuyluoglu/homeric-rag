"""Command-line entry point: homeric-rag {fetch,build,annotate,search,eval}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="homeric-rag",
        description="Entity-aware hybrid retrieval over the Iliad and Odyssey",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the Gutenberg texts and strip boilerplate")
    sub.add_parser("build", help="chunk the corpus into data/chunks.jsonl")
    sub.add_parser("annotate", help="entity-annotate chunks into data/chunk_entities.jsonl")
    sub.add_parser("embed", help="embed chunks into data/embeddings.npz (bge-small ONNX)")
    search = sub.add_parser("search", help="query the corpus")
    search.add_argument("query", help="query text")
    search.add_argument(
        "--mode",
        choices=["entity", "hybrid", "bm25", "dense"],
        default="entity",
        help="entity = entity-aware fusion (default); hybrid = plain bm25+dense RRF",
    )
    search.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    sub.add_parser("eval", help="run the retrieval evaluation")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        from homeric_rag.fetch import fetch_corpus

        fetch_corpus(DATA_DIR)
        return 0
    if args.command == "build":
        from homeric_rag.chunking import build_chunks

        build_chunks(DATA_DIR)
        return 0
    if args.command == "annotate":
        from homeric_rag.entities import annotate_chunks

        annotate_chunks(DATA_DIR)
        return 0
    if args.command == "embed":
        from homeric_rag.embed import main as embed_main

        embed_main(DATA_DIR)
        return 0
    if args.command == "search":
        return run_search(args.query, args.mode, args.k)
    if args.command == "eval":
        from homeric_rag.eval import run_eval

        run_eval()
        return 0

    print(f"unknown command {args.command!r}", file=sys.stderr)
    return 2


def run_search(query: str, mode: str, k: int) -> int:
    from homeric_rag.search import SearchIndex

    index = SearchIndex(DATA_DIR)

    if mode == "entity":
        from homeric_rag.entity_search import EntitySearch

        result = EntitySearch(index).search(query)
        ents = result.entities
        if ents.is_empty:
            print("entities: none recognized — plain hybrid ranking")
        else:
            for surface, char_id in ents.resolved.items():
                print(f"entity: {surface!r} -> {char_id}")
            for surface, candidates in ents.collective.items():
                print(f"entity: {surface!r} -> {', '.join(candidates)} (collective)")
            for surface, candidates in ents.ambiguous.items():
                print(
                    f"entity: {surface!r} AMBIGUOUS -> {' or '.join(candidates)}; "
                    "results cover all candidates"
                )
            if result.expanded_query != result.query:
                print(f"expanded query: {result.expanded_query}")
        ranking = result.ranking
    else:
        rank_fn = {
            "hybrid": index.hybrid_rank,
            "bm25": index.bm25_rank,
            "dense": index.dense_rank,
        }[mode]
        ranking = rank_fn(query)

    for position, (chunk_id, score) in enumerate(ranking[:k], start=1):
        chunk = index.chunks[chunk_id]
        ref = f"{chunk['work'].capitalize()}, Book {chunk['book']}"
        snippet = " ".join(chunk["text"].split())[:160]
        print(f"{position}. [{score:.4f}] {chunk_id} ({ref})")
        print(f"   {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
