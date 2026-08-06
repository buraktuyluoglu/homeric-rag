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
    search = sub.add_parser("search", help="query the hybrid index")
    search.add_argument("query", nargs="?", help="query text")
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

    print(f"'{args.command}' is not implemented yet (retrieval phase)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
