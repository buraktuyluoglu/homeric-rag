"""Split the cleaned corpus into retrieval chunks.

Structure: each work is split into its 24 books (headings like ``BOOK I.`` at
column 0), then each book into chunks of roughly 400-600 words with a
one-sentence overlap between consecutive chunks. Output is
``data/chunks.jsonl`` with one object per line:

    {"id": "iliad-01-000", "work": "iliad", "book": 1, "chunk_index": 0,
     "text": "..."}

Everything here is deterministic: same input text, same chunks, same bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# Chunk sizing (words). A chunk closes once it reaches TARGET_WORDS; a
# trailing remainder smaller than MIN_TAIL_WORDS is merged into the previous
# chunk instead of standing alone.
TARGET_WORDS = 400
MIN_TAIL_WORDS = 150

BOOK_HEADING = re.compile(r"^BOOK ([IVXL]+)\.?\s*$", re.MULTILINE)
# End-of-body markers: the Odyssey text carries a footnotes section after
# book 24 which is not part of the narrative corpus.
BODY_END = re.compile(r"^FOOTNOTES:\s*$", re.MULTILINE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[”’\"']*\s+")

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def roman_to_int(numeral: str) -> int:
    total = 0
    for ch, nxt in zip(numeral, list(numeral[1:]) + [None], strict=True):
        value = _ROMAN[ch]
        if nxt is not None and value < _ROMAN[nxt]:
            total -= value
        else:
            total += value
    return total


def split_books(text: str) -> dict[int, str]:
    """Return {book_number: book_text} for the 24 books of one work.

    Front matter before the first column-0 ``BOOK ...`` heading and any
    footnotes section after the last book are dropped.
    """
    end_match = BODY_END.search(text)
    if end_match:
        text = text[: end_match.start()]

    headings = list(BOOK_HEADING.finditer(text))
    if len(headings) != 24:
        raise ValueError(f"expected 24 book headings, found {len(headings)}")

    books: dict[int, str] = {}
    for i, match in enumerate(headings):
        number = roman_to_int(match.group(1))
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        books[number] = text[start:end].strip()

    if sorted(books) != list(range(1, 25)):
        raise ValueError(f"book numbers are not 1..24: {sorted(books)}")
    return books


def split_sentences(text: str) -> list[str]:
    """Whitespace-normalize a book and split it into sentences."""
    flat = " ".join(text.split())
    return [s.strip() for s in SENTENCE_SPLIT.split(flat) if s.strip()]


def chunk_book(sentences: list[str]) -> list[str]:
    """Group sentences into ~TARGET_WORDS-word chunks with 1-sentence overlap."""
    chunks: list[list[str]] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        current.append(sentence)
        count += len(sentence.split())
        if count >= TARGET_WORDS:
            chunks.append(current)
            current = [sentence]  # one-sentence overlap into the next chunk
            count = len(sentence.split())
    # `current` always holds at least the overlap sentence; it is new content
    # only if more sentences followed the last closed chunk.
    if current and (not chunks or current != [chunks[-1][-1]]):
        tail_words = sum(len(s.split()) for s in current)
        if chunks and tail_words < MIN_TAIL_WORDS:
            # Merge the short remainder into the previous chunk (drop the
            # duplicated overlap sentence first).
            chunks[-1].extend(current[1:])
        else:
            chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def build_chunks(data_dir: Path) -> Path:
    """Chunk both works and write data/chunks.jsonl. Returns the output path."""
    corpus_dir = data_dir / "corpus"
    records = []
    for work in ("iliad", "odyssey"):
        text = (corpus_dir / f"{work}.txt").read_text(encoding="utf-8")
        for book_number, book_text in sorted(split_books(text).items()):
            sentences = split_sentences(book_text)
            for chunk_index, chunk_text in enumerate(chunk_book(sentences)):
                records.append(
                    {
                        "id": f"{work}-{book_number:02d}-{chunk_index:03d}",
                        "work": work,
                        "book": book_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )

    out_path = data_dir / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    word_counts = [len(r["text"].split()) for r in records]
    total_words = sum(word_counts)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"chunks: {len(records)}")
    print(f"total words: {total_words} (~{int(total_words * 1.3)} tokens approx)")
    print(
        f"chunk words: min {min(word_counts)} / "
        f"mean {total_words // len(records)} / max {max(word_counts)}"
    )
    print(f"sha256(chunks.jsonl): {digest}")
    return out_path
