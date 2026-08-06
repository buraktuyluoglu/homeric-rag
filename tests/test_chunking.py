import hashlib
import json
from pathlib import Path

from homeric_rag.chunking import (
    MIN_TAIL_WORDS,
    build_chunks,
    chunk_book,
    roman_to_int,
    split_books,
    split_sentences,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def load_chunks():
    lines = (DATA / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_roman_numerals():
    assert [roman_to_int(n) for n in ("I", "IV", "IX", "XIV", "XIX", "XXIV")] == [
        1,
        4,
        9,
        14,
        19,
        24,
    ]


def test_48_books_present():
    chunks = load_chunks()
    books = {(c["work"], c["book"]) for c in chunks}
    assert len(books) == 48
    for work in ("iliad", "odyssey"):
        assert {b for w, b in books if w == work} == set(range(1, 25))


def test_chunk_size_bounds():
    for chunk in load_chunks():
        words = len(chunk["text"].split())
        assert MIN_TAIL_WORDS <= words <= 700, chunk["id"]


def test_no_gutenberg_strings_in_chunks():
    for chunk in load_chunks():
        assert "project gutenberg" not in chunk["text"].lower()
        assert "gutenberg.org" not in chunk["text"].lower()


def test_one_sentence_overlap():
    sentences = [f"Sentence number {i} has exactly six words." for i in range(200)]
    chunks = chunk_book(sentences)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.rsplit("Sentence", 1)[-1]
        assert nxt.startswith("Sentence" + last_sentence + " ")


def test_split_books_drops_front_matter_and_footnotes():
    text = (DATA / "corpus" / "odyssey.txt").read_text(encoding="utf-8")
    books = split_books(text)
    assert len(books) == 24
    assert "FOOTNOTES" not in books[24]
    # Butler's preface never enters any book.
    assert all("PREFACE" not in b for b in books.values())


def test_sentence_split_is_reasonable():
    sents = split_sentences('He spoke. "Go now," she said. Then they    left.')
    assert sents == ["He spoke.", '"Go now," she said.', "Then they left."]


def test_deterministic_rebuild(tmp_path):
    committed = hashlib.sha256((DATA / "chunks.jsonl").read_bytes()).hexdigest()

    work_dir = tmp_path / "data"
    (work_dir / "corpus").mkdir(parents=True)
    for work in ("iliad", "odyssey"):
        src = (DATA / "corpus" / f"{work}.txt").read_text(encoding="utf-8")
        (work_dir / "corpus" / f"{work}.txt").write_text(src, encoding="utf-8", newline="\n")

    rebuilt = build_chunks(work_dir)
    assert hashlib.sha256(rebuilt.read_bytes()).hexdigest() == committed
