from pathlib import Path

import pytest

from homeric_rag.fetch import WORKS, strip_gutenberg_boilerplate

DATA = Path(__file__).resolve().parents[1] / "data"

SAMPLE = """The Project Gutenberg eBook of Something
License blah blah.

*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

Actual body text here.
More body.

*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***

End license boilerplate.
"""


def test_strip_keeps_only_body():
    body = strip_gutenberg_boilerplate(SAMPLE)
    assert body == "Actual body text here.\nMore body.\n"


def test_strip_rejects_missing_markers():
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate("no markers at all")


def test_strip_rejects_duplicate_markers():
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate(SAMPLE + SAMPLE)


def test_committed_raw_files_exist():
    for meta in WORKS.values():
        assert (DATA / "raw" / f"pg{meta['ebook_id']}.txt").exists()


def test_cleaned_corpus_has_no_gutenberg_boilerplate():
    for work in WORKS:
        text = (DATA / "corpus" / f"{work}.txt").read_text(encoding="utf-8")
        assert "PROJECT GUTENBERG" not in text.upper()
        assert "gutenberg.org" not in text.lower()
