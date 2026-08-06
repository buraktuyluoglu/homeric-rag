"""Download the corpus texts from Project Gutenberg and strip boilerplate.

Sources (verified on gutenberg.org, 2026-08-06):
  - The Iliad, trans. Samuel Butler (prose) — Project Gutenberg ebook #2199
  - The Odyssey, trans. Samuel Butler (prose) — Project Gutenberg ebook #1727

Exactly one HTTP request per work. Raw downloads are saved verbatim to
``data/raw/`` and the boilerplate-stripped texts to ``data/corpus/`` so the
rest of the pipeline never needs the network.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

USER_AGENT = "homeric-rag/0.1 (corpus fetch; contact: tuyluogluburakemre@gmail.com)"

WORKS = {
    "iliad": {
        "ebook_id": 2199,
        "url": "https://www.gutenberg.org/cache/epub/2199/pg2199.txt",
        "title": "The Iliad",
        "translator": "Samuel Butler",
    },
    "odyssey": {
        "ebook_id": 1727,
        "url": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
        "title": "The Odyssey",
        "translator": "Samuel Butler",
    },
}

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Return only the content between the standard PG start/end markers.

    Deterministic: raises ``ValueError`` if either marker is missing or
    ambiguous instead of guessing.
    """
    lines = text.splitlines()
    start_idx = [i for i, line in enumerate(lines) if line.startswith(START_MARKER)]
    end_idx = [i for i, line in enumerate(lines) if line.startswith(END_MARKER)]
    if len(start_idx) != 1 or len(end_idx) != 1:
        raise ValueError(
            f"expected exactly one start and one end marker, "
            f"found {len(start_idx)} start / {len(end_idx)} end"
        )
    body = lines[start_idx[0] + 1 : end_idx[0]]
    return "\n".join(body).strip() + "\n"


def _download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def fetch_corpus(data_dir: Path) -> dict[str, Path]:
    """Download both works (skipping any already on disk) and write cleaned texts.

    Returns a mapping of work name to cleaned-text path.
    """
    raw_dir = data_dir / "raw"
    corpus_dir = data_dir / "corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    cleaned_paths: dict[str, Path] = {}
    for i, (work, meta) in enumerate(WORKS.items()):
        raw_path = raw_dir / f"pg{meta['ebook_id']}.txt"
        if raw_path.exists():
            print(f"{work}: raw file already present, skipping download")
            raw_text = raw_path.read_text(encoding="utf-8")
        else:
            if i > 0:
                time.sleep(2)  # be polite between the two requests
            print(f"{work}: downloading {meta['url']}")
            raw_text = _download(meta["url"])
            raw_path.write_text(raw_text, encoding="utf-8", newline="\n")

        cleaned = strip_gutenberg_boilerplate(raw_text)
        cleaned_path = corpus_dir / f"{work}.txt"
        cleaned_path.write_text(cleaned, encoding="utf-8", newline="\n")
        cleaned_paths[work] = cleaned_path
        print(f"{work}: {len(cleaned.split())} words -> {cleaned_path}")
    return cleaned_paths
