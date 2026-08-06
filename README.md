# homeric-rag

Entity-aware hybrid retrieval over Homer's Iliad and Odyssey. The project
applies author-disambiguation techniques (alias and epithet resolution for
characters such as Ulysses/Odysseus or "the son of Peleus"/Achilles) to
retrieval over a classical corpus.

Status: corpus and chunking layer. Entity layer, hybrid retrieval (BM25 +
embeddings + RRF), and a measured retrieval evaluation follow; generation is a
later phase.

## Corpus

Both texts are the public-domain prose translations by Samuel Butler, from
Project Gutenberg:

- The Iliad, rendered into English prose by Samuel Butler — Project Gutenberg
  ebook #2199
- The Odyssey, rendered into English prose by Samuel Butler — Project
  Gutenberg ebook #1727

The raw downloads are committed under `data/raw/`, the boilerplate-stripped
texts under `data/corpus/`, and the chunked corpus at `data/chunks.jsonl`
(719 chunks, 48 books, ~400-600 words per chunk with one-sentence overlap),
so the build reproduces offline. Butler's prefaces, the Gutenberg
header/footer, and the Odyssey footnotes section are excluded from chunks.

## Usage

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

homeric-rag fetch    # re-download from gutenberg.org (2 requests, polite UA)
homeric-rag build    # rebuild data/chunks.jsonl (deterministic)
homeric-rag search   # not implemented yet
homeric-rag eval     # not implemented yet

pytest
```

## Limitations

- Butler's translations are Victorian prose and use Roman names (Jove,
  Ulysses, Minerva); the entity layer must treat Roman/Greek name pairs as
  aliases rather than assuming modern Greek-name conventions.
- Sentence splitting is a deterministic regex heuristic, not a linguistic
  segmenter; occasional over- or under-splits shift chunk boundaries.
- Chunk sizes target 400-600 words but a book's final chunk may be as short
  as 150 words.

## License

MIT (code). The corpus texts are in the public domain in the United States.
