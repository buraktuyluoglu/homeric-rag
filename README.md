# homeric-rag

Entity-aware hybrid retrieval over Homer's Iliad and Odyssey. The project
applies author-disambiguation techniques (alias and epithet resolution for
characters such as Ulysses/Odysseus or "the son of Peleus"/Achilles) to
retrieval over a classical corpus.

Status: corpus, chunking, and entity layer. Hybrid retrieval (BM25 +
embeddings + RRF) and a measured retrieval evaluation follow; generation is a
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

## Entity layer

`data/characters.json` holds a hand-checkable roster of 90 Homeric characters
(plus 2 plural collectives), each with the aliases, epithets, and patronymics
Butler actually uses — every surface form is verified to occur in
`data/chunks.jsonl`, and a test enforces this. Butler writes Roman names, so
the canonical forms are his (Jove, Ulysses, Minerva, Diomed) with the Greek
name recorded alongside.

`src/homeric_rag/entities.py` resolves mentions by longest-match lookup with
explicit ambiguity, the same posture as author disambiguation: a surface form
that several characters share is returned as AMBIGUOUS with all candidates
rather than guessed. `homeric-rag annotate` writes per-chunk annotations to
`data/chunk_entities.jsonl`.

Numbers from the committed annotation run (719 chunks):

- 7,738 entity attributions (a collective mention counts once per member);
  711 of 719 chunks carry at least one resolved entity.
- 242 mentions left unattributed as ambiguous, by form:
  - `Ajax` — 142 (Telamonian vs the son of Oileus; resolves only when a
    disambiguating patronymic is adjacent, e.g. "Ajax, the fleet son of
    Oileus")
  - `son of Atreus` — 71 (Agamemnon vs Menelaus; the plural "sons of Atreus"
    is a collective reference to both, not an ambiguity)
  - `son of Priam` — 14 (among Priam's sons on the roster)
  - `old man of the sea` — 9 (Proteus in the Odyssey, Nereus in the Iliad)
  - `son of Telamon` — 6 (Ajax vs his half-brother Teucer)

## Usage

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

homeric-rag fetch    # re-download from gutenberg.org (2 requests, polite UA)
homeric-rag build    # rebuild data/chunks.jsonl (deterministic)
homeric-rag annotate # rebuild data/chunk_entities.jsonl (deterministic)
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
- Butler's prose drops most Homeric epithets ("Pelides" and "swift-footed"
  never occur), so the entity layer lists only the forms he uses.
- Adjacency disambiguation is a small connector-word heuristic, not
  coreference resolution: "Most noble son of Atreus, king of men, Agamemnon"
  stays ambiguous because the apposition is not adjacent.
- Homonyms across works are noted per character but not modeled as separate
  entities: the lone Iliad occurrences of Aeolus, Polyphemus, and Mentor are
  different figures, one "son of Oileus" is Medon the bastard son, and the
  Iliad's minor fighters named Orestes are counted with Agamemnon's son.
- "Hades" mixes the god and the place ("the house of Hades"), as Butler does.

## License

MIT (code). The corpus texts are in the public domain in the United States.
