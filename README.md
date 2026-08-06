# homeric-rag

Entity-aware hybrid retrieval over Homer's Iliad and Odyssey. The project
applies author-disambiguation techniques (alias and epithet resolution for
characters such as Ulysses/Odysseus or "the son of Peleus"/Achilles) to
retrieval over a classical corpus, and measures whether the entity layer
actually helps.

## Results

51 evaluation questions (12 alias, 10 homonym, 6 cross-book, 17
plain-factual, 6 no-answer), every gold passage located by reading the
corpus chunks and cited by chunk id in `data/eval_questions.jsonl`. All
numbers below are from the committed run in `out/results.json`
(`homeric-rag eval` regenerates it). hit@k = fraction of questions with at
least one gold chunk in the top k. Plain = RRF(BM25, dense) on the raw
query; entity-aware = RRF(BM25, dense, entity-match) on the
entity-expanded query.

| Question type | n | plain hit@1 | entity hit@1 | plain hit@3 | entity hit@3 | plain hit@5 | entity hit@5 |
|---|---|---|---|---|---|---|---|
| alias / epithet | 12 | 0.08 | **0.25** | 0.17 | **0.67** | 0.42 | **0.83** |
| homonym | 10 | **0.20** | 0.00 | **0.40** | 0.10 | **0.80** | 0.20 |
| cross-book | 6 | 0.33 | 0.33 | 0.33 | **0.50** | 0.67 | 0.67 |
| plain-factual | 17 | **0.53** | 0.24 | **0.65** | 0.59 | **0.71** | 0.59 |
| overall | 45 | **0.31** | 0.20 | 0.42 | **0.49** | **0.64** | 0.58 |

The split is sharp, and half of it is a negative result:

- **Alias questions are the win the entity layer exists for.** A question
  phrased with "Odysseus" or "the son of Peleus" misses Butler's prose,
  which says "Ulysses" and "Achilles"; expanding the query with the
  resolved character's surface forms doubles hit@5 (0.42 → 0.83).
- **Homonym questions get worse, not better.** Three reasons, visible in
  the per-question records: (1) the entity-match ranking is
  frequency-based, so chunks where a character is mentioned *most* (dense
  battle scenes) outrank the specific scene the question asks about;
  (2) the corpus-side annotator honestly leaves bare "Ajax" unattributed
  as ambiguous, so exactly the passages a homonym question needs (the
  Odyssey's "Ajax was wrecked...", which never says which Ajax) do not
  feed the entity signal; (3) expansion appends both candidates' full
  alias lists to an ambiguous query, diluting the discriminative terms
  ("lot", "sea-pike", "wrestle") that plain BM25 was winning with.
  The same dilution costs entity-aware the hit@1 lead on plain-factual
  questions. Reported as measured; fixing it (mention-level rather than
  frequency-level entity scoring, expansion only for out-of-vocabulary
  names) is future work, not something this table pretends is done.
- **No-answer refusal barely works with a score floor.** Refusal is
  measured as top-1 dense cosine below a floor set at the minimum top-1
  cosine observed on answerable questions in the same run (0.5599 —
  documented in `out/results.json`; false-refusal on answerable questions
  is therefore zero by construction). Only 1 of 6 no-answer questions
  falls below it, for both systems: bge-small cosines for in-domain-sounding
  but unanswerable questions ("How was the infant Achilles made
  invulnerable?", 0.71) sit well inside the answerable range (0.56–0.80).
  A raw similarity floor is not a usable no-answer detector here.

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

## How it works

The corpus is chunked and indexed three ways: Okapi BM25 (k1=1.2, b=0.75),
dense cosine over bge-small-en-v1.5 embeddings (384-dim, ONNX Runtime on
CPU, CLS pooling, query instruction prefix), and an entity-match ranking
built from `data/chunk_entities.jsonl`. A query is first run through the
character roster (`data/characters.json`, 90 characters with the aliases,
epithets, and patronymics Butler actually uses): resolved mentions expand
the query with the character's surface forms, and the three rankings are
fused with Reciprocal Rank Fusion (k=60). Ambiguity is handled the way an
author-disambiguation system handles it — a surface several characters
share ("Ajax", "son of Atreus") is never silently guessed: in a query it
contributes all candidates and is flagged; in a corpus chunk it is left
unattributed unless a disambiguating patronymic is adjacent. 242 corpus
mentions stay ambiguous on purpose (142 of them bare "Ajax").

## Reproduce

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
homeric-rag eval   # writes out/results.json (first run downloads the embedding model)
pytest
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
homeric-rag eval
pytest
```

Everything the eval needs (chunks, annotations, embeddings, questions) is
committed; `homeric-rag fetch / build / annotate / embed` rebuild the
artifacts from the raw Gutenberg texts, and `homeric-rag search "query"`
runs ad-hoc queries (`--mode entity|hybrid|bm25|dense`).

## Phase B roadmap

A LangGraph agent over the retrieval and entity APIs with local 7B
generation and recorded traces — retrieval-only evaluation first was
deliberate, so generation quality never obscures retrieval quality.

## Limitations

- The evaluation is retrieval-only: hit@k against gold passages, no answer
  generation and no answer grading yet.
- Single embedding model (bge-small-en-v1.5); the dense-retrieval and
  refusal findings may not transfer to larger or differently trained
  encoders.
- The entity-aware variant loses on homonym and plain-factual questions as
  configured (see Results); the entity signal is frequency-based and the
  query expansion is indiscriminate.
- The no-answer floor is calibrated on the answerable questions of the same
  run — it is a measurement of separability, not a deployable detector.
- 51 questions written by one person over one translation; surface forms
  are Butler-specific (his prose drops most Homeric epithets — "Pelides"
  and "swift-footed" never occur), so alias results depend on his Roman
  naming (Jove, Ulysses, Minerva, Diomed).
- Sentence splitting is a deterministic regex heuristic, not a linguistic
  segmenter; chunk sizes target 400-600 words but a book's final chunk may
  be as short as 150 words.
- Corpus-side adjacency disambiguation is a small connector-word heuristic,
  not coreference resolution: "Most noble son of Atreus, king of men,
  Agamemnon" stays ambiguous because the apposition is not adjacent.
- Homonyms across works are noted per character but not modeled as separate
  entities: the lone Iliad occurrences of Aeolus, Polyphemus, and Mentor
  are different figures from their Odyssey namesakes, and "Hades" mixes
  the god and the place, as Butler does.

## License

MIT (code). The corpus texts are in the public domain in the United States.
