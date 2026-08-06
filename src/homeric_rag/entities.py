"""Entity layer: alias, epithet, and patronymic resolution over the corpus.

The roster lives in ``data/characters.json``. Resolution is longest-match
surface lookup with explicit ambiguity: a surface form listed under more than
one character (``Ajax``, ``son of Atreus``, ``son of Telamon``, ``son of
Priam``, ``old man of the sea``) resolves to an AMBIGUOUS mention carrying all
candidates unless a disambiguating mention is adjacent ("Ajax, noble son of
Telamon" resolves; bare "Ajax" does not). Plural collectives ("sons of
Atreus", "the two Ajaxes") resolve to all members rather than being forced to
a single id.

Everything is deterministic and offline: same roster, same text, same
mentions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Mention kinds.
RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
COLLECTIVE = "collective"

# Words allowed in the gap between a name and a disambiguating apposition:
# "Ajax the son of Telamon", "Ajax, noble son of Telamon",
# "Ajax, the fleet son of Oileus". Conjunctions such as "and" are deliberately
# absent — "Ajax and Teucer, son of Telamon" must not resolve Ajax.
CONNECTOR_WORDS = frozenset(
    {"the", "noble", "great", "mighty", "brave", "fleet", "crafty", "royal", "king"}
)
MAX_GAP_CHARS = 24
_GAP_TOKEN = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class Mention:
    """One entity mention in a text.

    ``kind`` is RESOLVED (``char_id`` set), AMBIGUOUS (``candidates`` holds
    every character the surface could denote), or COLLECTIVE (``candidates``
    holds every member the plural denotes).
    """

    surface: str
    span: tuple[int, int]
    kind: str
    char_id: str | None = None
    candidates: tuple[str, ...] = ()


class EntityIndex:
    """Surface-form index over the character roster."""

    def __init__(self, roster: dict) -> None:
        self.characters = {c["id"]: c for c in roster["characters"]}
        self.collectives = {c["id"]: c for c in roster["collectives"]}

        surface_to_ids: dict[str, set[str]] = {}
        for char in roster["characters"]:
            for form in char["aliases"] + char["epithets"] + char["patronymics"]:
                surface_to_ids.setdefault(form, set()).add(char["id"])
        self._entity_surfaces = {form: tuple(sorted(ids)) for form, ids in surface_to_ids.items()}
        self._collective_surfaces = {
            form: tuple(coll["members"])
            for coll in roster["collectives"]
            for form in coll["surface_forms"]
        }

        all_forms = sorted(
            set(self._entity_surfaces) | set(self._collective_surfaces),
            key=len,
            reverse=True,
        )
        # Longest alternative first, so "Ajax son of Telamon" wins over "Ajax"
        # and "the two Ajaxes" over "Ajaxes". Case-sensitive on purpose:
        # capitalized names must not match common nouns.
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(f) for f in all_forms) + r")(?!\w)"
        )

    @property
    def ambiguous_forms(self) -> dict[str, tuple[str, ...]]:
        """Surface forms that denote more than one character."""
        return {f: ids for f, ids in self._entity_surfaces.items() if len(ids) > 1}

    def resolve_mentions(self, text: str) -> list[Mention]:
        mentions = [self._classify(m.group(), m.span()) for m in self._pattern.finditer(text)]
        return self._disambiguate_adjacent(mentions, text)

    def _classify(self, surface: str, span: tuple[int, int]) -> Mention:
        if surface in self._collective_surfaces:
            return Mention(surface, span, COLLECTIVE, candidates=self._collective_surfaces[surface])
        ids = self._entity_surfaces[surface]
        if len(ids) == 1:
            return Mention(surface, span, RESOLVED, char_id=ids[0])
        return Mention(surface, span, AMBIGUOUS, candidates=ids)

    def _disambiguate_adjacent(self, mentions: list[Mention], text: str) -> list[Mention]:
        """Resolve ambiguous mentions against adjacent appositions.

        For each neighboring pair joined only by connector words, intersect
        the candidate sets; a singleton intersection resolves both sides
        ("Ajax" x "son of Telamon" -> ajax_telamon). Repeats until stable so a
        resolution can propagate to the other neighbor.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(mentions) - 1):
                left, right = mentions[i], mentions[i + 1]
                if AMBIGUOUS not in (left.kind, right.kind):
                    continue
                if COLLECTIVE in (left.kind, right.kind):
                    continue
                if not self._gap_is_connector(text[left.span[1] : right.span[0]]):
                    continue
                shared = set(self._candidate_set(left)) & set(self._candidate_set(right))
                if len(shared) != 1:
                    continue
                (char_id,) = shared
                for j in (i, i + 1):
                    if mentions[j].kind == AMBIGUOUS:
                        mentions[j] = Mention(
                            mentions[j].surface, mentions[j].span, RESOLVED, char_id=char_id
                        )
                        changed = True
        return mentions

    @staticmethod
    def _candidate_set(mention: Mention) -> tuple[str, ...]:
        return (mention.char_id,) if mention.kind == RESOLVED else mention.candidates

    @staticmethod
    def _gap_is_connector(gap: str) -> bool:
        if len(gap) > MAX_GAP_CHARS:
            return False
        return all(token in CONNECTOR_WORDS for token in _GAP_TOKEN.split(gap) if token)


_default_index: EntityIndex | None = None


def default_index(data_dir: Path = DATA_DIR) -> EntityIndex:
    global _default_index
    if _default_index is None:
        roster = json.loads((data_dir / "characters.json").read_text(encoding="utf-8"))
        _default_index = EntityIndex(roster)
    return _default_index


def resolve_mentions(text: str) -> list[Mention]:
    """Resolve entity mentions in free text against the default roster."""
    return default_index().resolve_mentions(text)


def chunk_entities(chunk: dict) -> dict:
    """Index-time annotation for one chunk from data/chunks.jsonl.

    Returns ``{"id", "entities": {char_id: count}, "ambiguous": {surface:
    count}}``. Collective mentions count once for every member; ambiguous
    mentions are reported by surface, unattributed — the honest option at
    index time.
    """
    entities: dict[str, int] = {}
    ambiguous: dict[str, int] = {}
    for mention in resolve_mentions(chunk["text"]):
        if mention.kind == RESOLVED:
            entities[mention.char_id] = entities.get(mention.char_id, 0) + 1
        elif mention.kind == COLLECTIVE:
            for member in mention.candidates:
                entities[member] = entities.get(member, 0) + 1
        else:
            ambiguous[mention.surface] = ambiguous.get(mention.surface, 0) + 1
    return {
        "id": chunk["id"],
        "entities": dict(sorted(entities.items())),
        "ambiguous": dict(sorted(ambiguous.items())),
    }


def annotate_chunks(data_dir: Path = DATA_DIR) -> Path:
    """Annotate every chunk and write data/chunk_entities.jsonl."""
    chunks = [
        json.loads(line)
        for line in (data_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records = [chunk_entities(chunk) for chunk in chunks]

    out_path = data_dir / "chunk_entities.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    index = default_index(data_dir)
    resolved = sum(sum(r["entities"].values()) for r in records)
    ambiguous = sum(sum(r["ambiguous"].values()) for r in records)
    seen = {char_id for r in records for char_id in r["entities"]}
    print(f"chunks annotated: {len(records)}")
    print(f"characters in roster: {len(index.characters)} (seen in corpus: {len(seen)})")
    print(f"entity attributions (incl. collective members): {resolved}")
    print(f"ambiguous mentions left unattributed: {ambiguous}")
    for form, ids in sorted(index.ambiguous_forms.items()):
        count = sum(r["ambiguous"].get(form, 0) for r in records)
        print(f"  ambiguous form {form!r} -> {'/'.join(ids)}: {count} unresolved")
    return out_path
