import json
import re
from pathlib import Path

import pytest

from homeric_rag.entities import (
    AMBIGUOUS,
    COLLECTIVE,
    RESOLVED,
    chunk_entities,
    default_index,
    resolve_mentions,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def load_chunks():
    lines = (DATA / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def only(mentions):
    assert len(mentions) == 1, mentions
    return mentions[0]


# --- the two Ajaxes -------------------------------------------------------


def test_bare_ajax_is_ambiguous():
    m = only(resolve_mentions("Ajax threw his spear."))
    assert m.kind == AMBIGUOUS
    assert m.char_id is None
    assert set(m.candidates) == {"ajax_oileus", "ajax_telamon"}


def test_ajax_with_telamon_patronymic_resolves_as_one_mention():
    m = only(resolve_mentions("Forthwith uprose great Ajax son of Telamon."))
    assert m.kind == RESOLVED
    assert m.char_id == "ajax_telamon"
    assert m.surface == "Ajax son of Telamon"


def test_ajax_comma_variant_resolves():
    m = only(resolve_mentions("Ajax, son of Telamon, stood firm."))
    assert (m.kind, m.char_id) == (RESOLVED, "ajax_telamon")


def test_adjacent_apposition_resolves_oilean_ajax():
    # No explicit "Ajax, the fleet son of Oileus" alias exists; adjacency
    # between the ambiguous "Ajax" and the resolved patronymic must do it.
    mentions = resolve_mentions("Ajax, the fleet son of Oileus, commanded the Locrians.")
    assert [(m.kind, m.char_id) for m in mentions] == [
        (RESOLVED, "ajax_oileus"),
        (RESOLVED, "ajax_oileus"),
    ]


def test_adjacent_apposition_resolves_via_ambiguous_intersection():
    # "son of Telamon" alone is ambiguous (Ajax or Teucer); intersecting with
    # the ambiguous "Ajax" leaves exactly ajax_telamon.
    mentions = resolve_mentions("Ajax the son of Telamon rose also.")
    assert [(m.kind, m.char_id) for m in mentions] == [
        (RESOLVED, "ajax_telamon"),
        (RESOLVED, "ajax_telamon"),
    ]


def test_conjunction_blocks_apposition():
    # "Ajax and Teucer son of Telamon": Teucer's patronymic must not resolve
    # the bare Ajax.
    mentions = resolve_mentions("Ajax and Teucer son of Telamon shot on.")
    assert mentions[0].kind == AMBIGUOUS
    assert mentions[0].surface == "Ajax"


def test_son_of_telamon_alone_is_ambiguous_with_teucer():
    m = only(resolve_mentions("The son of Telamon now struck him under the ear."))
    assert m.kind == AMBIGUOUS
    assert set(m.candidates) == {"ajax_telamon", "teucer"}


def test_two_ajaxes_collective():
    m = only(resolve_mentions("Then came the two Ajaxes, clothed in valour."))
    assert m.kind == COLLECTIVE
    assert m.surface == "the two Ajaxes"
    assert set(m.candidates) == {"ajax_oileus", "ajax_telamon"}


# --- Atreides -------------------------------------------------------------


def test_son_of_atreus_singular_is_ambiguous():
    m = only(resolve_mentions("Thus spoke the son of Atreus."))
    assert m.kind == AMBIGUOUS
    assert set(m.candidates) == {"agamemnon", "menelaus"}


def test_sons_of_atreus_plural_is_collective():
    m = only(resolve_mentions("He besought the sons of Atreus."))
    assert m.kind == COLLECTIVE
    assert set(m.candidates) == {"agamemnon", "menelaus"}


def test_name_before_patronymic_resolves_atreides():
    mentions = resolve_mentions("Agamemnon son of Atreus rose in anger.")
    assert [(m.kind, m.char_id) for m in mentions] == [
        (RESOLVED, "agamemnon"),
        (RESOLVED, "agamemnon"),
    ]


def test_menelaus_apposition_resolves_the_other_atreides():
    mentions = resolve_mentions("Menelaus, son of Atreus, answered him.")
    assert [(m.kind, m.char_id) for m in mentions] == [
        (RESOLVED, "menelaus"),
        (RESOLVED, "menelaus"),
    ]


# --- epithets and patronymics --------------------------------------------


@pytest.mark.parametrize(
    ("text", "char_id"),
    [
        ("Pallas Minerva put valour into his heart.", "athena"),
        ("Phoebus came down in fury.", "apollo"),
        ("Phoebus Apollo heard his prayer.", "apollo"),
        ("But the Cyclops had filled his paunch.", "polyphemus"),
        ("Mercury, slayer of Argus, was sent as guide.", "hermes"),
        ("Neptune, lord of the earthquake, spoke first.", "poseidon"),
        ("So spoke the son of Peleus.", "achilles"),
        ("Then answered the son of Laertes.", "odysseus"),
        ("Brave Diomed the son of Tydeus mounted his car.", "diomedes"),
        ("The son of Saturn bowed his dark brows.", "zeus"),
        ("Ulysses wept as he heard the song.", "odysseus"),
    ],
)
def test_resolution(text, char_id):
    resolved = [m for m in resolve_mentions(text) if m.kind == RESOLVED]
    assert char_id in {m.char_id for m in resolved}


def test_bare_saturn_is_cronus_not_zeus():
    m = only(resolve_mentions("They feared the vengeance of Saturn."))
    assert (m.kind, m.char_id) == (RESOLVED, "cronus")


def test_old_man_of_the_sea_is_ambiguous():
    m = only(resolve_mentions("At noon the old man of the sea came up from under the waves."))
    assert m.kind == AMBIGUOUS
    assert set(m.candidates) == {"nereus", "proteus"}


def test_son_of_priam_alone_is_ambiguous_and_apposition_resolves():
    m = only(resolve_mentions("The spear struck the shield of the son of Priam."))
    assert m.kind == AMBIGUOUS
    assert "hector" in m.candidates and "paris" in m.candidates

    mentions = resolve_mentions("Hector son of Priam raged with intolerable fury.")
    assert [(m.kind, m.char_id) for m in mentions] == [
        (RESOLVED, "hector"),
        (RESOLVED, "hector"),
    ]


# --- mention mechanics ----------------------------------------------------


def test_spans_and_surfaces_are_exact():
    text = "Then Ulysses spoke to Minerva."
    mentions = resolve_mentions(text)
    assert [text[m.span[0] : m.span[1]] for m in mentions] == [m.surface for m in mentions]
    assert [m.surface for m in mentions] == ["Ulysses", "Minerva"]


def test_no_match_inside_longer_words():
    # "Helen" must not fire inside "Helenus".
    m = only(resolve_mentions("Helenus divined the counsel of the gods."))
    assert (m.kind, m.char_id) == (RESOLVED, "helenus")


def test_ambiguous_forms_are_exactly_the_documented_set():
    assert set(default_index().ambiguous_forms) == {
        "Ajax",
        "son of Atreus",
        "son of Telamon",
        "son of Priam",
        "old man of the sea",
    }


def test_roster_size_and_schema():
    index = default_index()
    assert 60 <= len(index.characters) <= 90
    for char in index.characters.values():
        assert char["id"] and char["canonical_name"]
        assert isinstance(char["aliases"], list) and char["aliases"]
        assert set(char["work_presence"]) <= {"iliad", "odyssey"}


# --- corpus validation ----------------------------------------------------


def test_every_surface_form_occurs_in_corpus():
    """Every alias/epithet/patronymic/collective form must occur at least
    once in data/chunks.jsonl — no invented epithets."""
    text = " ".join(c["text"] for c in load_chunks())
    index = default_index()
    forms = set()
    for char in index.characters.values():
        forms.update(char["aliases"] + char["epithets"] + char["patronymics"])
    for coll in index.collectives.values():
        forms.update(coll["surface_forms"])

    missing = [
        form
        for form in sorted(forms)
        if not re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", text)
    ]
    assert missing == [], f"surface forms absent from corpus: {missing}"


def test_chunk_entities_on_opening_chunk():
    chunk = next(c for c in load_chunks() if c["id"] == "iliad-01-000")
    record = chunk_entities(chunk)
    assert record["id"] == "iliad-01-000"
    for expected in ("achilles", "agamemnon", "zeus", "thetis", "chryses"):
        assert expected in record["entities"], record["entities"]
    # "the son of Atreus, king of men" stays honestly unresolved.
    assert record["ambiguous"].get("son of Atreus", 0) >= 1
    # collective: "the two sons of Atreus" appears in the opening scene
    assert record["entities"]["achilles"] >= 3


def test_chunk_entities_file_is_current():
    """data/chunk_entities.jsonl matches what chunk_entities produces today."""
    path = DATA / "chunk_entities.jsonl"
    assert path.exists(), "run: homeric-rag annotate"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chunks = load_chunks()
    assert len(records) == len(chunks)
    by_id = {c["id"]: c for c in chunks}
    # Every record: the full regex pass over 719 chunks takes seconds, and
    # a stale annotation anywhere silently skews the entity ranking.
    for record in records:
        assert record == chunk_entities(by_id[record["id"]])
