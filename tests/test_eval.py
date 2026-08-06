"""Eval dataset integrity and metric behavior (offline — no embedding)."""

import json

from homeric_rag.eval import (
    QUESTIONS_PATH,
    TYPE_ORDER,
    class_histogram,
    hit_at_k,
    load_questions,
)
from homeric_rag.search import DATA_DIR


def chunk_ids() -> set[str]:
    lines = (DATA_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["id"] for line in lines}


def test_dataset_shape():
    questions = load_questions(QUESTIONS_PATH)
    assert 40 <= len(questions) <= 60
    for q in questions:
        assert set(q) == {"id", "question", "type", "gold_char_ids", "gold_passages", "answerable"}
        assert q["type"] in TYPE_ORDER


def test_gold_passages_exist_in_corpus():
    ids = chunk_ids()
    for q in load_questions(QUESTIONS_PATH):
        for gold in q["gold_passages"]:
            assert gold in ids, f"{q['id']}: gold chunk {gold} not in corpus"


def test_answerable_consistency():
    for q in load_questions(QUESTIONS_PATH):
        if q["answerable"]:
            assert q["gold_passages"], f"{q['id']} answerable but has no gold passages"
        else:
            assert not q["gold_passages"], f"{q['id']} unanswerable but cites gold passages"
            assert q["type"] == "no-answer"


def test_at_least_five_no_answer():
    hist = class_histogram(load_questions(QUESTIONS_PATH))
    assert hist.get("no-answer", 0) >= 5
    # every class is represented
    assert set(hist) == set(TYPE_ORDER)


def test_gold_char_ids_are_on_roster():
    roster = {
        c["id"]
        for c in json.loads((DATA_DIR / "characters.json").read_text(encoding="utf-8"))[
            "characters"
        ]
    }
    for q in load_questions(QUESTIONS_PATH):
        for char_id in q["gold_char_ids"]:
            assert char_id in roster, f"{q['id']}: unknown character {char_id}"


def test_hit_at_k():
    ranking = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    assert hit_at_k(ranking, ["a"], 1)
    assert not hit_at_k(ranking, ["b"], 1)
    assert hit_at_k(ranking, ["b"], 3)
    assert hit_at_k(ranking, ["z", "c"], 3)
    assert not hit_at_k(ranking, ["z"], 3)
    assert not hit_at_k(ranking, [], 3)
