"""Round-4 reply reading: the number tied to the asked verb, answers keyed by the questions' own labels, a retry for
optional answers the analysis uses, and never a default value. The questions follow a repeated public-goods game
with peer punishment (a contribution, then deduction points for participants A, B and C)."""
from __future__ import annotations

import importlib.util
import sys

import pytest

from package_fixture import REPLIES

spec = importlib.util.spec_from_file_location("replies_round4", REPLIES)
replies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = replies
spec.loader.exec_module(replies)
Question, parse, ask, split_reply = replies.Question, replies.parse, replies.ask, replies.split_reply

C = Question("c", text="How many points do you contribute to the project?", kind="number", minimum=0, maximum=20,
             required=False, retry=True)
PEERS = [Question("d%d" % k, text="Points for Participant %s:" % p, kind="number", minimum=0, maximum=10,
                  required=False, retry=True) for k, p in enumerate("ABC")]


@pytest.mark.parametrize("reply,value", [
    ("I keep 5 points and contribute 15.", 15), ("I keep 20 and contribute 0", 0), ("Round 3: I contribute 8", 8),
    ("contribute: 7", 7), ("I contribute 10 points", 10), ("10", 10), ("ten", 10), ("10 MU", 10)])
def test_the_number_after_the_asked_verb(reply, value):
    assert parse(reply, C).get("value") == value


@pytest.mark.parametrize("reply", ["I contribute 5 or 6, and contribute 7", "I don't contribute 5", "I keep 5",
                                   "between 3 and 8", "12.5"])
def test_unsure_or_opposite_numbers_are_not_read(reply):
    assert parse(reply, C)["status"] != "ok"


def test_keep_questions_and_questions_without_a_verb():
    keep = Question("k", text="How many points do you keep?", kind="number", minimum=0, maximum=20)
    assert parse("I give 5 and keep 15", keep)["value"] == 15
    assert parse("I give 5", keep)["status"] == "ambiguous"
    assert parse("3 then 4", Question("n", text="Your number", kind="number"))["status"] == "ambiguous"


@pytest.mark.parametrize("reply", [
    "A: 2, B: 0, C: 3", "Participant A: 2\nParticipant B: 0\nParticipant C: 3",
    "Points for Participant A: 2\nPoints for Participant B: 0\nPoints for Participant C: 3",
    "2, 0, 3", '{"A": 2, "B": 0, "C": 3}', "Q1: 2\nQ2: 0\nQ3: 3"])
def test_answers_keyed_by_the_questions_own_labels(reply):
    assert split_reply(reply, PEERS) == {"d0": "2", "d1": "0", "d2": "3"}


def test_labels_that_do_not_name_one_answer_are_not_read():
    assert split_reply("A: 2, A: 3, B: 1", PEERS) == {"d1": "1"}
    assert split_reply("I assign 0 points to everyone", PEERS) == {}
    assert split_reply("2, 0", PEERS) == {}
    mixed = [Question("x", text="Pick one", options=["Yes", "No"]), Question("y", text="Your number", kind="number")]
    assert split_reply("1, 5", mixed) == {}
    verbs = [Question("give", text="How much do you give?", kind="number"),
             Question("keep", text="How much do you keep?", kind="number")]
    assert split_reply("give: 4; keep: 6", verbs) == {"give": "4", "keep": "6"}


def test_optional_answers_with_retry_are_asked_again_and_never_defaulted():
    answers = iter(["I am not sure", "I keep 5 and contribute 15"])
    page = ask(lambda prompt: next(answers), "screen", [C])
    assert (page.value("c"), len(page.replies)) == (15, 2)
    answers = iter(["hmm", "still unsure"])
    page = ask(lambda prompt: next(answers), "screen", [C])
    assert page.value("c") is None and page.answers["c"]["status"] == "unparsed" and page.ok
    seen = iter(["A: 2, B: x", "B: 1, C: 0"])
    page = ask(lambda prompt: next(seen), "screen", PEERS)
    assert [page.value(q.id) for q in PEERS] == [2, 1, 0]
    plain = Question("w", text="Comments?", options=["A", "B"], codes=["x", "y"], required=False)
    calls = []
    ask(lambda prompt: calls.append(prompt) or "no idea", "screen", [plain])
    assert len(calls) == 1      # without retry=True an optional answer is asked once, as before


def test_the_stub_writes_verb_sentences_and_labelled_answers_that_are_read():
    labelled, sentences = set(), set()
    for seed in range(30):
        stub = replies.StubParticipant(seed=seed, noise=0.0, use_keys=False)
        page = ask(stub, replies.render_questions(PEERS), PEERS)
        assert all(page.answers[q.id]["status"] == "ok" for q in PEERS) and not stub.misreads
        labelled.add("Q1:" not in page.replies[0])
        stub = replies.StubParticipant(seed=seed, noise=0.0, use_keys=False)
        page = ask(stub, replies.render_questions([C]), [C])
        assert page.answers["c"]["status"] == "ok" and not stub.misreads
        sentences.add("contribute" in page.replies[0])
    assert labelled == {True, False} and sentences == {True, False}


@pytest.mark.parametrize("text,reply", [
    ("Choose a number between 0 and 100", "My answer is 33, between 0 and 100"),
    ("Choose a number between 0 and 100", "I choose 33, which is between 0 and 100."),
    ("How many points do you assign to member 3?", "assign 2 to member 1 and 5 to member 3"),
    ("How much do you allocate to the other participant?", "I allocate 60 to myself and 40 to the other"),
    ("How many points do you contribute?", "I'd contribute 10 in round 1 and 5 now")])
def test_a_leftover_number_or_a_lookalike_word_is_never_read(text, reply):
    assert parse(reply, Question("q", text=text, kind="number"))["status"] == "ambiguous"


def test_the_question_names_the_side_that_is_asked():
    both = Question("b", text="How many of your 20 points do you contribute? The points you do not contribute you "
                              "keep.", kind="number", minimum=0, maximum=20)
    assert parse("I keep 5", both)["status"] == "ambiguous"
    assert parse("I keep 11 and contribute 9.", both)["value"] == 9


def test_a_labelled_answer_keeps_its_commas_and_thousands_are_not_split():
    qs = [Question("contribute", text="How much do you contribute?", kind="number"),
          Question("influenced", text="What influenced you?", options=["Fairness", "Money", "Others"], multiple=True),
          Question("comments", text="Any comments?", kind="text", required=False)]
    assert split_reply("contribute: 10\ninfluenced: 1, 3\ncomments: It was fine, though long", qs) == {
        "contribute": "10", "influenced": "1, 3", "comments": "It was fine, though long"}
    two = [Question("a", text="first", kind="number"), Question("b", text="second", kind="number")]
    assert split_reply("1,000", two) == {}
