"""Approximate restoration of held original text from the researcher's own copy of a source.

A published study holds back wording it may not redistribute and records where to find it. The exact path (hsenv
rehydrate) verifies every restored passage against its sha256. Some passages cannot pass that check even with the
right file: the build team corrected OCR errors of a scanned paper, or the PDF reader puts an invisible break
character inside a word. For those, a release may carry an `approx` fingerprint: the pages the passage is cited on,
its word count and salted 16-bit hashes of its word trigrams. Nothing readable is published: a 16-bit hash fits
thousands of trigrams.

On the researcher's computer the passage is the window of that page whose trigrams best match the fingerprint. It is
used only above a fixed score and when no other place on the page matches nearly as well, and it is reported as
approximate (the run says the wording came from the researcher's copy and may contain reading errors). A window that
is not a complete match must begin with the passage's first trigram and end with its last (`ends`), so reading
errors are accepted only inside a passage, never at an edge where words could be missing.

COPY of humanstudy-desktop pack/runtime/lib/hs_approx_text.py (the app restores with it; this repo writes the
fingerprints). Change both together.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata

VERSION = 1
MIN_WORDS = 6          # shorter passages (a scale label, a button) are placed by their exact words instead
THRESHOLD = 0.7        # share of the passage's trigrams found in the chosen window
MARGIN = 0.1           # a second, separate window this close to the best makes the place ambiguous
SPREAD = 0.25          # window lengths tried: the passage's word count +- this share

_INVISIBLE = re.compile("[­​‌‍⁠﻿￾�]")
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])[-‐]\s*\n\s*(?=[^\W\d_])")
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+")
# numbers, comparison operators and signs (a dash only before a digit), in order: they are not words, so trigrams
# cannot see them
_NUM = re.compile(r"\d+(?:[.,]\d+)*|[<>≤≥=±~+%]|[-−‐‑–—](?=\s*\d)")
_DASHES = str.maketrans({c: "-" for c in "−‐‑–—"})


def clean(text: str) -> str:
    """Readable text of a passage or a page: compatibility forms folded (ligatures), invisible characters removed,
    a word hyphenated across a line joined, whitespace collapsed."""
    t = unicodedata.normalize("NFKC", text)
    t = _INVISIBLE.sub("", t)
    t = _HYPHEN_BREAK.sub("", t)
    return _SPACE.sub(" ", t).strip()


def words(text: str) -> list[tuple[str, int, int]]:
    """(lower-case word, start, end) in `text` (already cleaned)."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(text)]


def _gram(salt: str, w: list[str], i: int) -> str:
    return hashlib.sha256((salt + "\x1f" + " ".join(w[i:i + 3])).encode("utf-8")).hexdigest()[:4]


def _seq(salt: str, w: list[str]) -> str:
    return hashlib.sha256((salt + "\x1e" + " ".join(w)).encode("utf-8")).hexdigest()[:8]


def fingerprint(text: str, pages: list[int], salt: str | None = None) -> dict | None:
    """The approx entry of one held passage, or None when it has no words. A passage shorter than MIN_WORDS carries
    a hash of its whole word sequence (found only where the same words stand); a longer one its trigram hashes."""
    c = clean(text)
    w = [x for x, _, _ in words(c)]
    if not w:
        return None
    salt = salt or os.urandom(4).hex()
    fp = {"v": VERSION, "pages": sorted({int(p) for p in pages if int(p) >= 1}), "words": len(w), "salt": salt}
    if len(w) < MIN_WORDS:
        fp["seq"] = _seq(salt, w)
    else:
        fp["grams"] = sorted({_gram(salt, w, i) for i in range(len(w) - 2)})
        # the passage's first and last trigram: a window that is not a complete match must begin and end with them,
        # so a misread passage is never placed with words cut off at either end (a "Do not" on the page before)
        fp["ends"] = [_gram(salt, w, 0), _gram(salt, w, len(w) - 3)]
    # Leading / trailing characters around the words the window should keep: a quote mark or a closing period, and
    # a sign or operator ("-5 points", "≤ 5 minutes"), which `nums` then requires.
    m = list(_WORD.finditer(c))
    lead, tail = c[:m[0].start()], c[m[-1].end():]
    fp["nums"] = _numbers(salt, c)
    if lead:
        fp["lead"] = lead
    if tail:
        fp["tail"] = tail
    return fp


def _numbers(salt: str, text: str) -> str:
    """Hash of the passage's numbers, comparison operators and signs in order ("5 ≤ x", "-5", "50%"): a reading that
    loses or changes one (a payoff of -5 read as 5, ≤ read as ≥) never passes."""
    toks = [m.group(0).translate(_DASHES) for m in _NUM.finditer(clean(text))]
    return hashlib.sha256((salt + "\x1d" + "|".join(toks)).encode("utf-8")).hexdigest()[:8]


def _best(page: str, fp: dict, exclude: tuple[int, int] | None = None) -> tuple[float, int, int]:
    """(score, start word, end word) of the best window of one cleaned page (none overlapping `exclude`)."""
    ws = words(page)
    n, salt, grams = int(fp["words"]), fp["salt"], set(fp["grams"])
    if len(ws) < 3 or not grams:
        return 0.0, 0, 0
    w = [x for x, _, _ in ws]
    # the gram at each position when it is one of the passage's (else None); a window scores its DISTINCT grams, so
    # a repeated phrase cannot stand in for the words it lacks
    hit = [g if g in grams else None for g in (_gram(salt, w, i) for i in range(len(w) - 2))]
    best = (0.0, 0, 0)
    lo, hi = max(3, int(n * (1 - SPREAD))), int(n * (1 + SPREAD)) + 1
    for length in range(lo, min(hi, len(w)) + 1):
        span = length - 2   # grams inside a window of `length` words
        count: dict[str, int] = {}
        for g in hit[:span]:
            if g is not None:
                count[g] = count.get(g, 0) + 1
        for a in range(0, len(w) - length + 1):
            if a:
                out_g, in_g = hit[a - 1], hit[a + span - 1]
                if out_g is not None:
                    count[out_g] -= 1
                    if not count[out_g]:
                        del count[out_g]
                if in_g is not None:
                    count[in_g] = count.get(in_g, 0) + 1
            if exclude and a < exclude[1] and a + length > exclude[0]:
                continue
            # recall against the passage's distinct grams, penalised when the window is longer than the passage
            score = min(1.0, len(count) / len(grams)) * min(1.0, (n - 2) / max(1, span))
            if score > best[0]:
                best = (score, a, a + length)
    return best


def locate(page_texts: list[str], fp: dict) -> tuple[str, float] | None:
    """(restored passage, score) from the researcher's pages, or None when it cannot be placed safely."""
    if not isinstance(fp, dict) or fp.get("v") != VERSION or not (fp.get("grams") or fp.get("seq")):
        return None
    cited = [p for p in fp.get("pages") or [] if isinstance(p, int)]
    # cited page and its neighbours first; then the rest of the file (a cite may give the journal's page number)
    near = [p for c in cited for p in (c, c - 1, c + 1) if 1 <= p <= len(page_texts)]
    order = list(dict.fromkeys(near + list(range(1, len(page_texts) + 1))))
    if fp.get("seq"):   # a short passage: the first place (cited page first) where exactly its words stand
        n = int(fp["words"])
        for p in order:
            page = clean(page_texts[p - 1])
            ws = words(page)
            w = [x for x, _, _ in ws]
            for a in range(0, len(w) - n + 1):
                if _seq(fp["salt"], w[a:a + n]) == fp["seq"]:
                    span = _span(page, ws, a, a + n, fp)
                    return (span, 1.0) if _nums_ok(span, fp) else None
        return None
    found = []
    grams = set(fp["grams"])
    for p in order:
        page = clean(page_texts[p - 1])
        # a page holding too few of the passage's trigrams at all cannot reach the threshold: skip the window scan
        w = [x for x, _, _ in words(page)]
        if len({g for g in (_gram(fp["salt"], w, i) for i in range(len(w) - 2)) if g in grams}) < (THRESHOLD - MARGIN) * len(grams):
            continue
        score, a, b = _best(page, fp)
        if score >= THRESHOLD - MARGIN:
            found.append((score, p, a, b, page))
    if not found:
        return None
    found.sort(key=lambda x: -x[0])
    score, p, a, b, page = found[0]
    if score < THRESHOLD:
        return None
    # ambiguous: another page, or a separate window of the same page, matches nearly as well
    if any(s2 >= score - MARGIN for s2, *_ in found[1:]) or _best(page, fp, exclude=(a, b))[0] >= score - MARGIN:
        return None
    ws = words(page)
    place = _boundary(page, ws, fp, a, b - a, score)
    if place is None:
        return None
    span = _span(page, ws, place[0], place[0] + place[1], fp)
    return (span, round(score, 3)) if _nums_ok(span, fp) else None


def _nums_ok(span: str, fp: dict) -> bool:
    """The placed text has the passage's numbers, signs and operators (`nums`); a fingerprint without it is not
    placed."""
    return isinstance(fp.get("nums"), str) and _numbers(fp["salt"], span) == fp["nums"]


def _window(ws: list, fp: dict, a: int, length: int) -> float:
    """The score of one window (as _best scores it)."""
    n, grams = int(fp["words"]), set(fp["grams"])
    w = [x for x, _, _ in ws[a:a + length]]
    got = {g for g in (_gram(fp["salt"], w, i) for i in range(len(w) - 2)) if g in grams}
    return min(1.0, len(got) / len(grams)) * min(1.0, (n - 2) / max(1, length - 2))


def _boundary(page: str, ws: list, fp: dict, a: int, length: int, score: float) -> tuple[int, int] | None:
    """(start, length) of the passage among the windows that score the same as the best one, shifted by up to two
    words or longer by up to the passage's word count: a misread word at an edge gives no trigram hit, so a window
    one word shorter or shifted scores the same (and could drop a word such as "Never" or "not"). The passage's own
    leading / trailing punctuation decides; when it cannot, the place is ambiguous (None) and the item is left to
    the researcher."""
    lead, tail = fp.get("lead") or "", fp.get("tail") or ""
    n = int(fp["words"])
    ties = [(x, k) for k in range(length, max(length, n) + 1) for x in range(max(0, a - 2), a + 3)
            if x + k <= len(ws) and ((x, k) == (a, length) or abs(_window(ws, fp, x, k) - score) < 1e-9)]
    if len(ties) == 1:
        return ties[0] if _ends(ws, fp, ties[0], score) else None

    def edges(t):
        start, end = ws[t[0]][1], ws[t[0] + t[1] - 1][2]
        return (bool(lead) and page[max(0, start - len(lead)):start] == lead) + \
            (bool(tail) and page[end:end + len(tail)] == tail)
    best = max(edges(t) for t in ties)
    picked = [t for t in ties if edges(t) == best]
    return picked[0] if best and len(picked) == 1 and _ends(ws, fp, picked[0], score) else None


def _ends(ws: list, fp: dict, place: tuple[int, int], score: float) -> bool:
    """The window begins with the passage's first trigram and ends with its last (`ends`), whatever its score: a
    match that holds every distinct trigram can still be clipped where the passage repeats itself ("Do not move. Do
    not move."), and a lesser one could miss words at an edge (a "Do not" at the foot of the page before). Otherwise
    the item is left to the researcher. Nor is a window with fewer words than the passage. A fingerprint without
    `ends` is never placed."""
    ends = fp.get("ends")
    if not (isinstance(ends, list) and len(ends) == 2 and all(isinstance(e, str) for e in ends)):
        return False
    x, k = place
    if k < int(fp["words"]):
        return False   # fewer words than the passage: something is missing (a repeated passage can keep every trigram)
    w = [t for t, _, _ in ws[x:x + k]]
    return len(w) >= 3 and _gram(fp["salt"], w, 0) == ends[0] and _gram(fp["salt"], w, len(w) - 3) == ends[1]


def _span(page: str, ws: list, a: int, b: int, fp: dict) -> str:
    """The text of words a..b, with as much of the passage's leading / trailing characters as the page has right
    next to them (a straight quote where the passage had a curly one stops the extension, never the words)."""
    start, end = ws[a][1], ws[b - 1][2]
    lead, tail = fp.get("lead") or "", fp.get("tail") or ""
    i = 0
    while i < len(lead) and start - i - 1 >= 0 and page[start - i - 1] == lead[len(lead) - 1 - i]:
        i += 1
    j = 0
    while j < len(tail) and end + j < len(page) and page[end + j] == tail[j]:
        j += 1
    return page[start - i:end + j]
