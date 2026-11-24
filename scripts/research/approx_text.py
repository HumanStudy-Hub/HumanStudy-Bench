"""Approximate restoration of held original text from the researcher's own copy of a source.

A published study holds back wording it may not redistribute and records where to find it. The exact path (hsenv
rehydrate) verifies every restored passage against its sha256. Some passages cannot pass that check even with the
right file: the build team corrected OCR errors of a scanned paper, or the PDF reader puts an invisible break
character inside a word. For those, a release may carry an `approx` fingerprint: the pages the passage is cited on,
its word count and salted 16-bit hashes of its word trigrams. Nothing readable is published: a 16-bit hash fits
thousands of trigrams.

On the researcher's computer the passage is the window of that page whose trigrams best match the fingerprint. It is
used only above a fixed score and when no other place on the page matches nearly as well, and it is reported as
approximate (the run says the wording came from the researcher's copy and may contain reading errors).

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
    # Leading / trailing punctuation the window should keep (a closing period or quote mark; never a word).
    lead = c[:len(c) - len(c.lstrip("\"'“‘(["))]
    tail = c[len(c.rstrip(".,;:!?\"'”’)]")):]
    if lead:
        fp["lead"] = lead
    if tail:
        fp["tail"] = tail
    return fp


def _best(page: str, fp: dict, exclude: tuple[int, int] | None = None) -> tuple[float, int, int]:
    """(score, start word, end word) of the best window of one cleaned page (none overlapping `exclude`)."""
    ws = words(page)
    n, salt, grams = int(fp["words"]), fp["salt"], set(fp["grams"])
    if len(ws) < 3 or not grams:
        return 0.0, 0, 0
    w = [x for x, _, _ in ws]
    hit = [1 if _gram(salt, w, i) in grams else 0 for i in range(len(w) - 2)]
    pre = [0]
    for h in hit:
        pre.append(pre[-1] + h)
    best = (0.0, 0, 0)
    lo, hi = max(3, int(n * (1 - SPREAD))), int(n * (1 + SPREAD)) + 1
    for length in range(lo, min(hi, len(w)) + 1):
        for a in range(0, len(w) - length + 1):
            if exclude and a < exclude[1] and a + length > exclude[0]:
                continue
            hits = pre[a + length - 2] - pre[a]
            # recall against the passage, penalised when the window is longer than the passage
            score = min(1.0, hits / (n - 2)) * min(1.0, (n - 2) / max(1, length - 2))
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
                    return _span(page, ws, a, a + n, fp), 1.0
        return None
    found = []
    for p in order:
        page = clean(page_texts[p - 1])
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
    return _span(page, words(page), a, b, fp), round(score, 3)


def _span(page: str, ws: list, a: int, b: int, fp: dict) -> str:
    start, end = ws[a][1], ws[b - 1][2]
    lead, tail = fp.get("lead") or "", fp.get("tail") or ""
    if lead and page[max(0, start - len(lead)):start] == lead:
        start -= len(lead)
    if tail and page[end:end + len(tail)] == tail:
        end += len(tail)
    return page[start:end]
