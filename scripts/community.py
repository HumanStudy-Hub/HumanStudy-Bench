"""Shared helpers for the community/ layout: paper keys, folder names, maintainers."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMUNITY_DIR = REPO_ROOT / "community"

# The single place that lists who may edit other people's contributions.
MAINTAINERS = ("XuanL17", "diana3135", "assassin808")
# The Hub's bot account. Overridable so a dogfood deployment can publish with another account.
BOT_LOGIN = os.environ.get("COMMUNITY_BOT_LOGIN", "humanstudy-bot")

MAX_FILE_BYTES = 25 * 1024 * 1024

# <slug>--<contributor-id>[-vN]
CONTRIBUTION_DIR_RE = re.compile(
    r"^(?P<slug>[a-z0-9][a-z0-9-]*?)--(?P<contributor>[a-z0-9][a-z0-9-]*?)"
    r"(?:-v(?P<version>[2-9]|[1-9][0-9]+))?$"
)

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def normalise_doi(doi: str) -> str:
    """Lower-case a DOI and strip any URL or ``doi:`` prefix."""
    return _DOI_PREFIX_RE.sub("", doi.strip()).strip().lower()


def _fold(text: str) -> str:
    """Strip accents so 'Müller' and 'Muller' give the same key."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalise_title(title: str, max_length: int = 60) -> str:
    """Lower-case ASCII words joined by single hyphens, cut at ``max_length`` on a word edge."""
    words = re.sub(r"[^a-z0-9]+", "-", _fold(title).lower()).strip("-")
    if len(words) <= max_length:
        return words
    cut = words[:max_length]
    if words[max_length] != "-" and "-" in cut:
        cut = cut.rsplit("-", 1)[0]
    return cut.strip("-")


def surname(author: str) -> str:
    """Last name of 'Joshua Knobe' or 'Knobe, Joshua', normalised like a title word."""
    author = author.strip()
    part = author.split(",")[0] if "," in author else (author.split() or [""])[-1]
    return normalise_title(part, max_length=30)


def make_paper_key(doi: Optional[str], title: str, year: Optional[int], first_author: str) -> str:
    """``doi:<doi>`` when a DOI is known, else ``title:<title>-<year>-<surname>``."""
    if doi and normalise_doi(doi).startswith("10."):
        return "doi:" + normalise_doi(doi)
    parts = [normalise_title(title), str(year) if year else "nd", surname(first_author) or "anon"]
    return "title:" + "-".join(p for p in parts if p)


def paper_key_to_dir(key: str) -> str:
    """Folder name for a paper key: every character outside ``A-Za-z0-9._-`` becomes ``_``."""
    return _UNSAFE_RE.sub("_", key)


def parse_contribution_dir(name: str) -> Optional[dict]:
    """Split a folder name into slug, contributor id and version; None if it does not fit."""
    match = CONTRIBUTION_DIR_RE.match(name)
    if not match:
        return None
    return {
        "slug": match.group("slug"),
        "contributor": match.group("contributor"),
        "version": int(match.group("version") or 1),
    }


def github_matches(handle: Optional[str], login: str) -> bool:
    return bool(handle) and str(handle).strip().lstrip("@").lower() == login.strip().lower()


def is_maintainer(login: str) -> bool:
    return login.lower() in {m.lower() for m in MAINTAINERS}
