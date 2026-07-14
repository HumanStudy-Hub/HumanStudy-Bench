from __future__ import annotations

import re
from typing import Any


IDENTIFIER_MAX_LEN = 80


def slugify_identifier(value: Any, *, fallback: str = "study") -> str:
    """Return a stable ASCII identifier without assigning an entity type."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_") or fallback
    return text[:IDENTIFIER_MAX_LEN].rstrip("_") or fallback


def canonical_sub_study_id(value: Any, *, fallback: str = "study") -> str:
    """Canonical identifier used by Stage 3 materials, targets, and Stage 4."""
    slug = slugify_identifier(value, fallback=fallback)
    if slug == "study" or slug.startswith("study_"):
        return slug
    if slug.startswith("study"):
        return f"study_{slug[len('study'):].lstrip('_')}"
    if slug == "pilot" or slug.startswith("pilot_"):
        return slug
    if slug.startswith("pilot"):
        return f"pilot_{slug[len('pilot'):].lstrip('_')}"
    return f"study_{slug}"
