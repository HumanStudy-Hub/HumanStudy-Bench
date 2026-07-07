"""OSF link discovery from paper JSON, PDF text, and optional LLM assistance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from generation_pipeline.utils.osf_crawler import OSF_URL_RE, parse_osf_url
from generation_pipeline.utils.pdf_extractor import extract_pdf_text

PDF_TEXT_MAX_CHARS = 400000
DISCOVERY_PROMPT_MAX_CHARS = 80000


def discover_osf_links_from_text(text: str) -> list[str]:
    """Return unique normalized OSF links found in free text."""
    links: list[str] = []
    seen: set[str] = set()
    for match in OSF_URL_RE.finditer(text or ""):
        # The shared OSF regex intentionally accepts 5-char node ids, but in
        # free text it can otherwise match the prefix of longer OSF paths such
        # as /preprints/. Treat an immediately-following alphanumeric
        # character as a non-node path and skip it.
        if match.end() < len(text or "") and (text or "")[match.end()].isalnum():
            continue
        raw = match.group(0).rstrip(").,;]")
        try:
            osf_id, view_only = parse_osf_url(raw)
        except ValueError:
            continue
        normalized = f"https://osf.io/{osf_id}/"
        if view_only:
            normalized = f"{normalized}?view_only={view_only}"
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def discover_osf_links_from_json(data: dict[str, Any]) -> list[str]:
    """Scan paper metadata and string values for OSF links."""
    return discover_osf_links_from_text(json.dumps(data, ensure_ascii=False))


def discover_osf_links_from_pdf(pdf_path: Path) -> list[str]:
    """Extract PDF text and scan it for OSF links."""
    text = extract_pdf_text(Path(pdf_path), max_chars=PDF_TEXT_MAX_CHARS)
    return discover_osf_links_from_text(text)


class OsfLinkDiscovery:
    """Rule-first OSF link discovery with optional LLM fallback."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def discover(
        self,
        *,
        paper_json: dict[str, Any] | None = None,
        json_path: Path | None = None,
        pdf_path: Path | None = None,
        extra_texts: Iterable[str] = (),
    ) -> list[str]:
        data = paper_json
        if data is None and json_path is not None and Path(json_path).exists():
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        texts: list[str] = []
        if data is not None:
            texts.append(json.dumps(data, ensure_ascii=False))
        texts.extend(text for text in extra_texts if text)

        links = self._unique_links(*(discover_osf_links_from_text(text) for text in texts))
        if links:
            return links

        pdf_text = ""
        if pdf_path is not None and Path(pdf_path).exists():
            try:
                pdf_text = extract_pdf_text(Path(pdf_path), max_chars=PDF_TEXT_MAX_CHARS)
            except Exception:
                pdf_text = ""
            links = discover_osf_links_from_text(pdf_text)
            if links:
                return links

        if self.llm_client is None or not pdf_text:
            return []
        return self._discover_with_llm(pdf_text)

    def _discover_with_llm(self, text: str) -> list[str]:
        try:
            from src.llm.helpers import generate_json
        except Exception:
            return []

        system = (
            "You locate Open Science Framework (OSF) data/material links in paper text. "
            "Only return explicit osf.io URLs that appear in the provided text."
        )
        prompt = (
            "Find every explicit OSF URL in this paper text. Do not infer links. "
            "Respond as JSON: {\"osf_links\": [\"https://osf.io/...\"]}.\n\n"
            f"TEXT:\n{text[:DISCOVERY_PROMPT_MAX_CHARS]}"
        )
        try:
            data = generate_json(self.llm_client, [{"role": "user", "content": prompt}], system=system)
        except Exception:
            return []
        links = data.get("osf_links") if isinstance(data, dict) else None
        if not isinstance(links, list):
            return []
        return self._unique_links(discover_osf_links_from_text("\n".join(str(item) for item in links)))

    @staticmethod
    def _unique_links(*groups: Iterable[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for link in group:
                if link not in seen:
                    seen.add(link)
                    out.append(link)
        return out
