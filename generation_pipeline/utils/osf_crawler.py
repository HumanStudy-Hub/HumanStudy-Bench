"""
OSF crawler for ai-ethics stage4 corpus.

Scans per-paper JSON files, finds effects whose experiment materials live on OSF
(not in the paper PDF), resolves OSF node/registration URLs, and downloads
matching supplementary files.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

OSF_API = "https://api.osf.io/v2"
OSF_URL_RE = re.compile(
    r"https?://(?:www\.)?osf\.io/(?P<id>[a-z0-9]{5})/?(?:\?[^\s\"']*)?",
    re.IGNORECASE,
)
STUDY_NUM_RE = re.compile(r"(?:study|experiment|exp\.?)\s*([0-9]+[a-z]?)", re.IGNORECASE)

MATERIAL_KEYWORDS = (
    "manipulation",
    "material",
    "materials",
    "stimulus",
    "stimuli",
    "scale",
    "survey",
    "questionnaire",
    "items",
    "instrument",
    "measure",
    "appendix",
    "supplement",
    "supplementary",
    "instruction",
    "scenario",
    "script",
    "vignette",
    "task",
    "protocol",
)

SLOTS = ("materials", "manipulation", "items")
NEEDS_OSF_STATUSES = {"osf_only", "not_in_paper", "cited_scale"}


@dataclass
class EffectNeed:
    study: str
    study_key: str | None
    effect_index: int
    slot: str
    status: str
    iv: str
    dv: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OsfFile:
    name: str
    kind: str
    path: str
    size: int | None
    download_url: str | None
    api_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperOsfPlan:
    paper_folder: str
    json_path: str
    paper_title: str
    osf_url: str
    osf_id: str
    view_only: str | None
    entity_type: str | None = None
    entity_title: str | None = None
    effects_needing_osf: list[EffectNeed] = field(default_factory=list)
    matched_files: list[OsfFile] = field(default_factory=list)
    downloaded_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def can_download(self) -> bool:
        return bool(self.osf_id)

    @property
    def needs_download(self) -> bool:
        """True when missing materials AND an OSF id is present."""
        return bool(self.effects_needing_osf and self.osf_id)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effects_needing_osf"] = [e.to_dict() if isinstance(e, EffectNeed) else e for e in self.effects_needing_osf]
        d["matched_files"] = [f.to_dict() if isinstance(f, OsfFile) else f for f in self.matched_files]
        return d


class OsfClient:
    """Minimal OSF API v2 client (public + view_only; optional bearer token)."""

    def __init__(
        self,
        token: str | None = None,
        view_only: str | None = None,
        delay_s: float = 0.15,
        timeout_s: float = 120,
    ):
        self.token = token
        self.view_only = view_only
        self.delay_s = delay_s
        self.timeout_s = timeout_s

    def _request(self, url: str) -> dict[str, Any]:
        time.sleep(self.delay_s)
        parsed = urllib.parse.urlparse(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if self.view_only and "view_only" not in query:
            query["view_only"] = self.view_only
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))

        headers = {"Accept": "application/vnd.api+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OSF API {e.code} for {url}: {body[:300]}") from e

    def download(self, url: str, dest: Path) -> None:
        time.sleep(self.delay_s)
        parsed = urllib.parse.urlparse(url)
        query = dict(urllib.parse.parse_qsl(parsed.query))
        if self.view_only and "view_only" not in query:
            query["view_only"] = self.view_only
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))

        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, headers=headers)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                tmp.write_bytes(resp.read())
            tmp.replace(dest)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    def resolve_entity(self, osf_id: str) -> tuple[str, str, str]:
        """Return (entity_type, title, files_index_url)."""
        for entity_type in ("nodes", "registrations"):
            url = f"{OSF_API}/{entity_type}/{osf_id}/"
            try:
                payload = self._request(url)
            except RuntimeError as e:
                if "404" in str(e):
                    continue
                raise
            data = payload.get("data")
            if not data:
                continue
            title = data.get("attributes", {}).get("title", osf_id)
            files_url = data.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
            if not files_url:
                raise RuntimeError(f"No files relationship for {entity_type}/{osf_id}")
            return entity_type, title, files_url
        raise RuntimeError(f"Could not resolve OSF id '{osf_id}' as node or registration")

    def list_files_recursive(self, provider_url: str, prefix: str = "") -> list[OsfFile]:
        payload = self._request(provider_url)
        out: list[OsfFile] = []
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            name = attrs.get("name", "unknown")
            rel_path = f"{prefix}/{name}".lstrip("/")
            kind = attrs.get("kind", "file")
            download_url = item.get("links", {}).get("download")
            out.append(
                OsfFile(
                    name=name,
                    kind=kind,
                    path=rel_path,
                    size=attrs.get("size"),
                    download_url=download_url if kind == "file" else None,
                    api_url=item.get("links", {}).get("info") or provider_url,
                )
            )
            if kind == "folder":
                child_url = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                if child_url:
                    out.extend(self.list_files_recursive(child_url, rel_path))
        return out


def parse_osf_url(url: str) -> tuple[str, str | None]:
    """Extract (osf_id, view_only_token) from an OSF URL."""
    if not url:
        raise ValueError("Empty OSF URL")
    m = OSF_URL_RE.search(url)
    if not m:
        raise ValueError(f"Not an OSF URL: {url}")
    parsed = urllib.parse.urlparse(url)
    view_only = urllib.parse.parse_qs(parsed.query).get("view_only", [None])[0]
    return m.group("id").lower(), view_only


def extract_study_key(study_name: str) -> str | None:
    m = STUDY_NUM_RE.search(study_name or "")
    return m.group(1).lower() if m else None


def find_paper_jsons(stage4_dir: Path) -> list[Path]:
    """Find main extraction JSON for each paper (supports flat or runnable/non_runnable layout)."""
    jsons: list[Path] = []
    seen: set[str] = set()

    def consider(folder: Path) -> None:
        if folder.name in {"runnable", "non_runnable", "replicable", "non_replicable"}:
            return
        candidates = [
            p for p in folder.glob("*.json")
            if p.name not in {"sum.json", "osf_manifest.json"}
            and "manifest" not in p.name.lower()
        ]
        if candidates and folder.name not in seen:
            jsons.append(candidates[0])
            seen.add(folder.name)

    for folder in sorted(stage4_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in {"runnable", "non_runnable"}:
            for sub in sorted(folder.iterdir()):
                if sub.is_dir():
                    consider(sub)
        else:
            consider(folder)
    return jsons


def analyze_paper_needs(data: dict[str, Any]) -> tuple[str | None, str | None, list[EffectNeed]]:
    """Return (osf_url, view_only, effects_needing_osf)."""
    meta = data.get("paper_metadata") or {}
    link = meta.get("link") or ""
    osf_url: str | None = None
    view_only: str | None = None
    if "osf.io" in link.lower():
        osf_url = link
        try:
            _, view_only = parse_osf_url(link)
        except ValueError:
            pass

    needs: list[EffectNeed] = []
    for study in data.get("eligible_studies", []):
        study_name = study.get("study", "")
        study_key = extract_study_key(study_name)
        for ei, effect in enumerate(study.get("effects", [])):
            notes = (effect.get("materials_notes") or "").lower()
            notes_osf = "osf" in notes or "open science framework" in notes
            for slot in SLOTS:
                slot_obj = effect.get(slot) or {}
                status = slot_obj.get("status")
                content = slot_obj.get("content")
                missing = content is None or (isinstance(content, str) and not content.strip())
                if status == "osf_only" or (status in NEEDS_OSF_STATUSES and missing) or (notes_osf and missing):
                    needs.append(
                        EffectNeed(
                            study=study_name,
                            study_key=study_key,
                            effect_index=ei,
                            slot=slot,
                            status=status or "unknown",
                            iv=effect.get("IV", "") or "",
                            dv=effect.get("DV", "") or "",
                        )
                    )
    return osf_url, view_only, needs


def build_paper_plan(json_path: Path) -> PaperOsfPlan:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    folder = json_path.parent.name
    osf_url, view_only, needs = analyze_paper_needs(data)

    osf_id = ""
    if osf_url:
        try:
            osf_id, parsed_view = parse_osf_url(osf_url)
            view_only = view_only or parsed_view
        except ValueError:
            pass

    # Only download when experiments actually reference missing OSF materials.
    if not needs:
        osf_url = osf_url or ""
    elif not osf_id:
        return PaperOsfPlan(
            paper_folder=folder,
            json_path=str(json_path),
            paper_title=data.get("paper_title", ""),
            osf_url=osf_url or "",
            osf_id="",
            view_only=view_only,
            effects_needing_osf=needs,
            errors=["Paper has effects needing OSF materials but no osf.io link in paper_metadata.link"],
        )

    return PaperOsfPlan(
        paper_folder=folder,
        json_path=str(json_path),
        paper_title=data.get("paper_title", ""),
        osf_url=osf_url or "",
        osf_id=osf_id,
        view_only=view_only,
        effects_needing_osf=needs,
    )


def _study_matches_path(study_key: str | None, study_name: str, path_lower: str) -> bool:
    if study_key and re.search(rf"(?:study|experiment|exp\.?)\s*{re.escape(study_key)}\b", path_lower):
        return True
    slug = re.sub(r"[^a-z0-9]+", " ", (study_name or "").lower()).strip()
    tokens = [t for t in slug.split() if len(t) > 3 and t not in {"study", "experiment"}]
    return any(t in path_lower for t in tokens[:3])


def score_file_relevance(path_lower: str, needs: list[EffectNeed]) -> float:
    score = 0.0
    for need in needs:
        if _study_matches_path(need.study_key, need.study, path_lower):
            score += 5.0
    for kw in MATERIAL_KEYWORDS:
        if kw in path_lower:
            score += 1.0
    # Deprioritize raw data / syntax unless nothing else matches.
    if any(x in path_lower for x in ("data", "syntax", "output", ".sav", ".r", ".spss")):
        score -= 1.5
    if path_lower.endswith((".pdf", ".docx", ".doc", ".txt", ".html", ".htm")):
        score += 0.5
    return score


HIGH_VALUE_KEYWORDS = (
    "appendix",
    "supplement",
    "supplementary",
    "manipulation",
    "material",
    "scale",
    "survey",
    "questionnaire",
    "instrument",
    "measure",
    "items",
    "stimulus",
    "scenario",
    "instruction",
)


def select_files(all_files: list[OsfFile], needs: list[EffectNeed], download_all: bool = False) -> list[OsfFile]:
    files = [f for f in all_files if f.kind == "file" and f.download_url]
    if download_all:
        return files

    selected: dict[str, OsfFile] = {}
    scored: list[tuple[float, OsfFile]] = []
    for f in files:
        path_lower = f.path.lower()
        s = score_file_relevance(path_lower, needs)
        if s > 0:
            scored.append((s, f))
        if any(kw in path_lower for kw in HIGH_VALUE_KEYWORDS):
            selected[f.path] = f

    if scored:
        scored.sort(key=lambda x: (-x[0], x[1].path))
        max_score = scored[0][0]
        threshold = max(1.0, max_score - 2.0)
        for s, f in scored:
            if s >= threshold:
                selected[f.path] = f

    if selected:
        return [selected[k] for k in sorted(selected.keys())]

    # Fallback: if we know materials are missing, grab everything except obvious data dumps.
    fallback = [
        f for f in files
        if not any(x in f.path.lower() for x in (".sav", ".csv", ".rdata", "syntax", "prereg"))
    ]
    return fallback or files


def fetch_osf_for_plan(
    plan: PaperOsfPlan,
    output_root: Path | None = None,
    token: str | None = None,
    download_all: bool = False,
    dry_run: bool = False,
) -> PaperOsfPlan:
    if not plan.can_download:
        return plan

    # JSON already complete → grab full OSF storage for archival.
    if not plan.effects_needing_osf:
        download_all = True

    out_dir = (output_root or Path(plan.json_path).parent) / "osf"
    files_dir = out_dir / "files"
    client = OsfClient(token=token, view_only=plan.view_only)

    try:
        entity_type, title, files_index = client.resolve_entity(plan.osf_id)
        plan.entity_type = entity_type
        plan.entity_title = title

        providers = client._request(files_index).get("data", [])
        all_files: list[OsfFile] = []
        for provider in providers:
            provider_files_url = (
                provider.get("relationships", {})
                .get("files", {})
                .get("links", {})
                .get("related", {})
                .get("href")
            )
            if provider_files_url:
                all_files.extend(client.list_files_recursive(provider_files_url))

        plan.matched_files = select_files(all_files, plan.effects_needing_osf, download_all=download_all)
        if not plan.matched_files:
            if not all_files:
                plan.errors.append("OSF project has no files in storage")
            else:
                plan.errors.append("No OSF files matched selection criteria")
            return plan

        for f in plan.matched_files:
            dest = files_dir / f.path
            if dry_run:
                plan.skipped_files.append(str(dest))
                continue
            try:
                client.download(f.download_url, dest)
                plan.downloaded_files.append(str(dest))
            except Exception as e:
                plan.errors.append(f"Failed to download {f.path}: {e}")

        if not dry_run:
            manifest_path = out_dir / "osf_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"plan": plan.to_dict(), "all_osf_files": [x.to_dict() for x in all_files]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    except Exception as e:
        plan.errors.append(str(e))

    return plan


def scan_stage4(
    stage4_dir: Path,
    only: Iterable[str] | None = None,
    *,
    all_linked: bool = False,
    missing_only: bool = False,
) -> list[PaperOsfPlan]:
    plans: list[PaperOsfPlan] = []
    for json_path in find_paper_jsons(stage4_dir):
        if only and not any(sub in json_path.as_posix() for sub in only):
            continue
        plan = build_paper_plan(json_path)
        if not plan.osf_id:
            continue
        if missing_only and not plan.effects_needing_osf:
            continue
        if all_linked or plan.effects_needing_osf:
            plans.append(plan)
    return plans


def run_batch(
    stage4_dir: Path,
    only: list[str] | None = None,
    token: str | None = None,
    download_all: bool = False,
    dry_run: bool = False,
    list_only: bool = False,
    *,
    all_linked: bool = True,
    missing_only: bool = False,
) -> list[PaperOsfPlan]:
    plans = scan_stage4(
        stage4_dir,
        only=only,
        all_linked=all_linked,
        missing_only=missing_only,
    )
    if list_only:
        return plans
    results: list[PaperOsfPlan] = []
    for plan in plans:
        if not plan.osf_id:
            results.append(plan)
            continue
        results.append(
            fetch_osf_for_plan(
                plan,
                token=token,
                download_all=download_all,
                dry_run=dry_run,
            )
        )
    return results
