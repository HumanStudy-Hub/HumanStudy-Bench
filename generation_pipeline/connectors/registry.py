"""OSF-only source connector registry"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generation_pipeline.connectors.base_connector import (
    BaseSourceConnector,
    ConnectorFetchResult,
    FetchedSource,
    SourceFetchPlan,
    iter_text_files,
    safe_relative_path,
    write_json,
)
from generation_pipeline.connectors.osf_connector import OsfConnector
from generation_pipeline.connectors.osf_link_discovery import OsfLinkDiscovery
from generation_pipeline.utils.osf_crawler import find_paper_jsons


class SourceConnectorRegistry:
    """Registry that currently dispatches only OSF source connectors."""

    def __init__(
        self,
        connectors: Iterable[BaseSourceConnector] | None = None,
        *,
        link_discovery: OsfLinkDiscovery | None = None,
        extract_text: bool = True,
        llm_client: Any | None = None,
    ):
        self.connectors = list(connectors) if connectors is not None else [OsfConnector()]
        self.link_discovery = link_discovery or OsfLinkDiscovery(llm_client=llm_client)
        self.extract_text = extract_text

    def connector_for(self, link: str) -> BaseSourceConnector | None:
        for connector in self.connectors:
            if connector.detect(link):
                return connector
        return None

    def plans_for_json(
        self,
        json_path: Path,
        *,
        pdf_path: Path | None = None,
    ) -> list[SourceFetchPlan]:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        links = self.link_discovery.discover(paper_json=data, json_path=json_path, pdf_path=pdf_path)
        folder = Path(json_path).parent.name
        metadata = data.get("paper_metadata") if isinstance(data.get("paper_metadata"), dict) else {}
        plans: list[SourceFetchPlan] = []
        seen: set[str] = set()
        for link in links:
            connector = self.connector_for(link)
            if connector is None or link in seen:
                continue
            seen.add(link)
            plans.append(
                SourceFetchPlan(
                    paper_folder=folder,
                    json_path=str(json_path),
                    paper_title=str(data.get("paper_title") or ""),
                    link=link,
                    connector=connector.name,
                    doi=metadata.get("doi"),
                    metadata=metadata,
                )
            )
        return plans

    def fetch_for_json(
        self,
        json_path: Path,
        *,
        pdf_path: Path | None = None,
        dest_dir: Path | None = None,
    ) -> dict[str, Any]:
        json_path = Path(json_path)
        dest_dir = Path(dest_dir) if dest_dir is not None else json_path.parent / "sources"

        reused = self._reuse_paired_pdf_sources(json_path, pdf_path, dest_dir, plans=[])
        if reused is not None:
            return reused

        plans = self.plans_for_json(json_path, pdf_path=pdf_path)

        dest_dir.mkdir(parents=True, exist_ok=True)
        results: list[ConnectorFetchResult] = []
        for plan in plans:
            connector = self.connector_for(plan.link)
            if connector is None:
                results.append(
                    ConnectorFetchResult(
                        connector=plan.connector or "unknown",
                        plan=plan,
                        dest=str(dest_dir),
                        errors=[f"No connector registered for link: {plan.link}"],
                    )
                )
                continue
            connector_dest = dest_dir / connector.name
            try:
                fetched = connector.fetch(plan, connector_dest)
                result = ConnectorFetchResult(
                    connector=connector.name,
                    plan=plan,
                    dest=str(connector_dest),
                    fetched=fetched,
                )
            except Exception as exc:
                result = ConnectorFetchResult(
                    connector=connector.name,
                    plan=plan,
                    dest=str(connector_dest),
                    errors=[str(exc)],
                )
            if self.extract_text:
                self._extract_text_bundle(connector, result, dest_dir)
            results.append(result)

        manifest = {
            "json_path": str(json_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "sources_dir": str(dest_dir),
            "plans": [plan.to_dict() for plan in plans],
            "results": [result.to_dict() for result in results],
            "summary": {
                "plans": len(plans),
                "connectors": sorted({result.connector for result in results}),
                "fetched_files": sum(len(result.fetched) for result in results),
                "skipped_files": sum(len(result.skipped) for result in results),
                "errors": sum(len(result.errors) for result in results),
            },
        }
        write_json(dest_dir / "source_manifest.json", manifest)
        return manifest

    def _reuse_paired_pdf_sources(
        self,
        json_path: Path,
        pdf_path: Path | None,
        dest_dir: Path,
        plans: list[SourceFetchPlan],
    ) -> dict[str, Any] | None:
        """Reuse local sources next to the paired PDF before downloading."""
        pdf_dir = Path(pdf_path).parent if pdf_path else None

        complete_candidates: list[Path] = []
        if pdf_dir is not None:
            complete_candidates.append(pdf_dir / "sources")
        complete_candidates.append(dest_dir)
        for source_dir in complete_candidates:
            if (source_dir / "combined_sources.txt").exists():
                files = _source_payload_files(source_dir)
                return self._local_reuse_manifest(
                    json_path=json_path,
                    pdf_path=pdf_path,
                    sources_dir=source_dir,
                    plans=plans,
                    files=files,
                    reuse_mode="existing_source_bundle",
                    reused_from=source_dir,
                    write_manifest=False,
                )

        osf_candidates: list[Path] = [dest_dir / "osf"]
        if pdf_dir is not None:
            osf_candidates.extend([pdf_dir / "sources" / "osf", pdf_dir / "osf"])
        for osf_dir in osf_candidates:
            files_dir = osf_dir / "files"
            source_files = _files_under(files_dir)
            if not source_files:
                continue

            target_osf_dir = dest_dir / "osf"
            target_files_dir = target_osf_dir / "files"
            if not _same_path(osf_dir, target_osf_dir):
                shutil.copytree(osf_dir, target_osf_dir, dirs_exist_ok=True)
                source_files = _files_under(target_files_dir)

            connector = self._connector_by_name("osf")
            result = ConnectorFetchResult(
                connector="osf",
                plan=_reuse_plan(json_path, plans),
                dest=str(target_osf_dir),
                fetched=[
                    FetchedSource(
                        path=str(path),
                        connector="osf",
                        source_url="local",
                        size=path.stat().st_size if path.exists() else None,
                    )
                    for path in source_files
                ],
            )
            if self.extract_text:
                self._extract_text_bundle(connector, result, dest_dir)
            manifest = self._local_reuse_manifest(
                json_path=json_path,
                pdf_path=pdf_path,
                sources_dir=dest_dir,
                plans=plans,
                files=[Path(item.path) for item in result.fetched],
                reuse_mode="legacy_osf_files",
                reused_from=osf_dir,
                result=result,
                write_manifest=True,
            )
            return manifest

        return None

    def _connector_by_name(self, name: str) -> BaseSourceConnector:
        for connector in self.connectors:
            if connector.name == name:
                return connector
        return OsfConnector()

    def _local_reuse_manifest(
        self,
        *,
        json_path: Path,
        pdf_path: Path | None,
        sources_dir: Path,
        plans: list[SourceFetchPlan],
        files: list[Path],
        reuse_mode: str,
        reused_from: Path,
        result: ConnectorFetchResult | None = None,
        write_manifest: bool,
    ) -> dict[str, Any]:
        results = [result.to_dict()] if result is not None else []
        manifest = {
            "json_path": str(json_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "sources_dir": str(sources_dir),
            "plans": [plan.to_dict() for plan in plans],
            "results": results,
            "reused_from": str(reused_from),
            "reuse_mode": reuse_mode,
            "summary": {
                "plans": len(plans),
                "connectors": ["osf"] if files else [],
                "fetched_files": len(files),
                "skipped_files": 0,
                "reused_files": len(files),
                "errors": 0,
            },
        }
        if write_manifest:
            write_json(sources_dir / "source_manifest.json", manifest)
        return manifest

    def _extract_text_bundle(
        self,
        connector: BaseSourceConnector,
        result: ConnectorFetchResult,
        sources_dir: Path,
    ) -> None:
        text_records: list[dict[str, Any]] = []
        combined_blocks: list[str] = []

        # Build the candidate file list from BOTH freshly-fetched files and files
        # that were skipped because they already existed on disk. Re-running
        # ``--fetch-sources`` on an already-downloaded paper must still regenerate
        # combined_sources.txt (e.g. after the QSF parser is upgraded) without
        # forcing a re-download.
        candidate_paths: list[str] = [item.path for item in result.fetched]
        seen_paths = set(candidate_paths)
        for skipped in result.skipped:
            if skipped not in seen_paths:
                candidate_paths.append(skipped)
                seen_paths.add(skipped)

        text_capable = set(iter_text_files(Path(p) for p in candidate_paths))

        # Sort files so the most informative come first in combined_sources.txt.
        # Priority (highest first): DOCX/QSF/pre-reg PDFs → other PDFs → text/code → data.
        def _file_priority(path_str: str) -> int:
            ext = Path(path_str).suffix.lower()
            if ext in {".docx"}:                          return 0
            if ext in {".qsf"}:                           return 1
            if "pre-regist" in path_str.lower():          return 2
            if ext == ".pdf":                             return 3
            if ext in {".txt", ".md", ".html", ".htm"}:   return 4
            if ext in {".do", ".r", ".py", ".js"}:        return 5
            return 6  # .xlsx, .sav, .csv, .tsv — data files, least informative

        sorted_paths = sorted(candidate_paths, key=_file_priority)
        path_to_orig_index = {item.path: i for i, item in enumerate(result.fetched)}
        for path_str in sorted_paths:
            file_path = Path(path_str)
            if not file_path.exists() or file_path not in text_capable:
                continue
            try:
                text = connector.extract_text(file_path)
            except Exception as exc:
                result.errors.append(f"Text extraction failed for {file_path}: {exc}")
                continue
            if not text.strip():
                continue
            rel = safe_relative_path(str(file_path.relative_to(sources_dir)))
            text_path = sources_dir / "text" / rel.with_suffix(rel.suffix + ".txt")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")
            orig_idx = path_to_orig_index.get(path_str)
            if orig_idx is not None:
                item = result.fetched[orig_idx]
                result.fetched[orig_idx] = FetchedSource(
                    path=item.path,
                    connector=item.connector,
                    source_url=item.source_url,
                    kind=item.kind,
                    size=item.size,
                    content_type=item.content_type,
                    text_path=str(text_path),
                )
            text_records.append(
                {
                    "source_path": path_str,
                    "text_path": str(text_path),
                    "connector": connector.name,
                    "chars": len(text),
                }
            )
            combined_blocks.append(f"\n--- Source: {path_str} ---\n{text}")

        if text_records:
            write_json(sources_dir / "source_text_index.json", text_records)
            (sources_dir / "combined_sources.txt").write_text("\n".join(combined_blocks), encoding="utf-8")


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except Exception:
        return Path(left) == Path(right)


def _files_under(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def _source_payload_files(sources_dir: Path) -> list[Path]:
    osf_files = _files_under(Path(sources_dir) / "osf" / "files")
    if osf_files:
        return osf_files
    combined = Path(sources_dir) / "combined_sources.txt"
    return [combined] if combined.exists() else []


def _reuse_plan(json_path: Path, plans: list[SourceFetchPlan]) -> SourceFetchPlan:
    if plans:
        return plans[0]
    return SourceFetchPlan(
        paper_folder=Path(json_path).parent.name,
        json_path=str(json_path),
        paper_title="",
        link="local",
        connector="osf",
    )


def infer_pdf_for_json(json_path: Path) -> Path | None:
    """Find a sibling PDF for a per-paper JSON."""
    pdfs = sorted(Path(json_path).parent.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def default_sources_dir_for_json(json_path: Path) -> Path:
    """Return the default source directory for a JSON file.

    Stage4 and current pipeline outputs store each paper in its own folder, so
    ``<paper>/sources`` is the intended layout. Legacy flat ``outputs`` files use
    a stem-specific directory to avoid collisions.
    """
    json_path = Path(json_path)
    stage_paper_id = json_path.stem.split("_stage", 1)[0] if "_stage" in json_path.stem else None
    if json_path.parent.name == "outputs":
        return json_path.parent / f"{json_path.stem}_sources"
    if stage_paper_id and json_path.parent.name == stage_paper_id:
        return json_path.parent / "sources"
    if "_stage1_" in json_path.name or "_stage2_" in json_path.name:
        return json_path.parent / f"{json_path.stem}_sources"
    return json_path.parent / "sources"


def paired_pdf_for_json(json_path: Path, pdf_paths: list[Path], index: int) -> Path | None:
    """Pair explicit PDFs to JSONs, falling back to a sibling PDF."""
    if len(pdf_paths) > 1:
        return pdf_paths[index] if index < len(pdf_paths) else infer_pdf_for_json(json_path)
    if len(pdf_paths) == 1:
        return pdf_paths[0]
    return infer_pdf_for_json(json_path)


def fetch_json_sources(
    json_paths: list[Path],
    *,
    pdf_paths: list[Path] | None = None,
    token: str | None = None,
    dry_run: bool = False,
    download_all: bool = True,
    extract_text: bool = True,
    output_dir: Path | None = None,
    llm_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch OSF sources for explicit JSON/PDF pairs.

    This is the path used by Stage 1 source discovery and by standalone checks
    of existing JSON files where the JSON may not live under ``stage4/<paper>/``
    yet. If the JSON lacks an OSF link, the registry still scans the paired PDF
    for explicit ``osf.io`` URLs.
    """
    connector = OsfConnector(token=token, dry_run=dry_run, download_all=download_all)
    registry = SourceConnectorRegistry(
        [connector], extract_text=extract_text and not dry_run, llm_client=llm_client
    )
    pdf_paths = [Path(path) for path in (pdf_paths or [])]
    results: list[dict[str, Any]] = []
    for index, json_path in enumerate(Path(path) for path in json_paths):
        pdf_path = paired_pdf_for_json(json_path, pdf_paths, index)
        if output_dir is None:
            dest_dir = default_sources_dir_for_json(json_path)
        else:
            out_root = Path(output_dir)
            dest_dir = out_root if len(json_paths) == 1 else out_root / json_path.stem
        results.append(registry.fetch_for_json(json_path, pdf_path=pdf_path, dest_dir=dest_dir))
    return results


def fetch_stage4_sources(
    stage4_dir: Path,
    *,
    only: list[str] | None = None,
    token: str | None = None,
    dry_run: bool = False,
    download_all: bool = True,
    extract_text: bool = True,
    llm_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch OSF sources for every matching paper JSON under stage4."""
    connector = OsfConnector(token=token, dry_run=dry_run, download_all=download_all)
    registry = SourceConnectorRegistry(
        [connector], extract_text=extract_text and not dry_run, llm_client=llm_client
    )
    results: list[dict[str, Any]] = []
    for json_path in find_paper_jsons(Path(stage4_dir)):
        if only and not any(part in json_path.as_posix() for part in only):
            continue
        results.append(
            registry.fetch_for_json(
                json_path,
                pdf_path=infer_pdf_for_json(json_path),
                dest_dir=default_sources_dir_for_json(json_path),
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OSF sources for stage4 paper JSONs")
    parser.add_argument("--stage4-dir", type=Path, default=Path("stage4"))
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--token", default=os.environ.get("OSF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-extract-text", action="store_true")
    args = parser.parse_args()

    results = fetch_stage4_sources(
        args.stage4_dir,
        only=args.only,
        token=args.token,
        dry_run=args.dry_run,
        extract_text=not args.no_extract_text,
    )
    ok = sum(1 for result in results if result["summary"]["errors"] == 0)
    for result in results:
        summary = result["summary"]
        label = Path(result["json_path"]).parent.name
        reused = summary.get("reused_files", 0)
        reused_part = f" reused={reused}" if reused else ""
        print(
            f"{label}: plans={summary['plans']} fetched={summary['fetched_files']} "
            f"errors={summary['errors']}{reused_part} sources={result['sources_dir']}"
        )
    print(f"Done: {ok}/{len(results)} paper(s) without connector errors.")
    if any(result["summary"]["errors"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
