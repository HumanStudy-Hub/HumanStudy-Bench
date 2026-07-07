"""OSF source connector.

This wraps the existing OSF crawler primitives while changing the storage
layout to the Phase-1 source backbone convention:

    stage4/<paper>/sources/osf/files/...
    stage4/<paper>/sources/osf/osf_manifest.json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from generation_pipeline.connectors.base_connector import (
    BaseSourceConnector,
    FetchedSource,
    SourceFetchPlan,
    safe_relative_path,
    write_json,
)
from generation_pipeline.utils.osf_crawler import (
    OsfClient,
    PaperOsfPlan,
    build_paper_plan,
    parse_osf_url,
    select_files,
)


class OsfConnector(BaseSourceConnector):
    name = "osf"

    def __init__(
        self,
        *,
        token: str | None = None,
        download_all: bool = True,
        dry_run: bool = False,
    ):
        self.token = token if token is not None else os.environ.get("OSF_TOKEN")
        self.download_all = download_all
        self.dry_run = dry_run

    def detect(self, link: str) -> bool:
        return "osf.io" in (link or "").lower()

    def fetch(self, plan: SourceFetchPlan, dest: Path) -> list[FetchedSource]:
        osf_plan = self._to_osf_plan(plan)
        if not osf_plan.can_download:
            return []

        dest = Path(dest)
        files_dir = dest / "files"
        client = OsfClient(token=self.token, view_only=osf_plan.view_only)

        entity_type, title, files_index = client.resolve_entity(osf_plan.osf_id)
        osf_plan.entity_type = entity_type
        osf_plan.entity_title = title

        providers = client._request(files_index).get("data", [])
        all_files = []
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

        matched = select_files(all_files, osf_plan.effects_needing_osf, download_all=self.download_all)
        osf_plan.matched_files = matched
        if not matched:
            reason = (
                "OSF project has no files in storage"
                if not all_files
                else "No OSF files matched selection criteria"
            )
            osf_plan.errors.append(reason)
            write_json(
                dest / "osf_manifest.json",
                {
                    "plan": osf_plan.to_dict(),
                    "all_osf_files": [item.to_dict() for item in all_files],
                    "source_layout": "sources/osf/files",
                    "dry_run": self.dry_run,
                },
            )
            raise RuntimeError(reason)

        fetched: list[FetchedSource] = []
        skipped: list[str] = []
        for item in matched:
            if not item.download_url:
                continue
            rel = safe_relative_path(item.path)
            out_path = files_dir / rel
            if self.dry_run:
                # Dry-run: record intent but do NOT add to fetched (file not on disk).
                osf_plan.skipped_files.append(str(out_path))
                skipped.append(str(out_path))
                continue
            if out_path.exists():
                # Already downloaded on a previous run — skip network round-trip.
                osf_plan.skipped_files.append(str(out_path))
            else:
                try:
                    client.download(item.download_url, out_path)
                    osf_plan.downloaded_files.append(str(out_path))
                except Exception as exc:
                    osf_plan.errors.append(f"Failed to download {item.path}: {exc}")
                    continue
            fetched.append(
                FetchedSource(
                    path=str(out_path),
                    connector=self.name,
                    source_url=item.download_url,
                    size=item.size,
                )
            )

        manifest = {
            "plan": osf_plan.to_dict(),
            "all_osf_files": [item.to_dict() for item in all_files],
            "source_layout": "sources/osf/files",
            "dry_run": self.dry_run,
            "skipped_existing": [s for s in osf_plan.skipped_files if not self.dry_run],
        }
        write_json(dest / "osf_manifest.json", manifest)
        return fetched

    def _to_osf_plan(self, plan: SourceFetchPlan) -> PaperOsfPlan:
        if plan.json_path:
            osf_plan = build_paper_plan(Path(plan.json_path))
            if osf_plan.osf_id:
                return osf_plan

        osf_id, view_only = parse_osf_url(plan.link)
        return PaperOsfPlan(
            paper_folder=plan.paper_folder,
            json_path=plan.json_path or "",
            paper_title=plan.paper_title,
            osf_url=plan.link,
            osf_id=osf_id,
            view_only=view_only,
        )
