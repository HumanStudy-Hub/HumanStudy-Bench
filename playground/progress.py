"""Publish playground run progress so the web app can show it while the run works.

The run writes `progress.json` next to its other files and, when a token is
available, mirrors the same payload to the private jobs repository through the
GitHub contents API. The web app polls that file.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ProgressWriter:
    def __init__(self, run_dir: Path, repo: Optional[str] = None, branch: Optional[str] = None, path: Optional[str] = None, min_interval: float = 10.0) -> None:
        self.local_path = run_dir / "progress.json"
        self.token = os.environ.get("PIPELINE_PROGRESS_TOKEN", "")
        self.repo = repo
        self.branch = branch
        self.path = path
        self.min_interval = min_interval
        self.sha: Optional[str] = None
        self._last_publish = 0.0
        if self._remote_enabled():
            self._load_sha()

    def _remote_enabled(self) -> bool:
        return bool(self.token and self.repo and self.branch and self.path)

    def _request(self, method: str, body: Optional[dict] = None) -> dict:
        url = f"https://api.github.com/repos/{self.repo}/contents/{urllib.parse.quote(self.path or '', safe='/')}"
        if method == "GET":
            url += "?ref=" + urllib.parse.quote(self.branch or "", safe="")
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "HumanStudy-Hub-Playground",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())

    def _load_sha(self) -> None:
        try:
            self.sha = self._request("GET").get("sha")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                print(f"[progress] could not read remote progress: {exc}", flush=True)
        except Exception as exc:
            print(f"[progress] could not read remote progress: {exc}", flush=True)

    def write(self, payload: Dict[str, Any], force: bool = False) -> None:
        payload = {**payload, "updatedAt": datetime.now(timezone.utc).isoformat()}
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(json.dumps(payload, indent=2) + "\n")
        if not self._remote_enabled():
            return
        # Publishing is a commit per call, so a long run only mirrors on an interval.
        if not force and time.monotonic() - self._last_publish < self.min_interval:
            return
        body = {
            "message": f"playground: progress {payload.get('phase', 'running')}",
            "branch": self.branch,
            "content": base64.b64encode((json.dumps(payload, indent=2) + "\n").encode()).decode(),
        }
        if self.sha:
            body["sha"] = self.sha
        try:
            response = self._request("PUT", body)
            self.sha = response["content"]["sha"]
            self._last_publish = time.monotonic()
        except Exception as exc:
            print(f"[progress] could not publish progress: {exc}", flush=True)
