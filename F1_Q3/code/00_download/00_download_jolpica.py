#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs, load_config, sha256_file


def request_json(url: str, user_agent: str, retries: int = 6) -> tuple[bytes, dict]:
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    for attempt in range(retries):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            delay = max(float(exc.headers.get("Retry-After", 0) or 0), 2 ** attempt)
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def download_endpoint(season: int, endpoint: str, config: dict, force: bool) -> list[dict]:
    raw_dir = ROOT / "data" / "raw" / "jolpica"
    limit = int(config["page_limit"])
    offset = 0
    manifest_rows = []
    while True:
        filename = f"{endpoint}_{season}_offset_{offset:04d}.json"
        path = raw_dir / filename
        url = f"{config['api_base']}/{season}/{endpoint}/?limit={limit}&offset={offset}"
        retrieved_at = None
        response_headers = {}
        if path.exists() and not force:
            payload = path.read_bytes()
        else:
            payload, response_headers = request_json(url, config["user_agent"])
            parsed = json.loads(payload.decode("utf-8"))
            if "MRData" not in parsed:
                raise ValueError(f"Unexpected API response for {url}")
            tmp = path.with_suffix(".json.tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
            retrieved_at = datetime.now(timezone.utc).isoformat()
            time.sleep(float(config["request_pause_seconds"]))
        parsed = json.loads(payload.decode("utf-8"))
        total = int(parsed["MRData"].get("total", 0))
        returned_limit = int(parsed["MRData"].get("limit", limit))
        manifest_rows.append({
            "season": season,
            "endpoint": endpoint,
            "offset": offset,
            "limit": returned_limit,
            "total": total,
            "url": url,
            "file": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "retrieved_at_utc": retrieved_at,
            "etag": response_headers.get("ETag", ""),
            "last_modified": response_headers.get("Last-Modified", ""),
        })
        offset += returned_limit
        if offset >= total or returned_limit <= 0:
            break
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refresh and overwrite frozen raw JSON.")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    rows = []
    for season in config["qualifying_seasons"]:
        rows.extend(download_endpoint(int(season), "qualifying", config, args.force))
    for season in config["results_seasons"]:
        rows.extend(download_endpoint(int(season), "results", config, args.force))
    manifest_path = ROOT / "data" / "raw" / "jolpica" / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log = {
        "pages": len(rows),
        "qualifying_seasons": config["qualifying_seasons"],
        "results_seasons": config["results_seasons"],
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    (ROOT / "output" / "logs" / "00_download.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
