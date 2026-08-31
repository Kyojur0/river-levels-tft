"""Collect live EA flood-monitoring increments for configured stations.

The archive extractor remains the training-history source. This module is a
small, serial collector for ``latest`` snapshots or station ``since`` catch-up
requests. It never infers a live station ID from a hydrology GUID: the
crosswalk in ``configs/stations.yml`` is required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse


LIVE_ATTRIBUTION = (
    "this uses Environment Agency flood and river level data from the "
    "real-time data API (Beta)"
)
OGL_ATTRIBUTION = "Contains public sector information licensed under the Open Government Licence v3.0."
DEFAULT_BASE_URL = "https://environment.data.gov.uk/flood-monitoring"


class LiveCollectorError(RuntimeError):
    """Raised for invalid crosswalks or live API responses."""


def _require_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LiveCollectorError("live collection requires requests; install environment.yml") from exc
    return requests


def _load_station_config(path: Path | str) -> tuple[str, list[dict[str, Any]]]:
    try:
        from .extract import load_config, station_specs
    except ImportError:  # pragma: no cover - direct script support
        from src.extract import load_config, station_specs  # type: ignore[no-redef]
    config = load_config(path)
    specs = station_specs(config)
    base_url = str(config.get("live_base_url") or DEFAULT_BASE_URL).rstrip("/")
    if urlparse(base_url).scheme not in {"http", "https"}:
        raise LiveCollectorError("live_base_url must be an http(s) URL")
    records = []
    for spec in specs:
        crosswalk = dict(spec.flood_monitoring)
        same_as = str(crosswalk.get("same_as") or "").strip()
        station_reference = str(crosswalk.get("station_reference") or "").strip()
        if not station_reference and same_as:
            station_reference = same_as.rstrip("/").rsplit("/", 1)[-1]
        if not station_reference:
            raise LiveCollectorError(
                f"{spec.station_guid} has no explicit flood_monitoring station_reference/same_as"
            )
        records.append(
            {
                "station_guid": spec.station_guid,
                "name": spec.name,
                "river": spec.river,
                "measure_id": spec.measure_id,
                "station_reference": station_reference,
                "same_as": same_as,
            }
        )
    return base_url, records


def _request_json(
    session: Any,
    url: str,
    *,
    params: Mapping[str, Any],
    timeout: tuple[int, int] = (20, 60),
    retries: int = 3,
    sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=dict(params), timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else sleep_seconds * (2**attempt)
                except (TypeError, ValueError):
                    delay = sleep_seconds * (2**attempt)
                time.sleep(min(delay, 30.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise LiveCollectorError(f"live response lacks an items list: {response.url}")
            return payload
        except Exception as exc:  # requests errors and malformed JSON
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(sleep_seconds * (2**attempt), 30.0))
    raise LiveCollectorError(f"GET {url} failed: {last_error}")


def _is_level_reading(item: Mapping[str, Any]) -> bool:
    measure = item.get("measure")
    if not isinstance(measure, Mapping):
        return False
    return str(measure.get("parameter") or "").lower() in {"level", "waterlevel", "water level"}


def collect_live(
    *,
    config_path: Path | str = Path("configs/stations.yml"),
    station_guids: Sequence[str] | None = None,
    since: str | None = None,
    latest: bool = False,
    page_limit: int = 1000,
    sleep_seconds: float = 1.0,
    session: Any | None = None,
) -> dict[str, Any]:
    """Collect level readings and return a provenance-bearing payload."""

    if not latest and not since:
        latest = True
    if latest and since:
        raise LiveCollectorError("choose either latest or since, not both")
    if page_limit <= 0 or sleep_seconds < 0:
        raise LiveCollectorError("page_limit must be positive and sleep_seconds non-negative")
    requests = _require_requests()
    base_url, records = _load_station_config(config_path)
    selected = set(station_guids or [])
    if selected:
        unknown = selected - {str(record["station_guid"]) for record in records}
        if unknown:
            raise LiveCollectorError(f"station GUID(s) not present in config: {sorted(unknown)}")
        records = [record for record in records if record["station_guid"] in selected]
    client = session or requests.Session()
    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output: list[dict[str, Any]] = []
    request_log: list[str] = []
    for index, record in enumerate(records):
        station_reference = quote(str(record["station_reference"]), safe="")
        url = f"{base_url}/id/stations/{station_reference}/readings"
        offset = 0
        seen_page_keys: set[tuple[str, str, str]] = set()
        while True:
            params: dict[str, Any] = {"_view": "full", "_limit": page_limit, "_offset": offset}
            if latest:
                params["latest"] = ""
            else:
                params.update({"since": since, "_sorted": ""})
            payload = _request_json(client, url, params=params)
            request_url = str(payload.get("meta", {}).get("requestedUrl") or "")
            if not request_url:
                request_url = str(requests.Request("GET", url, params=params).prepare().url)
            request_log.append(request_url)
            items = payload["items"]
            raw_effective_limit = payload.get("meta", {}).get("limit")
            try:
                effective_limit = int(raw_effective_limit)
            except (TypeError, ValueError):
                effective_limit = page_limit
            if effective_limit <= 0:
                effective_limit = page_limit
            page_keys = {
                (
                    str(item.get("@id") or ""),
                    str(item.get("dateTime") or ""),
                    str(item.get("value") or ""),
                )
                for item in items
                if isinstance(item, Mapping)
            }
            if offset and page_keys and page_keys.issubset(seen_page_keys):
                raise LiveCollectorError(
                    f"live pagination repeated a page for station {record['station_reference']}"
                )
            seen_page_keys.update(page_keys)
            for item in items:
                if not isinstance(item, Mapping) or not _is_level_reading(item):
                    continue
                row = dict(item)
                row.update(
                    {
                        "collector_station_guid": record["station_guid"],
                        "collector_station_name": record["name"],
                        "collector_river": record["river"],
                        "collector_hydrology_measure_id": record["measure_id"],
                        "collector_live_station_reference": record["station_reference"],
                        "collector_retrieved_at_utc": retrieved,
                    }
                )
                output.append(row)
            if len(items) < effective_limit:
                break
            next_offset = offset + len(items)
            if next_offset <= offset:
                raise LiveCollectorError("live pagination did not advance its offset")
            offset = next_offset
            if sleep_seconds:
                time.sleep(sleep_seconds)
        if index + 1 < len(records) and sleep_seconds:
            time.sleep(sleep_seconds)
    output.sort(key=lambda row: (str(row.get("collector_station_guid")), str(row.get("dateTime", ""))))
    return {
        "schema": "ea-flood-monitoring-live-v1",
        "retrieved_at_utc": retrieved,
        "mode": "latest" if latest else "since",
        "since": since,
        "base_url": base_url,
        "request_urls": request_log,
        "station_count": len(records),
        "reading_count": len(output),
        "items": output,
        "attribution": LIVE_ATTRIBUTION,
        "ogl_attribution": OGL_ATTRIBUTION,
        "notes": [
            "Hydrology archive remains the bulk-history source; this payload is an incremental live feed.",
            "Cross-API identity is retained explicitly from configs/stations.yml.",
            "This collector is not an operational warning service.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stations.yml"))
    parser.add_argument("--station", action="append", dest="station_guids", metavar="GUID")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--latest", action="store_true")
    mode.add_argument("--since")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = collect_live(
            config_path=args.config,
            station_guids=args.station_guids,
            since=args.since,
            latest=args.latest or not args.since,
            page_limit=args.limit,
            sleep_seconds=args.sleep_seconds,
        )
        rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (LiveCollectorError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
