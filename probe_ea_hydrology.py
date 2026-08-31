#!/usr/bin/env python3
"""Probe Environment Agency hydrology archive coverage and quality.

The probe deliberately makes one request at a time and writes nothing to disk.
It uses the station's live metadata to select a qualified 15-minute level
measure, then checks the requested date window for timestamp and value gaps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import quote, urlencode


BASE_URL = "https://environment.data.gov.uk/hydrology"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-01-01"
MAX_LIMIT = 2_000_000

DEFAULT_STATIONS = [
    ("8496ce69-482c-406a-a2f0-ac418ef8f099", "Kingston", "River Thames"),
    ("3c4d4f78-2d0e-474a-b884-65a9daca18fb", "Thorverton", "River Exe"),
    ("8820d897-a09e-4857-8095-5834fee6962f", "Bewdley", "River Severn"),
    ("0dcf81cb-5305-4e0b-b150-9b733ac44d0b", "Colwick", "River Trent"),
    ("213d70b2-894b-406b-9dc3-31d3ccec7f54", "Skelton", "River Ouse"),
    ("9ad5d28c-7cfe-46db-b39d-58701689cd59", "Caton", "River Lune"),
    ("e786e60f-a0f1-4955-aa57-f22ba39c7427", "Bywell", "River Tyne"),
    ("30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9", "Hereford Bridge", "River Wye"),
    ("3c43b72d-03a7-46e1-86c7-76dd97808544", "Roxton", "River Great Ouse"),
]


class ProbeError(RuntimeError):
    """An archive response could not be used by the probe."""


def fetch_json(url: str, timeout: int) -> dict[str, Any]:
    """Fetch JSON through curl so redirects and the platform CA store work."""

    try:
        completed = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--max-time",
                str(timeout),
                "--header",
                "Accept: application/json",
                url,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProbeError("curl is required but was not found on PATH") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"curl exit {completed.returncode}"
        raise ProbeError(f"GET failed: {url}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"GET returned non-JSON content: {url}") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"GET returned an unexpected JSON shape: {url}")
    return payload


def items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("items", [])
    if not isinstance(value, list):
        raise ProbeError("API response has a non-list items field")
    return [item for item in value if isinstance(item, dict)]


def first_item(payload: dict[str, Any], what: str) -> dict[str, Any]:
    values = items(payload)
    if not values:
        raise ProbeError(f"No {what} returned")
    return values[0]


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_datetime(value: str) -> datetime:
    """Parse API timestamps and compare them on a naive UTC-like axis.

    The current archive responses expose naive ISO timestamps. If an offset is
    supplied, it is converted to UTC before the timezone marker is removed.
    The report calls out the archive's timestamp convention separately.
    """

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def measure_token(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.rsplit("/", 1)[-1]


def choose_level_measure(station: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for measure in station.get("measures", []):
        if not isinstance(measure, dict):
            continue
        token = measure_token(
            measure.get("@id") or measure.get("id") or measure.get("notation")
        )
        if not token:
            continue
        parameter = str(measure.get("parameter") or "").lower()
        observed_property = str(measure.get("observedProperty") or "").lower()
        period_name = str(measure.get("periodName") or "").lower()
        period = measure.get("period")
        is_level = parameter in {"level", "waterlevel", "water level"}
        is_level = is_level or observed_property.endswith("waterlevel")
        is_15_min = period == 900 or period_name in {"15min", "15 min", "15-minute"}
        if not (is_level and is_15_min):
            continue
        observation_type = str(measure.get("observationType") or "").lower()
        qualified = observation_type == "qualified" or "-qualified" in token.lower()
        # Prefer Qualified 15-minute level, then any other 15-minute level.
        priority = 0 if qualified else 1
        candidates.append((priority, token, measure))
    if not candidates:
        raise ProbeError("station has no 15-minute level measure")
    _, token, metadata = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return token, metadata


def is_missing_value(row: dict[str, Any]) -> bool:
    value = row.get("value")
    return value is None or value == ""


def format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "(none)"


def concept_text(value: Any) -> str:
    """Return a concise label from an EA concept object or concept list."""

    if isinstance(value, list):
        labels = [concept_text(item) for item in value]
        return ", ".join(label for label in labels if label)
    if isinstance(value, dict):
        return str(value.get("label") or value.get("notation") or value.get("@id") or "")
    return str(value or "")


def fetch_window(
    measure_url: str,
    start: date,
    end: date,
    limit: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, bool]:
    """Read a window page by page, stopping when a short page is returned."""

    rows: list[dict[str, Any]] = []
    first_meta: dict[str, Any] = {}
    offset = 0
    page_count = 0
    for _ in range(100):
        query = urlencode(
            {
                "mineq-date": start.isoformat(),
                "max-date": end.isoformat(),
                "_limit": str(limit),
                "_offset": str(offset),
                "_view": "full",
            }
        )
        payload = fetch_json(f"{measure_url}?{query}", timeout)
        page = items(payload)
        page_count += 1
        if not first_meta and isinstance(payload.get("meta"), dict):
            first_meta = payload["meta"]
        rows.extend(page)
        if len(page) < limit:
            return rows, first_meta, page_count, False
        offset += len(page)
    # A very long window should use the asynchronous batch API instead.
    return rows, first_meta, page_count, True


@dataclass
class StationResult:
    guid: str
    label: str
    river: str
    status: str
    measure: str
    period: int
    period_name: str
    unit_name: str
    observation_type: str
    qualifier: str
    has_telemetry: Any
    datum: Any
    datum_type: Any
    earliest: dict[str, Any]
    latest: dict[str, Any]
    expected_slots: int
    rows: int
    unique_timestamps: int
    duplicates: int
    missing_slots: int
    timestamp_coverage_pct: float
    off_grid_unique: int
    first_in_window: str | None
    last_in_window: str | None
    mode_delta_seconds: int | None
    max_gap_hours: float | None
    missing_value_rows: int
    missing_value_timestamps: int
    value_coverage_pct: float
    quality: Counter[str]
    qcode: Counter[str]
    completeness: Counter[str]
    api_limit: Any
    pages_fetched: int
    truncated: bool


def probe_station(
    guid: str,
    fallback_label: str,
    fallback_river: str,
    start: date,
    end: date,
    limit: int,
    timeout: int,
    base_url: str,
) -> StationResult:
    station_url = f"{base_url.rstrip('/')}/id/stations/{quote(guid, safe='')}.json"
    station_payload = fetch_json(station_url, timeout)
    station = first_item(station_payload, "station metadata")
    measure, measure_metadata = choose_level_measure(station)

    measure_url = f"{base_url.rstrip('/')}/id/measures/{quote(measure, safe='')}/readings.json"
    earliest = first_item(fetch_json(f"{measure_url}?earliest", timeout), "earliest reading")
    latest = first_item(fetch_json(f"{measure_url}?latest", timeout), "latest reading")
    raw_rows, window_meta, pages_fetched, pagination_truncated = fetch_window(
        measure_url, start, end, limit, timeout
    )

    period = measure_metadata.get("period")
    if not isinstance(period, int) or period <= 0:
        period = 900
    period_name = str(measure_metadata.get("periodName") or f"{period}s")
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.min.time())
    in_window: list[tuple[dict[str, Any], datetime]] = []
    no_datetime_rows = 0
    for row in raw_rows:
        text_value = row.get("dateTime")
        if not isinstance(text_value, str):
            no_datetime_rows += 1
            continue
        try:
            parsed = parse_datetime(text_value)
        except ValueError:
            no_datetime_rows += 1
            continue
        if start_dt <= parsed < end_dt:
            in_window.append((row, parsed))

    timestamps = [parsed for _, parsed in in_window]
    timestamp_set = set(timestamps)
    expected_slots = max(0, int((end_dt - start_dt).total_seconds() // period))
    expected = {
        start_dt + timedelta(seconds=period * index) for index in range(expected_slots)
    }
    deltas = [
        int((right - left).total_seconds())
        for left, right in zip(sorted(timestamp_set), sorted(timestamp_set)[1:])
    ]
    delta_counts = Counter(deltas)
    mode_delta = delta_counts.most_common(1)[0][0] if delta_counts else None
    max_gap_hours = max(deltas) / 3600 if deltas else None
    missing_value_rows = sum(is_missing_value(row) for row, _ in in_window)
    missing_value_timestamps = {
        parsed for row, parsed in in_window if is_missing_value(row)
    }
    usable_timestamps = {
        parsed for row, parsed in in_window if not is_missing_value(row)
    }
    api_limit = window_meta.get("limit") if isinstance(window_meta, dict) else None

    return StationResult(
        guid=guid,
        label=str(station.get("label") or fallback_label),
        river=str(station.get("riverName") or fallback_river),
        status=concept_text(station.get("status")),
        measure=measure,
        period=period,
        period_name=period_name,
        unit_name=str(measure_metadata.get("unitName") or measure_metadata.get("unit") or ""),
        observation_type=concept_text(measure_metadata.get("observationType")),
        qualifier=concept_text(measure_metadata.get("qualifier")),
        has_telemetry=measure_metadata.get("hasTelemetry"),
        datum=measure_metadata.get("datum") or station.get("datum"),
        datum_type=measure_metadata.get("datumType") or station.get("datumType"),
        earliest=earliest,
        latest=latest,
        expected_slots=expected_slots,
        rows=len(in_window),
        unique_timestamps=len(timestamp_set),
        duplicates=len(timestamps) - len(timestamp_set),
        missing_slots=len(expected - timestamp_set),
        timestamp_coverage_pct=(100 * len(timestamp_set & expected) / expected_slots)
        if expected_slots
        else 0.0,
        off_grid_unique=len(timestamp_set - expected),
        first_in_window=min(timestamps).isoformat() if timestamps else None,
        last_in_window=max(timestamps).isoformat() if timestamps else None,
        mode_delta_seconds=mode_delta,
        max_gap_hours=max_gap_hours,
        missing_value_rows=missing_value_rows,
        missing_value_timestamps=len(missing_value_timestamps),
        value_coverage_pct=(100 * len(usable_timestamps & expected) / expected_slots)
        if expected_slots
        else 0.0,
        quality=Counter(str(row.get("quality") or "(blank)") for row, _ in in_window),
        qcode=Counter(
            str(row.get("qcode") or row.get("qflag") or "(blank)")
            for row, _ in in_window
        ),
        completeness=Counter(
            str(row.get("completeness") or "(blank)") for row, _ in in_window
        ),
        api_limit=api_limit,
        pages_fetched=pages_fetched,
        truncated=pagination_truncated or no_datetime_rows > 0,
    )


def history_years(earliest: dict[str, Any], latest: dict[str, Any]) -> float | None:
    first = earliest.get("dateTime")
    last = latest.get("dateTime")
    if not isinstance(first, str) or not isinstance(last, str):
        return None
    try:
        seconds = (parse_datetime(last) - parse_datetime(first)).total_seconds()
    except ValueError:
        return None
    return seconds / (365.2425 * 24 * 3600)


def print_results(results: Iterable[StationResult], start: date, end: date) -> None:
    result_list = list(results)
    expected = result_list[0].expected_slots if result_list else 0
    print("EA Hydrology archive probe")
    print(f"Window: [{start.isoformat()}, {end.isoformat()})")
    print(f"Expected slots per station: {expected} at the selected measure period")
    print(
        "station\triver\tmeasure\tearliest\thistory_years\trows\tunique\tduplicates\t"
        "timestamp_gap_pct\toffgrid\tmissing_value_pct\tvalue_coverage_pct\tmode_step_s\tmax_gap_h"
    )
    for result in result_list:
        years = history_years(result.earliest, result.latest)
        missing_pct = 100 * result.missing_value_timestamps / result.expected_slots if result.expected_slots else 0
        timestamp_gap_pct = 100 - result.timestamp_coverage_pct
        print(
            "\t".join(
                [
                    result.label,
                    result.river,
                    result.measure,
                    str(result.earliest.get("dateTime") or ""),
                    f"{years:.2f}" if years is not None else "",
                    str(result.rows),
                    str(result.unique_timestamps),
                    str(result.duplicates),
                    f"{timestamp_gap_pct:.3f}",
                    str(result.off_grid_unique),
                    f"{missing_pct:.3f}",
                    f"{result.value_coverage_pct:.3f}",
                    str(result.mode_delta_seconds or ""),
                    f"{result.max_gap_hours:.2f}" if result.max_gap_hours is not None else "",
                ]
            )
        )
        print(
            f"  {result.guid} | status={result.status} | period={result.period_name} "
            f"| unit={result.unit_name} | pages={result.pages_fetched}"
        )
        print(
            "  "
            f"quality: {format_counts(result.quality)} | "
            f"qcode: {format_counts(result.qcode)} | "
            f"completeness: {format_counts(result.completeness)}"
        )
        print(
            "  "
            f"observationType={result.observation_type or '(blank)'} "
            f"hasTelemetry={result.has_telemetry!r} "
            f"datum={result.datum!r} datumType={result.datum_type!r}"
        )
        if result.truncated:
            print("  WARNING: response may be truncated or contained rows without parseable dateTime")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START, type=parse_date)
    parser.add_argument("--end-date", default=DEFAULT_END, type=parse_date)
    parser.add_argument(
        "--station",
        action="append",
        dest="stations",
        metavar="GUID",
        help="station GUID; repeat to override the nine default stations",
    )
    parser.add_argument("--limit", type=int, default=100_000, help="archive _limit (1..2,000,000)")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per HTTP request")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="pause between stations to keep requests serial and polite",
    )
    parser.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.end_date <= args.start_date:
        parser.error("--end-date must be after --start-date")
    if not 1 <= args.limit <= MAX_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_LIMIT}")
    if args.timeout <= 0 or args.sleep_seconds < 0:
        parser.error("--timeout must be positive and --sleep-seconds cannot be negative")

    by_guid = {guid: (guid, label, river) for guid, label, river in DEFAULT_STATIONS}
    if args.stations:
        station_specs = [
            by_guid.get(guid, (guid, guid, "(river from API metadata)")) for guid in args.stations
        ]
    else:
        station_specs = DEFAULT_STATIONS

    results: list[StationResult] = []
    failures = 0
    for index, (guid, label, river) in enumerate(station_specs):
        try:
            result = probe_station(
                guid,
                label,
                river,
                args.start_date,
                args.end_date,
                args.limit,
                args.timeout,
                args.base_url,
            )
        except ProbeError as exc:
            failures += 1
            print(f"ERROR\t{guid}\t{exc}", file=sys.stderr)
        else:
            results.append(result)
        if index + 1 < len(station_specs):
            time.sleep(args.sleep_seconds)

    if results:
        print_results(results, args.start_date, args.end_date)
    if failures:
        print(f"{failures} station(s) failed", file=sys.stderr)
    return 1 if failures or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
