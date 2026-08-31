"""Serial extraction of raw Environment Agency hydrology readings.

This module is deliberately independent of pandas and PyTorch. It reads the
explicit station and measure IDs in configs/stations.yml, requests the archive
one page at a time, and can stream the original reading objects to JSONL, JSON,
or CSV. It never invents a flood-monitoring crosswalk from a hydrology GUID.

The default window is the two-year, half-open interval used by the live probe:
2024-01-01 <= dateTime < 2026-01-01. src.freeze builds on the public functions
here to add a manifest and checksums.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:  # Requests is listed in environment.yml, but the module has a stdlib fallback.
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal environments
    requests = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:  # pragma: no cover - clear error is preferable to import failure
    yaml = None  # type: ignore[assignment]


BASE_URL = "https://environment.data.gov.uk/hydrology"
DEFAULT_CONFIG = Path("configs/stations.yml")
DEFAULT_OUTPUT_DIR = Path("data/frozen")
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-01-01"
SOFT_LIMIT = 100_000
HARD_LIMIT = 2_000_000
DEFAULT_PAGE_LIMIT = SOFT_LIMIT
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 3
DEFAULT_SLEEP_SECONDS = 0.25
DEFAULT_MAX_PAGES = 10_000
OUTPUT_FORMATS = ("jsonl", "csv", "json")

# These are the fields present in live archive reading responses. Unknown
# fields are retained in the CSV extra_json column and remain untouched in
# JSON/JSONL output.
CSV_FIELDS = (
    "measure",
    "dateTime",
    "date",
    "value",
    "completeness",
    "quality",
    "qcode",
    "qflag",
    "valid",
    "invalid",
    "missing",
)


class ExtractionError(RuntimeError):
    """Raised when a station cannot be extracted reproducibly."""


class ConfigError(ExtractionError):
    """Raised for an invalid or ambiguous station configuration."""


class PaginationError(ExtractionError):
    """Raised when a response cannot be safely paged within archive limits."""


def parse_date(value: str) -> date:
    """Parse a command-line/config date as YYYY-MM-DD."""

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_datetime(value: str) -> datetime:
    """Parse an EA timestamp on an explicit UTC-like comparison axis.

    Current hydrology responses expose naive ISO timestamps. If a future
    response includes Z or an offset, it is converted to UTC and the timezone
    marker is removed for comparison with the configured date window. The
    original value is never changed in output files.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("dateTime is empty")
    normalised = value.strip()
    if normalised.endswith("Z"):
        normalised = normalised[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalised)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def iso_utc(value: datetime | None = None) -> str:
    """Return a stable UTC timestamp for provenance."""

    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the entire extract into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert common API values to JSON-safe values for manifests/CSV."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _text(value: Any) -> str:
    """Render an API scalar/concept as a concise stable string."""

    if isinstance(value, Mapping):
        return str(value.get("label") or value.get("notation") or value.get("@id") or "")
    if isinstance(value, (list, tuple)):
        return ", ".join(filter(None, (_text(item) for item in value)))
    return "" if value is None else str(value)


def _token(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.rsplit("/", 1)[-1]


def _response_items(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("items", [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExtractionError("EA response has a non-list 'items' field")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _first_item(payload: Mapping[str, Any], what: str) -> dict[str, Any]:
    values = _response_items(payload)
    if not values:
        raise ExtractionError(f"EA response contained no {what}")
    return values[0]


def _with_params(url: str, params: Mapping[str, Any] | None = None) -> str:
    if not params:
        return url
    split = urlsplit(url)
    existing = split.query
    query = urlencode(
        [(key, value) for key, value in params.items() if value is not None],
        doseq=True,
    )
    combined = "&".join(part for part in (existing, query) if part)
    return urlunsplit((split.scheme, split.netloc, split.path, combined, split.fragment))


class ArchiveClient:
    """Small serial HTTP client with bounded retries and URL provenance.

    A requests.Session is used when available because it handles the
    workstation certificate store consistently. The urllib path keeps
    minimal installations dependency-light. No credentials are sent: the
    public API is unauthenticated.
    """

    RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        user_agent: str = "river-levels-tft/phase2 (research; serial archive extract)",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds cannot be negative")
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.sleep = sleep
        self.user_agent = user_agent
        self.session = session
        if self.session is None and requests is not None:
            self.session = requests.Session()

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """GET one JSON response and return payload plus request URL."""

        request_url = _with_params(url, params)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.session is not None and requests is not None:
                    response = self.session.get(
                        request_url,
                        headers={
                            "Accept": "application/json",
                            "User-Agent": self.user_agent,
                        },
                        timeout=self.timeout,
                    )
                    status = int(response.status_code)
                    body = response.text
                    if status >= 400:
                        if status in self.RETRY_STATUS and attempt < self.retries:
                            self._wait_for_retry(response, attempt)
                            continue
                        raise ExtractionError(
                            f"GET {request_url} returned HTTP {status}: {body[:300]}"
                        )
                else:
                    request = Request(
                        request_url,
                        headers={
                            "Accept": "application/json",
                            "User-Agent": self.user_agent,
                        },
                    )
                    with urlopen(request, timeout=self.timeout) as response:  # nosec B310
                        status = int(getattr(response, "status", 200))
                        body = response.read().decode("utf-8")
                        if status >= 400:
                            raise ExtractionError(
                                f"GET {request_url} returned HTTP {status}: {body[:300]}"
                            )
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise ExtractionError(
                        f"GET {request_url} returned non-JSON content"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise ExtractionError(
                        f"GET {request_url} returned an unexpected JSON shape"
                    )
                return dict(payload), request_url
            except ExtractionError:
                raise
            except HTTPError as exc:
                last_error = exc
                status = int(exc.code)
                if status in self.RETRY_STATUS and attempt < self.retries:
                    self._wait_for_retry(exc, attempt)
                    continue
                raise ExtractionError(f"GET {request_url} failed: {exc}") from exc
            except (URLError, OSError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    self._wait_for_retry(None, attempt)
                    continue
                raise ExtractionError(f"GET {request_url} failed: {exc}") from exc
        raise ExtractionError(f"GET {request_url} failed: {last_error}")

    def _wait_for_retry(self, response: Any, attempt: int) -> None:
        retry_after: float | None = None
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get("Retry-After")
            try:
                retry_after = float(value) if value is not None else None
            except (TypeError, ValueError):
                retry_after = None
        delay = (
            retry_after
            if retry_after is not None
            else min(30.0, self.sleep_seconds * (2.0**attempt))
        )
        if delay:
            self.sleep(delay)


@dataclass(frozen=True)
class StationSpec:
    """Explicit station/measure configuration used by an extraction."""

    station_guid: str
    name: str
    river: str
    role: str
    measure_id: str
    flood_monitoring: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StationSpec":
        if not isinstance(raw, Mapping):
            raise ConfigError("each stations.yml entry must be a mapping")
        station_guid = str(raw.get("station_guid") or "").strip()
        name = str(raw.get("name") or "").strip()
        river = str(raw.get("river") or "").strip()
        measure_id = str(raw.get("measure_id") or "").strip()
        if not station_guid:
            raise ConfigError("station entry is missing station_guid")
        if not measure_id:
            raise ConfigError(
                f"{station_guid} is missing explicit measure_id; "
                "the extractor will not infer one from a GUID"
            )
        if not name:
            name = station_guid
        if not river:
            river = "(river not configured)"
        crosswalk = raw.get("flood_monitoring") or {}
        if not isinstance(crosswalk, Mapping):
            raise ConfigError(f"{station_guid}.flood_monitoring must be a mapping")
        return cls(
            station_guid=station_guid,
            name=name,
            river=river,
            role=str(raw.get("role") or ""),
            measure_id=measure_id,
            flood_monitoring=dict(crosswalk),
        )


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a stations YAML/JSON mapping without changing it on disk."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"configuration file not found: {config_path}")
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc
    if config_path.suffix.lower() == ".json":
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON configuration {config_path}") from exc
    else:
        if yaml is None:
            loaded = _simple_yaml_config(raw_text)
        else:
            try:
                loaded = yaml.safe_load(raw_text)
            except Exception as exc:
                raise ConfigError(f"invalid YAML configuration {config_path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return dict(loaded)


def _simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _simple_yaml_config(text: str) -> dict[str, Any]:
    """Parse the scalar/list subset needed by the project station catalogue."""

    result: dict[str, Any] = {}
    stations: list[dict[str, Any]] = []
    in_stations = False
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "stations:":
            in_stations = True
            result["stations"] = stations
            continue
        if in_stations and stripped.startswith("- "):
            current = {}
            stations.append(current)
            key_value = stripped[2:].strip()
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                current[key.strip()] = _simple_yaml_scalar(value)
            continue
        if in_stations and current is not None and indent > 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key.strip() in {
                "station_guid",
                "name",
                "river",
                "role",
                "measure_id",
            }:
                current[key.strip()] = _simple_yaml_scalar(value)
            continue
        if ":" in stripped and indent == 0:
            key, value = stripped.split(":", 1)
            result[key.strip()] = _simple_yaml_scalar(value)
            in_stations = False
    if not result:
        raise ConfigError("fallback YAML parser found no settings")
    return result


def station_specs(
    config: Mapping[str, Any],
    station_guids: Sequence[str] | None = None,
) -> list[StationSpec]:
    """Validate stations and optionally select explicit GUIDs."""

    raw_stations = config.get("stations")
    if not isinstance(raw_stations, list) or not raw_stations:
        raise ConfigError("configuration must contain a non-empty stations list")
    specs = [StationSpec.from_mapping(raw) for raw in raw_stations]
    seen_guids: set[str] = set()
    seen_measures: set[str] = set()
    for spec in specs:
        if spec.station_guid in seen_guids:
            raise ConfigError(f"duplicate station_guid: {spec.station_guid}")
        if spec.measure_id in seen_measures:
            raise ConfigError(f"duplicate measure_id: {spec.measure_id}")
        seen_guids.add(spec.station_guid)
        seen_measures.add(spec.measure_id)
    if station_guids:
        requested = [str(value) for value in station_guids]
        by_guid = {spec.station_guid: spec for spec in specs}
        unknown = [value for value in requested if value not in by_guid]
        if unknown:
            raise ConfigError(
                "station GUID(s) not present in config: " + ", ".join(unknown)
            )
        return [by_guid[value] for value in requested]
    return specs


def archive_base_url(config: Mapping[str, Any]) -> str:
    value = str(config.get("archive_base_url") or BASE_URL).strip().rstrip("/")
    if not value.startswith(("https://", "http://")):
        raise ConfigError("archive_base_url must be an http(s) URL")
    return value


def _station_url(base_url: str, station_guid: str) -> str:
    return f"{base_url.rstrip('/')}/id/stations/{station_guid}.json"


def _measure_url(base_url: str, measure_id: str) -> str:
    return f"{base_url.rstrip('/')}/id/measures/{measure_id}.json"


def _readings_url(base_url: str, measure_id: str) -> str:
    return f"{base_url.rstrip('/')}/id/measures/{measure_id}/readings.json"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _measure_period(measure: Mapping[str, Any]) -> int:
    period = _int_or_none(measure.get("period"))
    if period is None or period <= 0:
        raise ExtractionError("configured measure metadata has no positive period")
    return period


def _validate_measure(
    station: Mapping[str, Any],
    measure: Mapping[str, Any],
    spec: StationSpec,
) -> int:
    actual_measure = _token(measure.get("@id") or measure.get("id") or measure.get("notation"))
    if actual_measure and actual_measure != spec.measure_id:
        raise ExtractionError(
            f"measure metadata ID {actual_measure!r} does not match configured "
            f"measure_id {spec.measure_id!r}"
        )
    actual_station = str(station.get("stationGuid") or "").strip()
    if actual_station and actual_station != spec.station_guid:
        raise ExtractionError(
            f"station metadata GUID {actual_station!r} does not match "
            f"configured station_guid {spec.station_guid!r}"
        )
    period = _measure_period(measure)
    if period != 900:
        raise ExtractionError(
            f"{spec.station_guid} measure {spec.measure_id} has period {period}s; "
            "Phase 2 expects the configured 15-minute (900s) series"
        )
    parameter = str(measure.get("parameter") or "").lower()
    if parameter and parameter not in {"level", "waterlevel", "water level"}:
        raise ExtractionError(
            f"{spec.measure_id} is parameter={measure.get('parameter')!r}, not level"
        )
    observation = _text(measure.get("observationType")).lower()
    if observation and observation != "qualified":
        raise ExtractionError(
            f"{spec.measure_id} is observationType={observation!r}, not Qualified"
        )
    return period


@dataclass
class QualityAccumulator:
    """Streaming quality/gap counters for one station's half-open window."""

    start: datetime
    end: datetime
    period_seconds: int
    rows_seen: int = 0
    rows_in_window: int = 0
    rows_outside_window: int = 0
    malformed_datetime_rows: int = 0
    duplicate_timestamp_rows: int = 0
    missing_value_rows: int = 0
    timestamps: set[datetime] = field(default_factory=set)
    grid_indices: set[int] = field(default_factory=set)
    nonmissing_grid_indices: set[int] = field(default_factory=set)
    usable_grid_indices: set[int] = field(default_factory=set)
    off_grid_timestamps: set[datetime] = field(default_factory=set)
    quality: Counter[str] = field(default_factory=Counter)
    qcode: Counter[str] = field(default_factory=Counter)
    completeness: Counter[str] = field(default_factory=Counter)
    step_seconds: Counter[int] = field(default_factory=Counter)

    def observe(self, row: Mapping[str, Any]) -> None:
        self.rows_seen += 1
        raw_datetime = row.get("dateTime")
        try:
            timestamp = parse_datetime(raw_datetime)
        except (TypeError, ValueError):
            self.malformed_datetime_rows += 1
            return
        if not self.start <= timestamp < self.end:
            self.rows_outside_window += 1
            return
        self.rows_in_window += 1
        quality_key = str(row.get("quality") or "(blank)")
        qcode_key = str(row.get("qcode") or row.get("qflag") or "(blank)")
        completeness_key = str(row.get("completeness") or "(blank)")
        self.quality[quality_key] += 1
        self.qcode[qcode_key] += 1
        self.completeness[completeness_key] += 1
        if timestamp in self.timestamps:
            self.duplicate_timestamp_rows += 1
        self.timestamps.add(timestamp)
        offset = (timestamp - self.start).total_seconds()
        on_grid = offset >= 0 and offset % self.period_seconds == 0
        if on_grid:
            self.grid_indices.add(int(offset // self.period_seconds))
        else:
            self.off_grid_timestamps.add(timestamp)
        missing = row.get("value") is None or row.get("value") == ""
        if missing:
            self.missing_value_rows += 1
        elif on_grid:
            self.nonmissing_grid_indices.add(int(offset // self.period_seconds))
            if quality_key in {"Good", "Estimated"}:
                self.usable_grid_indices.add(int(offset // self.period_seconds))

    def finish(self) -> dict[str, Any]:
        expected_slots = max(
            0, int((self.end - self.start).total_seconds() // self.period_seconds)
        )
        timestamp_grid_count = len(self.grid_indices)
        usable_grid_count = len(self.usable_grid_indices)
        ordered = sorted(self.timestamps)
        if len(ordered) > 1:
            for left, right in zip(ordered, ordered[1:]):
                seconds = int((right - left).total_seconds())
                if seconds > 0:
                    self.step_seconds[seconds] += 1
        mode_step = self.step_seconds.most_common(1)[0][0] if self.step_seconds else None
        max_gap = max(self.step_seconds) if self.step_seconds else None
        return {
            "rows_seen": self.rows_seen,
            "rows_in_window": self.rows_in_window,
            "rows_outside_window": self.rows_outside_window,
            "malformed_datetime_rows": self.malformed_datetime_rows,
            "unique_timestamps": len(self.timestamps),
            "duplicate_timestamp_rows": self.duplicate_timestamp_rows,
            "expected_slots": expected_slots,
            "timestamp_grid_count": timestamp_grid_count,
            "timestamp_gap_slots": max(0, expected_slots - timestamp_grid_count),
            "timestamp_coverage_pct": (
                100.0 * timestamp_grid_count / expected_slots if expected_slots else 0.0
            ),
            "off_grid_unique": len(self.off_grid_timestamps),
            "missing_value_rows": self.missing_value_rows,
            "non_missing_value_count": len(self.nonmissing_grid_indices),
            "non_missing_value_coverage_pct": (
                100.0 * len(self.nonmissing_grid_indices) / expected_slots if expected_slots else 0.0
            ),
            "usable_value_count": usable_grid_count,
            "usable_value_coverage_pct": (
                100.0 * usable_grid_count / expected_slots if expected_slots else 0.0
            ),
            "first_timestamp": ordered[0].isoformat() if ordered else None,
            "last_timestamp": ordered[-1].isoformat() if ordered else None,
            "mode_step_seconds": mode_step,
            "max_step_seconds": max_gap,
            "quality_counts": dict(sorted(self.quality.items())),
            "qcode_counts": dict(sorted(self.qcode.items())),
            "completeness_counts": dict(sorted(self.completeness.items())),
        }


@dataclass(frozen=True)
class ReadingPage:
    """One archive page and its request provenance."""

    rows: list[dict[str, Any]]
    meta: dict[str, Any]
    url: str


def iter_reading_pages(
    client: ArchiveClient,
    *,
    base_url: str,
    measure_id: str,
    start: date,
    end: date,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> Iterator[ReadingPage]:
    """Yield a complete half-open window using limit/offset pages."""

    if not 1 <= page_limit <= HARD_LIMIT:
        raise PaginationError(f"page_limit must be between 1 and {HARD_LIMIT}")
    if end <= start:
        raise PaginationError("end date must be after start date")
    if max_pages <= 0:
        raise PaginationError("max_pages must be positive")
    readings_url = _readings_url(base_url, measure_id)
    offset = 0
    for _page_number in range(1, max_pages + 1):
        if offset >= HARD_LIMIT:
            raise PaginationError(
                "the 2,000,000-row archive hard limit was reached; "
                "use the asynchronous batch-readings endpoint"
            )
        payload, request_url = client.get_json(
            readings_url,
            {
                "mineq-date": start.isoformat(),
                "max-date": end.isoformat(),
                "_limit": page_limit,
                "_offset": offset,
                "_view": "full",
            },
        )
        rows = _response_items(payload)
        raw_meta = payload.get("meta")
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        yield ReadingPage(rows=rows, meta=meta, url=request_url)
        effective_limit = _int_or_none(meta.get("limit")) or page_limit
        if len(rows) < effective_limit:
            return
        offset += len(rows)
        if offset > HARD_LIMIT:
            raise PaginationError(
                "the archive response exceeded the 2,000,000-row hard limit"
            )
    raise PaginationError(
        f"pagination exceeded max_pages={max_pages}; use a narrower window or batch API"
    )


class AtomicRowWriter:
    """Stream rows to a temporary file and atomically publish on success."""

    def __init__(self, destination: Path, output_format: str, force: bool = False):
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"unsupported output format: {output_format}")
        self.destination = Path(destination)
        self.output_format = output_format
        self.force = force
        self.temp_path: Path | None = None
        self.stream: Any = None
        self._first_json = True
        self._csv_writer: csv.DictWriter[str] | None = None

    def __enter__(self) -> "AtomicRowWriter":
        if self.destination.exists() and not self.force:
            raise ExtractionError(
                f"output exists: {self.destination}; pass --force to replace it"
            )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{self.destination.name}.",
            suffix=".part",
            dir=self.destination.parent,
            delete=False,
        )
        self.temp_path = Path(handle.name)
        self.stream = handle
        if self.output_format == "json":
            self.stream.write("[")
        elif self.output_format == "csv":
            self._csv_writer = csv.DictWriter(
                self.stream,
                fieldnames=[*CSV_FIELDS, "extra_json"],
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self.stream is None:
            raise ExtractionError("row writer is not open")
        if self.output_format == "jsonl":
            self.stream.write(
                json.dumps(_jsonable(dict(row)), ensure_ascii=True, separators=(",", ":"))
                + "\n"
            )
            return
        if self.output_format == "json":
            if not self._first_json:
                self.stream.write(",")
            self.stream.write(
                json.dumps(_jsonable(dict(row)), ensure_ascii=True, separators=(",", ":"))
            )
            self._first_json = False
            return
        assert self._csv_writer is not None
        known = set(CSV_FIELDS)
        output: dict[str, Any] = {}
        for field_name in CSV_FIELDS:
            value = row.get(field_name)
            if isinstance(value, (Mapping, list, tuple)):
                output[field_name] = json.dumps(
                    _jsonable(value), ensure_ascii=True, separators=(",", ":")
                )
            else:
                output[field_name] = value
        extras = {key: _jsonable(value) for key, value in row.items() if key not in known}
        output["extra_json"] = (
            json.dumps(extras, ensure_ascii=True, separators=(",", ":")) if extras else ""
        )
        self._csv_writer.writerow(output)

    def commit(self) -> Path:
        if self.stream is None or self.temp_path is None:
            raise ExtractionError("row writer is not open")
        if self.output_format == "json":
            self.stream.write("]")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temp_path, self.destination)
        self.stream = None
        self.temp_path = None
        return self.destination

    def abort(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if self.temp_path is not None:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass
            self.temp_path = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False


@dataclass
class StationExtraction:
    """Result returned after one station's rows have been fully processed."""

    spec: StationSpec
    station_metadata: dict[str, Any]
    measure_metadata: dict[str, Any]
    summary: dict[str, Any]
    request_urls: list[str]
    api_metas: list[dict[str, Any]]
    output_path: Path | None = None

    def manifest_record(self, output_dir: Path) -> dict[str, Any]:
        relative = (
            self.output_path.relative_to(output_dir).as_posix()
            if self.output_path
            else None
        )
        versions = sorted(
            {
                str(meta.get("version"))
                for meta in self.api_metas
                if meta.get("version") not in (None, "")
            }
        )
        return {
            "station_guid": self.spec.station_guid,
            "name": self.spec.name,
            "river": self.spec.river,
            "role": self.spec.role,
            "measure_id": self.spec.measure_id,
            "flood_monitoring_crosswalk": _jsonable(dict(self.spec.flood_monitoring)),
            "file": relative,
            "request_urls": list(self.request_urls),
            "api_versions": versions,
            "station_metadata": _jsonable(self.station_metadata),
            "measure_metadata": _jsonable(self.measure_metadata),
            "summary": _jsonable(self.summary),
        }


def extract_station(
    client: ArchiveClient,
    spec: StationSpec,
    *,
    base_url: str = BASE_URL,
    start: date,
    end: date,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    output_path: Path | None = None,
    output_format: str = "jsonl",
    force: bool = False,
) -> StationExtraction:
    """Extract one configured station and optionally stream its rows to disk."""

    if end <= start:
        raise ExtractionError("end date must be after start date")
    station_payload, station_request_url = client.get_json(
        _station_url(base_url, spec.station_guid)
    )
    station_metadata = _first_item(station_payload, "station metadata")
    measure_payload, measure_request_url = client.get_json(
        _measure_url(base_url, spec.measure_id)
    )
    measure_metadata = _first_item(measure_payload, "measure metadata")
    period = _validate_measure(station_metadata, measure_metadata, spec)

    accumulator = QualityAccumulator(
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        period_seconds=period,
    )
    request_urls = [station_request_url, measure_request_url]
    api_metas: list[dict[str, Any]] = []
    for payload in (station_payload, measure_payload):
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            api_metas.append(dict(meta))

    writer = (
        AtomicRowWriter(output_path, output_format, force=force)
        if output_path is not None
        else None
    )
    try:
        if writer is not None:
            writer.__enter__()
        page_count = 0
        page_limits: list[int] = []
        for page in iter_reading_pages(
            client,
            base_url=base_url,
            measure_id=spec.measure_id,
            start=start,
            end=end,
            page_limit=page_limit,
            max_pages=max_pages,
        ):
            page_count += 1
            request_urls.append(page.url)
            api_metas.append(page.meta)
            page_limit_from_meta = _int_or_none(page.meta.get("limit"))
            if page_limit_from_meta is not None:
                page_limits.append(page_limit_from_meta)
            for row in page.rows:
                accumulator.observe(row)
                if writer is not None:
                    writer.write(row)
        summary = accumulator.finish()
        summary.update(
            {
                "period_seconds": period,
                "period_name": str(measure_metadata.get("periodName") or ""),
                "unit_name": str(
                    measure_metadata.get("unitName") or measure_metadata.get("unit") or ""
                ),
                "parameter": str(measure_metadata.get("parameter") or ""),
                "value_type": str(measure_metadata.get("valueType") or ""),
                "observation_type": _text(measure_metadata.get("observationType")),
                "qualifier": _text(measure_metadata.get("qualifier")),
                "has_telemetry": measure_metadata.get("hasTelemetry"),
                "page_count": page_count,
                "page_limits": page_limits,
                "requested_page_limit": page_limit,
                "pagination_truncated": False,
                "station_status": _text(station_metadata.get("status")),
                "datum": station_metadata.get("datum"),
                "datum_type": station_metadata.get("datumType"),
            }
        )
        if writer is not None:
            writer.commit()
            published_path = output_path
        else:
            published_path = None
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    return StationExtraction(
        spec=spec,
        station_metadata=station_metadata,
        measure_metadata=measure_metadata,
        summary=summary,
        request_urls=request_urls,
        api_metas=api_metas,
        output_path=published_path,
    )


def output_suffix(output_format: str) -> str:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output format must be one of {', '.join(OUTPUT_FORMATS)}")
    return {"jsonl": ".jsonl", "csv": ".csv", "json": ".json"}[output_format]


def build_plan(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    start: date = parse_date(DEFAULT_START),
    end: date = parse_date(DEFAULT_END),
    output_format: str = "jsonl",
    station_guids: Sequence[str] | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
) -> list[dict[str, Any]]:
    """Build a no-network extraction plan suitable for dry-run output."""

    if end <= start:
        raise ConfigError("end date must be after start date")
    if not 1 <= page_limit <= HARD_LIMIT:
        raise ConfigError(f"page_limit must be between 1 and {HARD_LIMIT}")
    config = load_config(config_path)
    specs = station_specs(config, station_guids)
    base_url = archive_base_url(config)
    output_root = Path(output_dir)
    suffix = output_suffix(output_format)
    plan = []
    for spec in specs:
        readings_url = _readings_url(base_url, spec.measure_id)
        first_page_url = _with_params(
            readings_url,
            {
                "mineq-date": start.isoformat(),
                "max-date": end.isoformat(),
                "_limit": page_limit,
                "_offset": 0,
                "_view": "full",
            },
        )
        plan.append(
            {
                "station_guid": spec.station_guid,
                "name": spec.name,
                "river": spec.river,
                "measure_id": spec.measure_id,
                "window_start": start.isoformat(),
                "window_end_exclusive": end.isoformat(),
                "page_limit": page_limit,
                "first_page_url": first_page_url,
                "output_path": str(output_root / f"{spec.station_guid}{suffix}"),
            }
        )
    return plan


def print_plan(plan: Iterable[Mapping[str, Any]]) -> None:
    rows = list(plan)
    print("EA hydrology extraction plan (dry run; no network requests made)")
    print("station_guid\tname\triver\tmeasure_id\twindow\tpage_limit\toutput")
    for item in rows:
        print(
            "\t".join(
                [
                    str(item["station_guid"]),
                    str(item["name"]),
                    str(item["river"]),
                    str(item["measure_id"]),
                    f"[{item['window_start']}, {item['window_end_exclusive']})",
                    str(item["page_limit"]),
                    str(item["output_path"]),
                ]
            )
        )
        print(f"  first_page={item['first_page_url']}")


def run_extraction(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    start: date = parse_date(DEFAULT_START),
    end: date = parse_date(DEFAULT_END),
    output_format: str = "jsonl",
    station_guids: Sequence[str] | None = None,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    force: bool = False,
    client: ArchiveClient | None = None,
) -> list[StationExtraction]:
    """Run a serial extraction for the selected configured stations."""

    config = load_config(config_path)
    specs = station_specs(config, station_guids)
    base_url = archive_base_url(config)
    output_root = Path(output_dir)
    if output_format not in OUTPUT_FORMATS:
        raise ExtractionError(f"output format not supported: {output_format}")
    if not 1 <= page_limit <= HARD_LIMIT:
        raise ExtractionError(f"page_limit must be between 1 and {HARD_LIMIT}")
    if max_pages <= 0:
        raise ExtractionError("max_pages must be positive")
    if end <= start:
        raise ExtractionError("end date must be after start date")
    selected_client = client or ArchiveClient(
        timeout=timeout,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )
    suffix = output_suffix(output_format)
    results: list[StationExtraction] = []
    for index, spec in enumerate(specs):
        destination = output_root / f"{spec.station_guid}{suffix}"
        result = extract_station(
            selected_client,
            spec,
            base_url=base_url,
            start=start,
            end=end,
            page_limit=page_limit,
            max_pages=max_pages,
            output_path=destination,
            output_format=output_format,
            force=force,
        )
        results.append(result)
        if index + 1 < len(specs) and sleep_seconds:
            time.sleep(sleep_seconds)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", type=parse_date, default=parse_date(DEFAULT_START))
    parser.add_argument("--end-date", type=parse_date, default=parse_date(DEFAULT_END))
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=OUTPUT_FORMATS,
        default="jsonl",
        help="raw row output format (default: jsonl)",
    )
    parser.add_argument(
        "--station",
        action="append",
        dest="station_guids",
        metavar="GUID",
        help="explicit station GUID; repeat to select a subset",
    )
    parser.add_argument(
        "--limit",
        dest="page_limit",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help=f"archive _limit per page (1..{HARD_LIMIT}; default {DEFAULT_PAGE_LIMIT})",
    )
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print URLs/output paths without making requests or writing files",
    )
    return parser


def _print_results(results: Sequence[StationExtraction]) -> None:
    print("EA hydrology extraction complete")
    print("station_guid\tname\trows\tunique\tduplicates\tmissing\tusable_coverage_pct\tfile")
    for result in results:
        summary = result.summary
        print(
            "\t".join(
                [
                    result.spec.station_guid,
                    result.spec.name,
                    str(summary.get("rows_in_window", 0)),
                    str(summary.get("unique_timestamps", 0)),
                    str(summary.get("duplicate_timestamp_rows", 0)),
                    str(summary.get("missing_value_rows", 0)),
                    f"{float(summary.get('usable_value_coverage_pct', 0.0)):.3f}",
                    str(result.output_path or ""),
                ]
            )
        )
        print(
            "  quality="
            + json.dumps(summary.get("quality_counts", {}), sort_keys=True)
            + " qcode="
            + json.dumps(summary.get("qcode_counts", {}), sort_keys=True)
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.end_date <= args.start_date:
            parser.error("--end-date must be after --start-date")
        if args.page_limit < 1 or args.page_limit > HARD_LIMIT:
            parser.error(f"--limit must be between 1 and {HARD_LIMIT}")
        if args.max_pages <= 0 or args.timeout <= 0 or args.retries < 0:
            parser.error("--max-pages/--timeout must be positive and --retries non-negative")
        if args.sleep_seconds < 0:
            parser.error("--sleep-seconds cannot be negative")
        if args.dry_run:
            print_plan(
                build_plan(
                    args.config,
                    output_dir=args.output_dir,
                    start=args.start_date,
                    end=args.end_date,
                    output_format=args.output_format,
                    station_guids=args.station_guids,
                    page_limit=args.page_limit,
                )
            )
            return 0
        results = run_extraction(
            args.config,
            output_dir=args.output_dir,
            start=args.start_date,
            end=args.end_date,
            output_format=args.output_format,
            station_guids=args.station_guids,
            page_limit=args.page_limit,
            max_pages=args.max_pages,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            force=args.force,
        )
        _print_results(results)
        return 0
    except (ConfigError, ExtractionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
