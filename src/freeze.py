"""Freeze an EA hydrology extract with provenance and SHA-256 checksums.

The command is explicit about writes. The dry-run option reads only the station
configuration and prints the planned URLs/files. A normal run requests the
configured stations serially through src.extract, writes raw 15-minute rows,
then creates manifest.json and checksums.sha256 in the output directory. No
cross-API ID is inferred and no generated data is created until the caller
omits dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .extract import (
        DEFAULT_CONFIG,
        DEFAULT_END,
        DEFAULT_MAX_PAGES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAGE_LIMIT,
        DEFAULT_RETRIES,
        DEFAULT_SLEEP_SECONDS,
        DEFAULT_START,
        DEFAULT_TIMEOUT,
        HARD_LIMIT,
        OUTPUT_FORMATS,
        ArchiveClient,
        ConfigError,
        ExtractionError,
        StationExtraction,
        archive_base_url,
        build_plan,
        extract_station,
        iso_utc,
        load_config,
        output_suffix,
        parse_date,
        sha256_file,
        station_specs,
    )
except ImportError:  # pragma: no cover - supports python src/freeze.py
    from extract import (  # type: ignore[no-redef]
        DEFAULT_CONFIG,
        DEFAULT_END,
        DEFAULT_MAX_PAGES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAGE_LIMIT,
        DEFAULT_RETRIES,
        DEFAULT_SLEEP_SECONDS,
        DEFAULT_START,
        DEFAULT_TIMEOUT,
        HARD_LIMIT,
        OUTPUT_FORMATS,
        ArchiveClient,
        ConfigError,
        ExtractionError,
        StationExtraction,
        archive_base_url,
        build_plan,
        extract_station,
        iso_utc,
        load_config,
        output_suffix,
        parse_date,
        sha256_file,
        station_specs,
    )


MANIFEST_NAME = "manifest.json"
CHECKSUM_NAME = "checksums.sha256"
MANIFEST_SCHEMA = "ea-hydrology-freeze-v1"
LICENSE_URL = "http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
LICENSE_ACKNOWLEDGEMENT = (
    "Contains public sector information licensed under the Open Government Licence v3.0."
)


def _atomic_text(path: Path, content: str, *, force: bool = False) -> None:
    """Atomically write a small manifest/checksum file."""

    if path.exists() and not force:
        raise ExtractionError(
            f"output exists: {path}; pass --force to replace the frozen dataset"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _config_hash(config_path: Path) -> str:
    try:
        return sha256_file(config_path)
    except OSError as exc:
        raise ConfigError(f"cannot hash configuration {config_path}: {exc}") from exc


def _first_meta(results: Sequence[StationExtraction]) -> dict[str, Any]:
    """Return the first useful API metadata object in stable form."""

    for result in results:
        for meta in result.api_metas:
            if isinstance(meta, Mapping):
                return dict(meta)
    return {}


def _api_versions(results: Sequence[StationExtraction]) -> list[str]:
    versions = {
        str(meta.get("version"))
        for result in results
        for meta in result.api_metas
        if isinstance(meta, Mapping) and meta.get("version") not in (None, "")
    }
    return sorted(versions)


def _record_with_hash(result: StationExtraction, output_dir: Path) -> dict[str, Any]:
    if result.output_path is None:
        raise ExtractionError(
            f"{result.spec.station_guid} completed without an output file"
        )
    record = result.manifest_record(output_dir)
    path = result.output_path
    record["file_sha256"] = sha256_file(path)
    record["file_bytes"] = path.stat().st_size
    record["row_count"] = result.summary.get("rows_in_window", 0)
    return record


def _checksums_text(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> str:
    """Build a sha256sum-compatible list for data files and the manifest."""

    lines = [
        f"{record['file_sha256']}  {record['file']}"
        for record in records
        if record.get("file") and record.get("file_sha256")
    ]
    lines.append(f"{sha256_file(manifest_path)}  {manifest_path.name}")
    return "\n".join(lines) + "\n"


def _source_metadata(
    results: Sequence[StationExtraction],
    *,
    base_url: str,
    output_format: str,
    start: date,
    end: date,
    page_limit: int,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    meta = _first_meta(results)
    return {
        "service": "Environment Agency hydrology archive API",
        "base_url": base_url,
        "output_format": output_format,
        "window": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "timestamp_handling": {
            "input": "preserved exactly as returned by API",
            "comparison_axis": "naive ISO values interpreted on a UTC-like axis",
            "authoritative_archive_timezone": "[UNVERIFIED]",
        },
        "request_policy": {
            "serial": True,
            "page_limit": page_limit,
            "soft_row_limit": 100_000,
            "hard_row_limit": HARD_LIMIT,
            "retries": retries,
            "sleep_seconds_between_stations": sleep_seconds,
            "batch_endpoint_for_larger_extracts": (
                f"{base_url}/data/batch-readings/batch"
            ),
        },
        "api_version": meta.get("version"),
        "api_versions_seen": _api_versions(results),
        "publisher": meta.get("publisher", "Environment Agency"),
        "license_url": meta.get("license", LICENSE_URL),
        "license_name": meta.get("licenseName", "OGL 3"),
        "attribution": LICENSE_ACKNOWLEDGEMENT,
    }


def freeze_dataset(
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
    dry_run: bool = False,
    client: ArchiveClient | None = None,
) -> dict[str, Any]:
    """Extract and freeze configured stations, returning a manifest mapping.

    dry_run makes no network calls and writes no files. On a normal run,
    failures abort before the manifest is published; any successfully completed
    raw files remain intact so a caller can inspect or remove them explicitly.
    """

    if end <= start:
        raise ExtractionError("end date must be after start date")
    if output_format not in OUTPUT_FORMATS:
        raise ExtractionError(f"output format must be one of {OUTPUT_FORMATS}")
    if not 1 <= page_limit <= HARD_LIMIT:
        raise ExtractionError(f"page_limit must be between 1 and {HARD_LIMIT}")
    if max_pages <= 0 or timeout <= 0 or retries < 0 or sleep_seconds < 0:
        raise ExtractionError("max_pages/timeout must be positive; retry/sleep values invalid")

    if dry_run:
        plan = build_plan(
            config_path,
            output_dir=output_dir,
            start=start,
            end=end,
            output_format=output_format,
            station_guids=station_guids,
            page_limit=page_limit,
        )
        return {
            "schema": MANIFEST_SCHEMA,
            "dry_run": True,
            "station_count": len(plan),
            "window": {
                "start_inclusive": start.isoformat(),
                "end_exclusive": end.isoformat(),
            },
            "output_format": output_format,
            "plan": plan,
        }

    config_file = Path(config_path)
    config = load_config(config_file)
    specs = station_specs(config, station_guids)
    base_url = archive_base_url(config)
    output_root = Path(output_dir)
    manifest_path = output_root / MANIFEST_NAME
    checksum_path = output_root / CHECKSUM_NAME
    if manifest_path.exists() and not force:
        raise ExtractionError(
            f"manifest exists: {manifest_path}; pass --force to create a new freeze"
        )
    if checksum_path.exists() and not force:
        raise ExtractionError(
            f"checksum file exists: {checksum_path}; pass --force to create a new freeze"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = iso_utc()
    selected_client = client or ArchiveClient(
        timeout=timeout,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )
    results: list[StationExtraction] = []
    for index, spec in enumerate(specs):
        destination = output_root / f"{spec.station_guid}{output_suffix(output_format)}"
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

    records = [_record_with_hash(result, output_root) for result in results]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": started_at,
        "finished_at_utc": iso_utc(),
        "config": {
            "path": str(config_file),
            "sha256": _config_hash(config_file),
            "station_count": len(specs),
        },
        "source": _source_metadata(
            results,
            base_url=base_url,
            output_format=output_format,
            start=start,
            end=end,
            page_limit=page_limit,
            retries=retries,
            sleep_seconds=sleep_seconds,
        ),
        "stations": records,
        "files": [record["file"] for record in records],
        "quality_fields_preserved": [
            "quality",
            "qcode",
            "qflag",
            "completeness",
            "valid",
            "invalid",
            "missing",
        ],
        "checksums_file": CHECKSUM_NAME,
        "notes": [
            "Raw rows are preserved as returned by the archive.",
            "Archive readings can be revised; retain this manifest and checksum.",
            "No flood-monitoring ID was inferred from a hydrology station GUID.",
            "This extract is research data and is not a safety-critical warning feed.",
        ],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    _atomic_text(manifest_path, manifest_text, force=force)
    _atomic_text(
        checksum_path,
        _checksums_text(records, manifest_path=manifest_path),
        force=force,
    )
    return manifest


def _print_manifest_summary(manifest: Mapping[str, Any]) -> None:
    if manifest.get("dry_run"):
        print(
            "EA hydrology freeze plan (dry run; no network requests or files written)"
        )
        print(
            f"stations={manifest.get('station_count', 0)} "
            f"window=[{manifest.get('window', {}).get('start_inclusive')}, "
            f"{manifest.get('window', {}).get('end_exclusive')}) "
            f"format={manifest.get('output_format')}"
        )
        for item in manifest.get("plan", []):
            print(
                f"{item['station_guid']}\t{item['name']}\t{item['river']}\t"
                f"{item['measure_id']}\t{item['output_path']}"
            )
        return
    print("EA hydrology freeze complete")
    print(f"stations={len(manifest.get('stations', []))}")
    print(f"manifest={MANIFEST_NAME}")
    print(f"checksums={CHECKSUM_NAME}")
    for record in manifest.get("stations", []):
        summary = record.get("summary", {})
        print(
            f"{record.get('station_guid')}\t{record.get('name')}\t"
            f"rows={summary.get('rows_in_window', 0)}\t"
            f"missing={summary.get('missing_value_rows', 0)}\t"
            f"duplicates={summary.get('duplicate_timestamp_rows', 0)}\t"
            f"sha256={record.get('file_sha256')}"
        )


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
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the station/measure/window plan without downloading",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = freeze_dataset(
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
            dry_run=args.dry_run,
        )
        _print_manifest_summary(manifest)
        return 0
    except (ConfigError, ExtractionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
