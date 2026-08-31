"""Prepare frozen EA JSONL rows for hourly river modelling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Preparation requires pandas; install environment.yml first.") from exc
    return pd


def load_station_rows(path: Path, *, station_guid: str, measure_id: str) -> Any:
    """Load one JSONL file and attach immutable station/measure identity."""

    pd = _require_pandas()
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    frame["station_guid"] = station_guid
    frame["measure_id"] = measure_id
    return frame


def prepare_frozen_station(
    path: Path,
    *,
    station_guid: str,
    measure_id: str,
    source_timezone: str,
    target_timezone: str = "UTC",
    min_valid: int = 3,
) -> Any:
    """Return one station's hourly feature frame."""

    from .aggregate import aggregate_hourly, prepare_level_frame
    from .features import add_calendar_features, add_time_index

    raw = load_station_rows(path, station_guid=station_guid, measure_id=measure_id)
    prepared = prepare_level_frame(
        raw,
        source_timezone=source_timezone,
        target_timezone=target_timezone,
    )
    hourly = aggregate_hourly(prepared, min_valid=min_valid)
    hourly = add_calendar_features(hourly, timestamp_col="dateTime")
    hourly = add_time_index(hourly, timestamp_col="dateTime", group_col="station_guid")
    return hourly


def prepare_frozen_dataset(
    *,
    frozen_dir: Path | str = Path("data/frozen"),
    output_dir: Path | str | None = Path("data/hourly"),
    source_timezone: str,
    target_timezone: str = "UTC",
    min_valid: int = 3,
    station_guids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare configured JSONL files, optionally writing hourly CSV files."""

    pd = _require_pandas()
    frozen_root = Path(frozen_dir)
    manifest_path = frozen_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"frozen manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("stations")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no station records")
    selected = set(station_guids or [])
    if selected and not selected.issubset({str(r.get("station_guid")) for r in records}):
        unknown = sorted(selected - {str(r.get("station_guid")) for r in records})
        raise ValueError(f"station GUID(s) not present in manifest: {unknown}")
    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for record in records:
        guid = str(record.get("station_guid"))
        if selected and guid not in selected:
            continue
        relative = record.get("file")
        measure_id = str(record.get("measure_id") or "")
        if not isinstance(relative, str) or not measure_id:
            raise ValueError(f"manifest record for {guid} lacks file/measure_id")
        hourly = prepare_frozen_station(
            frozen_root / relative,
            station_guid=guid,
            measure_id=measure_id,
            source_timezone=source_timezone,
            target_timezone=target_timezone,
            min_valid=min_valid,
        )
        valid = int(hourly["aggregation_valid"].sum())
        summary = {
            "station_guid": guid,
            "name": record.get("name"),
            "river": record.get("river"),
            "raw_file": relative,
            "hourly_rows": int(len(hourly)),
            "valid_hourly_bins": valid,
            "masked_hourly_bins": int(len(hourly) - valid),
            "source_timezone": source_timezone,
            "target_timezone": target_timezone,
            "min_valid_per_hour": min_valid,
        }
        summaries.append(summary)
        if output_root is not None:
            hourly.to_csv(output_root / f"{guid}.csv", index=False)
    return summaries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=Path("data/frozen"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hourly"))
    parser.add_argument(
        "--source-timezone",
        required=True,
        help="explicit timezone for naive archive timestamps (for example UTC)",
    )
    parser.add_argument("--target-timezone", default="UTC")
    parser.add_argument("--min-valid", type=int, default=3)
    parser.add_argument("--station", action="append", dest="station_guids", metavar="GUID")
    parser.add_argument("--no-write", action="store_true", help="print summaries without writing hourly CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summaries = prepare_frozen_dataset(
            frozen_dir=args.frozen_dir,
            output_dir=None if args.no_write else args.output_dir,
            source_timezone=args.source_timezone,
            target_timezone=args.target_timezone,
            min_valid=args.min_valid,
            station_guids=args.station_guids,
        )
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
