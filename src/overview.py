"""Create one offline series/gap/quality figure per prepared station."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _require_pandas_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Overview figures require pandas and matplotlib; install environment.yml first.") from exc
    return pd, plt


def plot_station_file(
    csv_path: Path,
    output_path: Path,
    *,
    station_name: str,
    river: str,
) -> Path:
    """Plot level, invalid-bin mask, and quality labels for one station."""

    pd, plt = _require_pandas_matplotlib()
    frame = pd.read_csv(csv_path, parse_dates=["dateTime"])
    if "dateTime" not in frame or "value" not in frame:
        raise ValueError(f"hourly file lacks dateTime/value columns: {csv_path}")
    frame.sort_values("dateTime", inplace=True)
    valid = frame.get("aggregation_valid", pd.Series(True, index=frame.index)).astype(bool)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=(4, 1))
    axes[0].plot(frame["dateTime"], frame["value"], color="#1f5a85", linewidth=0.7)
    axes[0].scatter(
        frame.loc[~valid, "dateTime"],
        frame.loc[~valid, "value"],
        color="#b83b3b",
        s=7,
        label="masked hourly bin",
    )
    axes[0].set_ylabel("Level (m)")
    axes[0].set_title(f"{station_name} - {river}: hourly level and quality mask")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)
    axes[1].step(frame["dateTime"], valid.astype(int), where="post", color="#2f7d4a")
    axes[1].set_yticks([0, 1], ["masked", "valid"])
    axes[1].set_ylabel("Bin")
    axes[1].set_xlabel("UTC timestamp (pipeline assumption; confirm archive semantics)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def create_overviews(
    *,
    hourly_dir: Path | str = Path("data/hourly"),
    manifest_path: Path | str = Path("data/frozen/manifest.json"),
    output_dir: Path | str = Path("reports/figures"),
) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest.get("stations")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no stations")
    hourly_root = Path(hourly_dir)
    output_root = Path(output_dir)
    paths: list[Path] = []
    for record in records:
        guid = str(record.get("station_guid") or "")
        if not guid:
            raise ValueError("manifest station has no station_guid")
        paths.append(
            plot_station_file(
                hourly_root / f"{guid}.csv",
                output_root / f"{guid}.png",
                station_name=str(record.get("name") or guid),
                river=str(record.get("river") or ""),
            )
        )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-dir", type=Path, default=Path("data/hourly"))
    parser.add_argument("--manifest", type=Path, default=Path("data/frozen/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    try:
        paths = create_overviews(
            hourly_dir=args.hourly_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(f"created {len(paths)} overview figures")
        for path in paths:
            print(path)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
