"""Plot saved TFT variable-selection/attention diagnostics.

The input is a JSON result from ``src.river_experiment --run-tft``. The plot
is descriptive only; it must not be read as a causal explanation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Interpretability plotting requires matplotlib.") from exc
    return plt


def load_station_interpretation(result_path: Path | str, station_guid: str | None = None) -> tuple[str, dict[str, Any]]:
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    tft = payload.get("tft", payload)
    if isinstance(tft, dict) and isinstance(tft.get("result"), dict):
        tft = tft["result"]
    candidates = tft.get("by_station", []) if isinstance(tft, dict) else []
    for record in candidates:
        if record.get("status") != "measured":
            continue
        guid = str(record.get("station_guid", ""))
        if station_guid is None or guid == station_guid:
            interpretation = record.get("interpretability")
            if isinstance(interpretation, dict):
                return guid, interpretation
    raise ValueError("result does not contain a measured station interpretation")


def plot_interpretation(
    result_path: Path | str,
    output_path: Path | str,
    *,
    station_guid: str | None = None,
) -> Path:
    """Write a compact variable-importance and attention figure."""

    plt = _require_matplotlib()
    guid, interpretation = load_station_interpretation(result_path, station_guid)
    groups = [
        ("Encoder variables", interpretation.get("encoder_variables", {})),
        ("Decoder variables", interpretation.get("decoder_variables", {})),
        ("Static variables", interpretation.get("static_variables", {})),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for axis, (title, values) in zip(axes, groups):
        if not isinstance(values, dict) or not values:
            axis.text(0.5, 0.5, "no values", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = list(values)
        numbers = [float(values[label]) for label in labels]
        order = sorted(range(len(labels)), key=lambda index: numbers[index])
        labels = [labels[index] for index in order]
        numbers = [numbers[index] for index in order]
        axis.barh(labels, numbers, color="#285f8f")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(f"TFT interpretation diagnostics: {guid}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, default=Path("results/river_one_station_tft.json"))
    parser.add_argument("--station", dest="station_guid")
    parser.add_argument("--output", type=Path, default=Path("reports/figures/tft_interpretability.png"))
    args = parser.parse_args(argv)
    try:
        path = plot_interpretation(args.results_json, args.output, station_guid=args.station_guid)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
