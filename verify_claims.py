"""Independent verification of documented claims against checked-in artifacts.

Compares every measured number stated in RESULTS.md / ANOMALIES.md /
BENCHMARKS.md / DATA_QUALITY.md against the files they reference, and
recomputes pooled metrics from the scored CSVs where possible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
issues: list[str] = []
skips: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        issues.append(f"{name}: {detail}")


def skip(name: str, detail: str = "") -> None:
    """Record a check skipped because untracked raw data is absent locally.

    A skip is not a failure: raw extracts and large downloads are gitignored
    by design, so a fresh clone cannot audit them until they are regenerated.
    """
    print(f"[SKIP] {name}" + (f" -- {detail}" if detail else ""))
    skips.append(name)


def close(a, b, tol=5e-7) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


NAMES = {
    "8496ce69-482c-406a-a2f0-ac418ef8f099": "Kingston",
    "3c4d4f78-2d0e-474a-b884-65a9daca18fb": "Thorverton",
    "8820d897-a09e-4857-8095-5834fee6962f": "Bewdley",
    "0dcf81cb-5305-4e0b-b150-9b733ac44d0b": "Colwick",
    "213d70b2-894b-406b-9dc3-31d3ccec7f54": "Skelton",
    "e786e60f-a0f1-4955-aa57-f22ba39c7427": "Bywell",
    "9ad5d28c-7cfe-46db-b39d-58701689cd59": "Caton",
    "30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9": "Hereford Bridge",
    "3c43b72d-03a7-46e1-86c7-76dd97808544": "Roxton",
}

print("=" * 80)
print("A. RESULTS.md baseline ladder vs results/river_baselines.json")
print("=" * 80)
rb = json.loads((RESULTS / "river_baselines.json").read_text())
expected = {
    "seasonal_naive": (0.0740606, 0.0726406, 0.148121),
    "ets": (0.0598863, 0.0634740, 0.119773),
    "lightgbm": (0.0785723, 0.0655181, 0.157145),
}
for mname, (p50, p90, mae) in expected.items():
    m = rb["baselines"][mname]
    check(f"{mname} status ok/measured", m["status"] in ("ok", "measured"), m["status"])
    check(f"{mname} p50=={p50}", close(m["metrics"]["p50_pinball"], p50), str(m["metrics"].get("p50_pinball")))
    check(f"{mname} p90=={p90}", close(m["metrics"]["p90_pinball"], p90), str(m["metrics"].get("p90_pinball")))
    check(f"{mname} mae=={mae}", close(m["metrics"]["mae_from_p50"], mae, 5e-6), str(m["metrics"].get("mae_from_p50")))
    check(f"{mname} n_forecasts==3072", m["n_forecasts"] == 3072, str(m["n_forecasts"]))
    check(f"{mname} 8 stations in by_station", len(m["by_station"]) == 8, str(len(m["by_station"])))
check("9 stations in station catalogue", len(rb["stations"]) == 9, str(len(rb["stations"])))
check("split matches configs (train 2025-07-01/val 2025-10-01/test 2026-01-01)",
      rb["split"]["train_end_exclusive"] == "2025-07-01"
      and rb["split"]["validation_end_exclusive"] == "2025-10-01"
      and rb["split"]["test_end_exclusive"] == "2026-01-01", json.dumps(rb["split"]))

print()
print("=" * 80)
print("B. RESULTS.md TFT smoke runs")
print("=" * 80)
one = json.loads((RESULTS / "river_one_station_tft.json").read_text())
t1 = one["tft"]["result"]
check("one-station TFT status ok", one["tft"]["status"] == "ok")
check("one-station TFT is Kingston", len(t1["by_station"]) == 1
      and t1["by_station"][0]["station_guid"] == "8496ce69-482c-406a-a2f0-ac418ef8f099")
pooled1 = t1["pooled"]["metrics"]
check("Kingston TFT p50==0.0242913", close(pooled1["p50_pinball"], 0.0242913), str(pooled1.get("p50_pinball")))
check("Kingston TFT p90==0.0087182", close(pooled1["p90_pinball"], 0.0087182), str(pooled1.get("p90_pinball")))
check("Kingston TFT mae==0.0485825", close(pooled1["mae_from_p50"], 0.0485825), str(pooled1.get("mae_from_p50")))
check("Kingston TFT n_forecasts==48", len(t1["forecasts"]) == 48, str(len(t1["forecasts"])))

allr = json.loads((RESULTS / "river_all_stations_tft_epoch1.json").read_text())
ta = allr["tft"]["result"]
pa = ta["pooled"]["metrics"]
check("all-station TFT pooled p50==0.0377434", close(pa["p50_pinball"], 0.0377434), str(pa.get("p50_pinball")))
check("all-station TFT pooled p90==0.0227445", close(pa["p90_pinball"], 0.0227445), str(pa.get("p90_pinball")))
check("all-station TFT pooled mae==0.0754868", close(pa["mae_from_p50"], 0.0754868), str(pa.get("mae_from_p50")))
check("all-station TFT n_forecasts==384", len(ta["forecasts"]) == 384, str(len(ta["forecasts"])))
_measured = [s for s in ta["by_station"] if s.get("status") == "measured"]
_skipped = [s for s in ta["by_station"] if s.get("status") == "skipped"]
check("all-station TFT 8 measured + 1 explicit skip (Hereford)", len(ta["by_station"]) == 9
      and len(_measured) == 8 and len(_skipped) == 1
      and _skipped[0]["station_guid"] == "30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9"
      and "validation" in _skipped[0].get("reason", ""), str(_skipped))
per_station_doc = {
    "Colwick": (0.0279538, 0.0212620, 0.0559076),
    "Skelton": (0.0532578, 0.0321248, 0.106516),
    "Roxton": (0.0191547, 0.0114058, 0.0383094),
    "Thorverton": (0.0467124, 0.0146771, 0.0934248),
    "Kingston": (0.0242913, 0.0087182, 0.0485825),
    "Bewdley": (0.0498073, 0.0433767, 0.0996145),
    "Caton": (0.0533705, 0.0194122, 0.106741),
    "Bywell": (0.0273994, 0.0309792, 0.0547987),
}
for s in _measured:
    nm = NAMES[s["station_guid"]]
    mtr = s["metrics"]
    p50, p90, mae = per_station_doc[nm]
    ok = close(mtr["p50_pinball"], p50) and close(mtr["p90_pinball"], p90) and close(mtr["mae_from_p50"], mae, 5e-6)
    check(f"per-station TFT metrics {nm}", ok,
          f"got {mtr['p50_pinball']}/{mtr['p90_pinball']}/{mtr['mae_from_p50']} want {p50}/{p90}/{mae}")

print()
print("=" * 80)
print("C. ANOMALIES.md seasonal-naive review vs anomaly_review.json + anomaly_scored.csv")
print("=" * 80)
ar = json.loads((RESULTS / "anomaly_review.json").read_text())
check("calibration n_forecasts==17640", ar["calibration"]["n_forecasts"] == 17640, str(ar["calibration"]["n_forecasts"]))
check("calibration n_references==192", ar["calibration"]["n_references"] == 192, str(ar["calibration"]["n_references"]))
check("test n_forecasts==3072", ar["test"]["n_forecasts"] == 3072, str(ar["test"]["n_forecasts"]))
check("threshold==3.5", ar["threshold"] == 3.5)
summ = {NAMES[s["station_guid"]]: s for s in ar["summary_by_station"]}
anom_doc = {  # (large_residual, outside_interval, combined)
    "Colwick": (64, 384, 384), "Skelton": (123, 384, 384), "Roxton": (58, 381, 381),
    "Thorverton": (63, 384, 384), "Kingston": (7, 382, 382), "Bewdley": (143, 383, 383),
    "Caton": (76, 384, 384), "Bywell": (60, 384, 384),
}
for nm, (lr, oi, cb) in anom_doc.items():
    s = summ[nm]
    ok = s["large_residual_count"] == lr and s["outside_interval_count"] == oi and s["anomaly_count"] == cb and s["n_forecasts"] == 384
    check(f"seasonal-naive anomalies {nm} ({lr}/{oi}/{cb})", ok,
          f"got {s['large_residual_count']}/{s['outside_interval_count']}/{s['anomaly_count']}")
tot = ar["summary_by_station"]
check("pooled large==594", sum(s["large_residual_count"] for s in tot) == 594)
check("pooled outside==3066", sum(s["outside_interval_count"] for s in tot) == 3066)
check("pooled combined==3066", sum(s["anomaly_count"] for s in tot) == 3066)

sc = pd.read_csv(RESULTS / "anomaly_scored.csv")
check("anomaly_scored.csv rows==3072", len(sc) == 3072, str(len(sc)))
check("csv large_residual sum==594", int(sc["large_residual"].sum()) == 594, str(int(sc["large_residual"].sum())))
check("csv outside_interval sum==3066", int(sc["outside_interval"].sum()) == 3066, str(int(sc["outside_interval"].sum())))
check("csv anomaly sum==3066", int(sc["anomaly"].sum()) == 3066, str(int(sc["anomaly"].sum())))
# recompute seasonal-naive pooled p50 pinball from the scored csv (q=0.5 => 0.5*MAE)
p50_re = float(np.mean(0.5 * np.abs(sc["actual"] - sc["p50"])))
check("recomputed seasonal-naive p50 pinball==0.0740606", close(p50_re, 0.0740606, 5e-6), f"{p50_re:.7f}")
# point baseline degenerate intervals claim
check("seasonal-naive intervals degenerate (p10==p50==p90)",
      bool((sc["p10"] == sc["p50"]).all() and (sc["p50"] == sc["p90"]).all()))
# recompute residual scores to confirm definition
rs = np.abs(sc["actual"] - sc["p50"] - sc["residual_median"]) / sc["residual_scale"]
check("residual_score column consistent with |residual-median|/scale",
      bool(np.allclose(rs, sc["residual_score"], atol=1e-9)))
check("large_residual flag == (score>=3.5)",
      bool(((sc["residual_score"] >= 3.5) == sc["large_residual"]).all()))

print()
print("=" * 80)
print("D. ANOMALIES.md TFT reviews")
print("=" * 80)
tall = json.loads((RESULTS / "tft_all_stations_anomaly_review.json").read_text())
check("TFT review calibration rows==384", tall["calibration"]["n_forecasts"] == 384, str(tall["calibration"]["n_forecasts"]))
check("TFT review references==192", tall["calibration"]["n_references"] == 192, str(tall["calibration"]["n_references"]))
check("TFT review test rows==384", tall["test"]["n_forecasts"] == 384, str(tall["test"]["n_forecasts"]))
tft_doc = {
    "Colwick": (20, 2, 22), "Skelton": (41, 0, 41), "Roxton": (30, 22, 45), "Thorverton": (45, 41, 48),
    "Kingston": (30, 12, 38), "Bewdley": (29, 2, 29), "Caton": (2, 0, 2), "Bywell": (16, 0, 16),
}
summ_t = {NAMES[s["station_guid"]]: s for s in tall["summary_by_station"]}
for nm, (lr, oi, cb) in tft_doc.items():
    s = summ_t[nm]
    ok = s["large_residual_count"] == lr and s["outside_interval_count"] == oi and s["anomaly_count"] == cb and s["n_forecasts"] == 48
    check(f"TFT anomalies {nm} ({lr}/{oi}/{cb})", ok,
          f"got {s['large_residual_count']}/{s['outside_interval_count']}/{s['anomaly_count']}")
check("TFT pooled large==213", sum(s["large_residual_count"] for s in tall["summary_by_station"]) == 213)
check("TFT pooled outside==79", sum(s["outside_interval_count"] for s in tall["summary_by_station"]) == 79)
check("TFT pooled combined==241", sum(s["anomaly_count"] for s in tall["summary_by_station"]) == 241)

tsc = pd.read_csv(RESULTS / "tft_all_stations_anomaly_scored.csv")
check("tft_all_stations csv rows==384", len(tsc) == 384)
check("tft csv large==213", int(tsc["large_residual"].sum()) == 213)
check("tft csv outside==79", int(tsc["outside_interval"].sum()) == 79)
check("tft csv combined==241", int(tsc["anomaly"].sum()) == 241)
# TFT quantile ordering sanity
check("TFT p10<=p50<=p90 everywhere",
      bool(((tsc["p10"] <= tsc["p50"]) & (tsc["p50"] <= tsc["p90"])).all()))
# recompute pooled TFT pinball from scored csv
def pinball(df, q, col):
    diff = df["actual"] - df[col]
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
check("recomputed TFT p50 pinball==0.0377434", close(pinball(tsc, 0.5, "p50"), 0.0377434, 5e-6), f"{pinball(tsc,0.5,'p50'):.7f}")
check("recomputed TFT p90 pinball==0.0227445", close(pinball(tsc, 0.9, "p90"), 0.0227445, 5e-6), f"{pinball(tsc,0.9,'p90'):.7f}")

tk = json.loads((RESULTS / "tft_anomaly_review.json").read_text())
ks = tk["summary_by_station"][0]
ks_doc = tft_doc["Kingston"]
check("one-station TFT review matches all-station Kingston counts",
      ks["large_residual_count"] == ks_doc[0] and ks["outside_interval_count"] == ks_doc[1] and ks["anomaly_count"] == ks_doc[2],
      f"got {ks['large_residual_count']}/{ks['outside_interval_count']}/{ks['anomaly_count']}")

print()
print("=" * 80)
print("E. BENCHMARKS.md measured run vs phase1_tft_8series_epoch5_cpu_fixed.json")
print("=" * 80)
bmpath = ROOT / "data/benchmark/electricity/phase1_tft_8series_epoch5_cpu_fixed.json"
if bmpath.exists():
    bm = json.loads(bmpath.read_text())
    print("benchmark keys:", sorted(bm.keys()))
    s = json.dumps(bm)
    check("benchmark P50 2.9622848 present", "2.9622848" in s)
    check("benchmark P90 1.4711090 present", "1.4711090" in s)
    check("benchmark sha256 recorded", "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1" in s)
    check("deltas correct", close(2.9622848 - 0.055, 2.9072848, 1e-7) and close(1.4711090 - 0.027, 1.4441090, 1e-7))
else:
    skip("benchmark measured-run evidence", "phase1_tft_8series_epoch5_cpu_fixed.json missing")
zpath = ROOT / "data/benchmark/electricity/LD2011_2014.txt.zip"
if zpath.exists():
    h = hashlib.sha256(zpath.read_bytes()).hexdigest()
    check("raw UCI zip sha256 matches documented value", h == "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1", h)
else:
    skip("raw UCI zip sha256", "raw archive not tracked; re-download per BENCHMARKS.md to audit it")

print()
print("=" * 80)
print("F. DATA_QUALITY.md frozen extract + hourly preparation")
print("=" * 80)
man = json.loads((ROOT / "data/frozen/manifest.json").read_text())
print("manifest keys:", sorted(man.keys()))
entries = man.get("measures") or man.get("stations") or []
check("manifest lists 9 measures", len(entries) == 9, str(len(entries)))
shapath = ROOT / "data/frozen/checksums.sha256"
if shapath.exists():
    sha_lines = shapath.read_text().strip().splitlines()
    recorded = {line.split()[1]: line.split()[0] for line in sha_lines}
    for fn, want in sorted(recorded.items()):
        p = ROOT / "data/frozen" / fn
        if p.exists():
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            check(f"checksum {fn}", got == want, f"recorded={want} got={got}")
        else:
            skip(f"checksum {fn}", "raw extract not tracked; regenerate with python -m src.freeze")
else:
    skip("frozen checksums", "checksums.sha256 missing")
# jsonl line counts vs manifest
for e in entries:
    guid = e.get("measure_id", "").replace("-level-i-900-m-qualified", "") or e.get("station_guid", "")
    p = ROOT / "data/frozen" / f"{guid}.jsonl"
    if p.exists():
        n = sum(1 for _ in p.open())
        rows = e.get("rows") or e.get("row_count")
        nm = NAMES.get(guid, guid[:8])
        check(f"manifest row count {nm}", rows is None or n == rows, f"file_lines={n} manifest={rows}")
    else:
        skip(f"frozen file for {guid[:8]}", "not tracked; regenerate with python -m src.freeze")

doc_valid = {
    "Kingston": 17539, "Thorverton": 17544, "Bewdley": 17544, "Colwick": 17544,
    "Skelton": 14233, "Bywell": 16057, "Caton": 17538, "Hereford Bridge": 12822, "Roxton": 15184,
}
for guid, nm in NAMES.items():
    p = ROOT / "data/hourly" / f"{guid}.csv"
    if not p.exists():
        skip(f"hourly {nm}", "derived hourly CSV not tracked; regenerate with python -m src.prepare")
        continue
    df = pd.read_csv(p)
    ok_len = len(df) == 17544
    check(f"hourly {nm} rows==17544", ok_len, str(len(df)))
    if not ok_len:
        continue
    tcol = [c for c in df.columns if "time" in c.lower()][0]
    vcol = "value" if "value" in df.columns else [c for c in df.columns if c != tcol and "count" not in c.lower() and "mask" not in c.lower()][0]
    n_valid = int(df[vcol].notna().sum())
    check(f"hourly {nm} valid bins=={doc_valid[nm]}", n_valid == doc_valid[nm], f"got {n_valid} col={vcol}")
    ts = pd.to_datetime(df[tcol])
    check(f"hourly {nm} full UTC grid", ts.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
          and ts.iloc[-1] == pd.Timestamp("2025-12-31 23:00", tz="UTC")
          and (ts.diff().dropna() == pd.Timedelta(hours=1)).all())

print()
print("=" * 80)
print(f"SUMMARY: {len(issues)} FAILURES, {len(skips)} SKIPPED (untracked raw data)")
print("=" * 80)
for i in issues:
    print(" -", i)
