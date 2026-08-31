# Data quality and frozen extract contract

This project uses the Environment Agency hydrology archive for bulk history and
the flood-monitoring API only for live increments. The source investigation and
live 2024-2025 probe are documented in
[`EA_HYDROLOGY_FIT_REPORT.md`](EA_HYDROLOGY_FIT_REPORT.md).

## Frozen extract

Phase 2 uses the nine explicitly configured qualified 15-minute water-level
measures in [`configs/stations.yml`](configs/stations.yml), with the half-open
window:

```text
2024-01-01 <= dateTime < 2026-01-01
```

Run a no-write plan first:

```text
python -m src.freeze --dry-run
```

To create raw JSONL files plus `manifest.json` and `checksums.sha256`:

```text
python -m src.freeze --output-dir data/frozen --format jsonl
```

Requests are serial and paged with `_limit`/`_offset`; the extractor stops at a
short page and refuses limits above the documented 2,000,000-row hard cap. Use
the archive batch endpoint for larger ranges. Raw response objects and quality
fields are preserved. The generated data files are intentionally ignored by
the source-control rules; keep the manifest, checksums, request URLs, API
version, and retrieval time with any distributed copy.

## Quality rules

The declared modelling rule is:

- retain raw `quality`, `qcode`/`qflag`, and `completeness` values;
- include `Good` and `Estimated` observations in the default target mask;
- mask `Suspect`, `Unchecked`, `Missing`, blank, and unknown quality values;
- reject duplicate timestamps and off-grid observations rather than rounding;
- do not impute before chronological train/validation/test splitting;
- require at least three valid 15-minute observations in each four-observation
  hourly bin; retain the valid count and quality mask next to the mean;
- resolve the archive's naive timestamp convention before assigning a timezone.

`src.aggregate.prepare_level_frame` enforces the explicit timezone requirement,
quality mask, duplicate check, and 15-minute grid check. It deliberately raises
if a caller passes naive timestamps without `source_timezone`. `aggregate_hourly`
does not fill invalid bins.

## Verified probe snapshot

The 2024-2025 live probe expected 70,176 slots per station. No duplicate
timestamps were observed. Timestamp gaps were seven at Skelton and five at
Bywell; the other stations had no timestamp gaps in this window. Effective
non-missing coverage was 99.974% at Kingston, 99.964% at Caton, 87.613% at
Hereford Bridge, and 100% for the other six stations (before masking quality
categories). Quality and revision counts are in the source report and are not
replaced by a blank `completeness` field.

This is a live snapshot, not a claim that the multi-decade archive is free of
duplicates, outages, revisions, off-grid rows, or datum changes. A new freeze
must rerun these checks and record its own counts. The archive returns level
units as `m` for the selected series, but cross-station level scales can differ
(`mAOD`, `mASD`, or unspecified `m` in the corresponding service metadata).
Use station-specific normalization until the datum mapping is explicit.

## Hourly preparation check

The frozen files were passed through `python -m src.prepare --source-timezone
UTC` as an explicit pipeline test. Each output contains 17,544 hourly bins for
2024-2025. The default `Good`/`Estimated` and minimum-three-valid-quarter-hours
rule produced these valid-bin counts: Kingston 17,539; Thorverton 17,544;
Bewdley 17,544; Colwick 17,544; Skelton 14,233; Bywell 16,057; Caton 17,538;
Hereford Bridge 12,822; Roxton 15,184. The UTC argument is a declared pipeline
assumption for this run, not confirmation of the archive's authoritative
timezone; keep that caveat in any published analysis.

## Licence and use boundary

Every raw extract and derived output must retain:

> Contains public sector information licensed under the Open Government Licence v3.0.

This is a retrospective research and decision-support dataset. It is not a
flood-warning service and must not be used for safety-critical decisions.
