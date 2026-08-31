# Environment Agency Hydrology API fit report

Probe date: 2026-08-27. All API results below came from unauthenticated HTTPS
requests made on that date. Counts are a live snapshot and can change as the
Environment Agency revises or extends the archive.

## Executive finding

The hydrology API is suitable for a multi-station TFT training study, subject
to a frozen extract and explicit quality, time-zone, and datum handling. The
nine sampled stations each have a multi-decade 15-minute qualified level series.
The common two-calendar-year probe was timestamp-complete at seven stations and
had only five or seven missing timestamps at two stations, but the value and
quality flags show material station-to-station differences. This supports
`FIT WITH CONDITIONS`, not an assumption that every returned row is usable.

## 1. Which API does what?

Both EA services are needed for the proposed workflow:

| Workflow | Use this service | Decision | Reason |
| --- | --- | --- | --- |
| Bulk history and frozen training data | [Hydrology archive reference](https://environment.data.gov.uk/hydrology/doc/reference), using `/hydrology/id/measures/{measure-id}/readings` or the asynchronous batch-readings endpoint | **Use the hydrology API** | It exposes historic and recent hydrological series, `earliest`/`latest`, date filters, quality fields, and station/measure metadata. |
| Live incremental collector | [Flood-monitoring reference](https://environment.data.gov.uk/flood-monitoring/doc/reference), using `/flood-monitoring/data/readings?latest` every 15 minutes plus `since` catch-up | **Use the flood-monitoring API** | It is the real-time/near-real-time feed and documents the cached `latest` pattern. |

The services are not interchangeable. In a live check, a hydrology qualified
measure returned `2026-08-16T06:45:00`, while the flood-monitoring feed returned
current readings on 2026-08-27. The archive is therefore appropriate for a
frozen historical dataset but must not be treated as the low-latency collector.
The live endpoint's `latest` response is also not a complete historical trace;
use station readings and `since` for catch-up.

## 2. Access, formats, limits, and licence

### Access and formats

- Direct GETs to station, measure, and readings URLs returned HTTP 200 without a
  token, key, or registration. The EA support FAQ also states that registration
  and personal information are not needed for the service or associated APIs:
  [FAQ](https://environment.data.gov.uk/support/faqs/275811163/275811229).
  The published [OpenAPI document](https://environment.data.gov.uk/hydrology/doc/oas.json)
  has no security scheme. This confirms access for the calls made here; an
  institutional proxy or future service change could still impose its own
  controls.
- JSON and CSV were both confirmed. A `.json` or `.csv` suffix works, and the
  bare endpoint also negotiates with `Accept: application/json` or
  `Accept: text/csv`. JSON responses contain `meta` and `items`; CSV readings
  included `measure,dateTime,date,value,completeness,quality,qcode`.
- The archive documentation also lists HTML, RDF, and GeoJSON representations,
  but JSON/CSV are the formats used by the probe and training pipeline.

### Pagination and row limits

- Collection endpoints use `_limit` and `_offset`; the response `meta` echoed
  the applied values in live tests. The station-list default is 100 rows.
- The readings documentation describes a 100,000-row soft limit per request,
  an override via `_limit`, and a 2,000,000-row hard limit. A 175,392-row
  five-year 15-minute window was retrieved as two 100,000-row pages by
  `probe_ea_hydrology.py`. The probe stops only after a short page and reports
  if it reaches its page guard.
- Larger extracts should use the asynchronous
  [`/hydrology/data/batch-readings/batch`](https://environment.data.gov.uk/hydrology/doc/reference#batch-api)
  endpoint, which returns a queued job and a CSV result URL. Do not make a
  single request assuming that a long range fits under the soft limit.
- `_limit=0` was an unsafe edge case in a live station-list check: it returned
  9,536 items rather than an empty page. The probe rejects zero and out-of-range
  limits.

### Rate/concurrency behaviour

The hydrology documentation asks bulk users to keep non-batch requests to one
in flight and reserves the right to enforce concurrency limits or block
excessive demand. **[UNVERIFIED]** Neither the hydrology nor flood-monitoring
reference page publishes a numeric requests-per-minute or daily quota, and no
rate-limit headers appeared in the sampled responses. The probe is deliberately
serial and pauses between stations.

### Licence and attribution

Live hydrology JSON metadata identified the publisher as Environment Agency and
returned the [Open Government Licence v3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
URL (`licenseName: OGL 3`). The hydrology page footer says OGL v3.0 except where
otherwise stated. The standard acknowledgement wording is:

> Contains public sector information licensed under the Open Government Licence v3.0.

The hydrology reference did not show a separate hydrology-specific attribution
sentence in the live page. **[UNVERIFIED]** Confirm with the dataset owner if a
future download supplies an additional notice. For the separate live collector,
the flood-monitoring page specifies this wording:

> this uses Environment Agency flood and river level data from the real-time data API (Beta)

Keep the applicable acknowledgement, source link, extraction date, and licence
metadata alongside a frozen dataset.

## 3. Inventory and identifiers

Live `_count=@id` queries returned the following current counts. Property counts
overlap because one station can expose several properties.

| Item | Stations | Measures | What it represents |
| --- | ---: | ---: | --- |
| All hydrology stations | 9,537 | - | Current station catalogue snapshot; the reference page describes the service as having "nearly 8,000" stations. |
| Water level | 2,638 | 7,914 | 15-minute instantaneous levels plus daily min/max series. |
| Water flow | 1,103 | 4,412 | 15-minute instantaneous flow and daily mean/min/max, in m3/s. |
| Rainfall | 995 | 1,990 | 15-minute totals and daily totals, in mm. |
| Groundwater level | 3,611 | 4,169 | Dipped/logged groundwater; metadata uses mAOD for the sampled groundwater series. |
| Active water-level stations | 2,551 | - | Water-level station query filtered by `status.label=Active`. |

The API also documents water-quality measures (for example dissolved oxygen,
turbidity, conductivity, temperature, ammonium, nitrate, and pH). Groundwater
logged sub-daily series may be mixed 15-minute/hourly rather than guaranteed to
be 15-minute, so period metadata must be checked rather than inferred from the
property name.

The join is explicit and should be retained in the frozen metadata:

1. A hydrology station has a primary SUID/GUID and a station URI such as
   `/hydrology/id/stations/{stationGuid}`. Some colocated SUIDs are not globally
   unique, so retain the complete station notation and any disambiguator.
2. The station's `measures` array contains measure URIs. The selected canonical
   level measure has the form
   `{stationGuid}-level-i-900-m-qualified`; its nested station points back to the
   station SUID and exposes `stationReference`, WISKI, and RLOI identifiers where
   available.
3. The station metadata can include `sameAs` pointing to the corresponding
   flood-monitoring station. Join the two APIs through this explicit
   `sameAs`/stationReference/WISKI mapping. Do **not** assume a flood station
   reference such as `45118` is the hydrology GUID.

The probe filters the archive by the hydrology station GUID and discovers the
qualified 15-minute level measure from station metadata. A production extractor
should persist the complete station and measure JSON, not only the readable
label.

## 4. History depth and completeness

### Probe design

`probe_ea_hydrology.py` was run against nine geographically spread named rivers
with the half-open window `2024-01-01 <= dateTime < 2026-01-01`. Each selected
measure reported `period=900`, `periodName=15min`, `valueType=instantaneous`,
`unitName=m`, `observationType=Qualified`, and `hasTelemetry=true`. The window
contains 70,176 expected 15-minute slots (96 per day, including the 2024 leap
day). Archive timestamps currently have no `Z` or offset suffix; see the time
zone caveat in section 5.

`earliest` and `latest` were queried separately for each measure. The earliest
reading can be much later than `dateOpened` (for example, Kingston opened in
1883 but its sampled qualified series starts in 1986), so the reading result,
not the opening date, is the history-depth measure.

| Station and river | Hydrology station GUID | Qualified 15-minute measure | Earliest reading | Latest returned | Span to latest |
| --- | --- | --- | --- | --- | ---: |
| Kingston - River Thames | `8496ce69-482c-406a-a2f0-ac418ef8f099` | `8496ce69-482c-406a-a2f0-ac418ef8f099-level-i-900-m-qualified` | 1986-11-02 01:15 | 2026-08-27 18:30 | 39.82 y |
| Thorverton - River Exe | `3c4d4f78-2d0e-474a-b884-65a9daca18fb` | `3c4d4f78-2d0e-474a-b884-65a9daca18fb-level-i-900-m-qualified` | 1956-05-01 01:00 | 2026-08-27 18:30 | 70.33 y |
| Bewdley - River Severn | `8820d897-a09e-4857-8095-5834fee6962f` | `8820d897-a09e-4857-8095-5834fee6962f-level-i-900-m-qualified` | 1970-01-01 16:15 | 2026-08-27 18:30 | 56.65 y |
| Colwick - River Trent | `0dcf81cb-5305-4e0b-b150-9b733ac44d0b` | `0dcf81cb-5305-4e0b-b150-9b733ac44d0b-level-i-900-m-qualified` | 1964-12-30 09:00 | 2026-08-27 18:30 | 61.66 y |
| Skelton - River Ouse | `213d70b2-894b-406b-9dc3-31d3ccec7f54` | `213d70b2-894b-406b-9dc3-31d3ccec7f54-level-i-900-m-qualified` | 1969-09-18 11:30 | 2026-08-27 18:30 | 56.94 y |
| Caton - River Lune | `9ad5d28c-7cfe-46db-b39d-58701689cd59` | `9ad5d28c-7cfe-46db-b39d-58701689cd59-level-i-900-m-qualified` | 1979-01-01 09:00 | 2026-08-27 18:30 | 47.65 y |
| Bywell - River Tyne | `e786e60f-a0f1-4955-aa57-f22ba39c7427` | `e786e60f-a0f1-4955-aa57-f22ba39c7427-level-i-900-m-qualified` | 1964-12-07 13:00 | 2026-08-27 18:30 | 61.72 y |
| Hereford Bridge - River Wye | `30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9` | `30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9-level-i-900-m-qualified` | 1988-07-01 09:00 | 2026-08-27 18:30 | 38.16 y |
| Roxton - River Great Ouse | `3c43b72d-03a7-46e1-86c7-76dd97808544` | `3c43b72d-03a7-46e1-86c7-76dd97808544-level-i-900-m-qualified` | 1972-10-23 14:15 | 2026-08-27 18:30 | 53.84 y |

Therefore, multi-year series exist for every sampled station. The verified
2024-2025 span has 17,544 hourly bins per station (2024 is a leap year); after a
168-hour encoder and 24-hour horizon there are roughly 17,353 possible rolling windows before quality
filtering. That is enough to form chronological train/validation/test splits for
an initial study, although only two seasonal cycles is a modest basis for
seasonality claims. Extending backward is feasible through paging or batch and
should be re-probed after extraction.

### Two-year quality/completeness result

`timestamp gaps` counts absent 15-minute grid slots. `missing values` counts
rows present in the timestamp grid whose `value` was null/omitted. `usable
coverage` is the percentage of expected slots with both a timestamp and a
non-missing value; it is not a claim that a value marked Estimated, Suspect, or
Unchecked is scientifically correct.

| Station | Rows / expected | Timestamp gaps | Missing values | Usable coverage | Max observed step | Quality counts | `qcode=Edited` |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Kingston | 70,176 / 70,176 | 0 (0.000%) | 18 (0.026%) | 99.974% | 0.25 h | Good 70,157; Estimated 1; Missing 18 | 28 |
| Thorverton | 70,176 / 70,176 | 0 (0.000%) | 0 | 100.000% | 0.25 h | Good 68,639; Estimated 1,537 | 12,110 |
| Bewdley | 70,176 / 70,176 | 0 (0.000%) | 0 | 100.000% | 0.25 h | Good 70,176 | 10,949 |
| Colwick | 70,176 / 70,176 | 0 (0.000%) | 0 | 100.000% | 0.25 h | Good 70,176 | 12,343 |
| Skelton | 70,169 / 70,176 | 7 (0.010%) | 0 | 99.990% | 0.75 h | Good 56,933; Unchecked 13,236 | 4 |
| Caton | 70,176 / 70,176 | 0 (0.000%) | 25 (0.036%) | 99.964% | 0.25 h | Estimated 46,252; Good 23,898; Unchecked 1; Missing 25 | 46,956 |
| Bywell | 70,171 / 70,176 | 5 (0.007%) | 0 | 99.993% | 0.75 h | Good 64,088; Unchecked 5,943; Estimated 140 | 22 |
| Hereford Bridge | 70,176 / 70,176 | 0 (0.000%) | 8,693 (12.387%) | 87.613% | 0.25 h | Good 51,297; Unchecked 10,024; Suspect 162; Missing 8,693 | 18,017 |
| Roxton | 70,176 / 70,176 | 0 (0.000%) | 0 | 100.000% | 0.25 h | Good 60,743; Suspect 9,433 | 42,915 |

No duplicate timestamp was observed in this window for any of the nine
stations. This is a window result, not proof that the entire multi-decade
archive is duplicate-free. The `completeness` field was blank for all sampled
instantaneous rows even though it is present in the response schema; blank must
not be interpreted as complete.

## 5. Data-quality traps and controls

- **Missing and outages.** Missing-value rows are explicitly returned with no
  value (18 at Kingston, 25 at Caton, and 8,693 at Hereford Bridge). Skelton and
  Bywell each had small timestamp holes and a 0.75-hour maximum observed gap.
  All nine station records currently report Active, but status is metadata, not
  proof that every historical interval is available. Keep an outage/gap mask and
  avoid imputation before chronological splitting.
- **Revisions.** `qcode=Edited` occurred at every station, from 4 rows at Skelton
  to 46,956 at Caton. This is evidence that values can be revised. No revision
  timestamp or prior-value history was returned, so a reproducible dataset must
  freeze the extraction and retain the original `qcode`. A later re-download may
  differ.
- **Quality flags.** The selected measure is `observationType=Qualified`, but
  returned row quality still included Good, Estimated, Suspect, Unchecked, and
  Missing. Decide in advance whether the TFT target uses Good-only rows, allows
  Estimated rows with a flag covariate, or treats Suspect/Unchecked as masked.
  The reference overview calls the field `qflag` in one place while live JSON/CSV
  uses `qcode`; a parser should accept both names. **[UNVERIFIED]** The complete
  historical meaning and revision policy for every qcode is not recoverable from
  these sampled responses alone.
- **Datum and level scale.** The sampled hydrology station metadata returned
  null/absent `datum` and `datumType` fields, while the selected level measures
  all reported unit `m`. A live cross-check of corresponding flood-monitoring
  measures showed real variation between `mAOD`, `mASD`, and plain `m`, with
  station stage-scale metadata. Do not concatenate raw levels across stations
  until each measure's vertical datum/scale is mapped. **[UNVERIFIED]** Whether
  every historic datum or scale change is represented as a machine-readable
  event in the hydrology archive is not confirmed.
- **Units and measures.** Level is metres, flow is m3/s, rainfall is mm, and
  groundwater commonly uses mAOD. Never select a series by label alone; persist
  `parameter`, `period`, `valueType`, `unitName`, `observationType`, and
  `qualifier` and validate them on every extraction.
- **Time zones and daylight saving.** The archive's returned `dateTime` strings
  were naive (for example `2024-01-01T00:00:00`), while the documentation asks
  clients to include a timezone in filters and the flood feed returns explicit
  UTC timestamps. **[UNVERIFIED]** The archive's authoritative timezone
  convention is not established by these responses. Confirm it with EA, convert
  to an explicit UTC axis, and use the same axis for hourly aggregation, joins,
  and rolling-origin splits.
- **Off-grid readings.** The main nine-station window was on a 15-minute grid,
  but a separate five-year probe found a Skelton reading at `15:15:28`. Reject
  or explicitly resample off-grid records rather than silently rounding them.
- **Mutable/current data.** The archive can lag the real-time service and is
  revised. Record request URLs, retrieval timestamp, API version (`2.1.1` in the
  sampled metadata), station/measure metadata, row counts, and a checksum for a
  frozen training extract. This study is not a safety-critical warning system.

## Recommended frozen study configuration

For a first reproducible dataset, use the nine IDs in the history table, with a
core quality set of Kingston, Thorverton, Bewdley, Colwick, Skelton, and Bywell.
Retain Caton, Hereford Bridge, and Roxton as explicitly flagged stress-test
stations rather than dropping them silently; Hereford's 12.387% missing-value
rate and Roxton's Suspect-heavy window need a documented inclusion rule.

Use the qualified instantaneous water-level measure
`{stationGuid}-level-i-900-m-qualified` and preserve raw 15-minute rows and all
quality fields. For the TFT input, aggregate to **hourly** bins (four
15-minute observations per bin, with a predeclared minimum-valid-count rule)
after time-zone normalization. Hourly matches the scale of the published TFT
electricity experiment and turns a 168-hour encoder plus 24-hour horizon into
168 plus 24 steps instead of 672 plus 96. Daily aggregation would discard the
short river responses that matter to a 24-hour horizon; raw 15-minute modelling
is possible but substantially longer and more expensive. Keep the raw series so
15-minute sensitivity checks remain possible.

The fully checked frozen range is `2024-01-01 <= t < 2026-01-01`. It supplies two
complete calendar years for train/validation/test rolling-origin splits. A
longer target such as `2020-01-01 <= t < 2026-01-01` is feasible for the core
stations, but full six-year completeness and quality numbers for every listed
station were not established in this probe. **[UNVERIFIED]** Extend backward in
annual pages or a batch job only after rerunning the quality checks and freezing
the resulting files.

## README blockers/caveats to carry forward

1. Hydrology is the bulk/history source; flood-monitoring is the 15-minute live
   incremental source. Keep their IDs and timestamp conventions in an explicit
   crosswalk.
2. Freeze the archive extract, metadata, API version, retrieval time, request
   parameters, and checksum. The service can revise rows and its current archive
   may lag the live feed.
3. Enforce serial requests, paginate at `_limit`/`_offset`, and use batch jobs
   for large ranges. There is no published numeric rate quota [UNVERIFIED].
4. Normalize timestamps to an explicit UTC convention [UNVERIFIED until EA
   confirms the archive's naive timestamp semantics].
5. Handle Missing/Estimated/Suspect/Unchecked values, off-grid timestamps,
   duplicate checks, and `qcode=Edited` as first-class features/masks; do not
   assume a blank `completeness` field means complete.
6. Resolve station-specific datum/level-scale metadata before cross-station
   modelling; retain station identity or per-station normalization.
7. Include the OGL v3.0 acknowledgement and source link in every published
   dataset or result. This is research data handling, not a safety-critical
   flood warning or operational decision service.

FIT VERDICT: FIT WITH CONDITIONS

- Recommended stations (IDs + names): core `8496ce69-482c-406a-a2f0-ac418ef8f099` (Kingston), `3c4d4f78-2d0e-474a-b884-65a9daca18fb` (Thorverton), `8820d897-a09e-4857-8095-5834fee6962f` (Bewdley), `0dcf81cb-5305-4e0b-b150-9b733ac44d0b` (Colwick), `213d70b2-894b-406b-9dc3-31d3ccec7f54` (Skelton), and `e786e60f-a0f1-4955-aa57-f22ba39c7427` (Bywell). Flagged stress-test stations: `9ad5d28c-7cfe-46db-b39d-58701689cd59` (Caton), `30b20164-0eb7-49cf-b8fc-5b8e7ef6caf9` (Hereford Bridge), and `3c43b72d-03a7-46e1-86c7-76dd97808544` (Roxton).
- Recommended measure/date/resolution: each station's `...-level-i-900-m-qualified` water-level measure; frozen verified range `2024-01-01 <= t < 2026-01-01`; preserve raw 15-minute data, model on hourly aggregates after explicit timezone and datum handling.
- README blockers/caveats: mutable/revised archive; hydrology/live-feed latency split; 100k soft and 2m hard row limits; serial/fair-use access; quality/missing/off-grid/revision flags; station-specific datum and unit scales; archive timestamp timezone [UNVERIFIED]; OGL attribution; and no safety-critical use.
