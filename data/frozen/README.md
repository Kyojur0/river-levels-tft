# Frozen data directory

The checked-in study window is represented by the raw 15-minute station
extracts plus `manifest.json` and `checksums.sha256`. Raw JSONL files are
ignored by the source-control rules because they are large; regenerate them
with `python -m src.freeze --output-dir data/frozen --format jsonl` and verify
the resulting checksums before modelling. The manifest records request URLs,
retrieval time, API version, row counts, quality fields, and checksums so the
extract can be regenerated or audited.

Every redistributed extract or derived result must retain:

> Contains public sector information licensed under the Open Government Licence v3.0.
