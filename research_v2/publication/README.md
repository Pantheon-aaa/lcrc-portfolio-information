# Published evidence and lossless records

This directory packages the completed 2026-10-08 v2 research run for public review.

- `records_index.json` maps every canonical top-level `results/*.jsonl` to ordered gzip shards, byte lengths, row counts and SHA-256 checksums.
- `records/` contains the complete original bytes, not a selected sample or rounded export. Each source is split at line boundaries before compression; source bytes and nonfinite Python JSON values are preserved.
- `local_archive_inventory.json` identifies superseded results and redundant worker partitions retained locally. Their hashes are published; their duplicate/raw payloads are not all uploaded. Canonical negative findings and open gaps **are** included in the public records.
- CSV tables, small diagnostic JSON files, figures and the theory PDF are committed in their normal locations.

From the repository root:

```bash
python -m research_v2.publication verify
python -m research_v2.publication restore
python -m research_v2.verify_delivery
```

Restoration refuses to overwrite a different existing result and verifies both compressed and uncompressed hashes. Existing matching files are retained. Restored raw JSONL is Git-ignored to avoid committing both raw and compressed versions. `pack` is for the author to build a new publication snapshot; it is not needed to review this run.

The original research-wide `manifest.json` remains local because it describes local archives and temporary rendering outputs as well. Use the public record index for the published dataset checksums and Git commit identity for source provenance. Unpublished old source versions cannot be reconstructed from hashes alone.

## Timing semantics

The protocol was created at **2026-10-08 16:03:19 Asia/Shanghai**. The last experimental ledger entry finished at **18:36:41**: approximately **2.556 elapsed hours** between those two markers.

The ledger sums **13.442 experiment-process/task wall hours** across concurrent jobs, and separately **13.268 recorded process CPU hours**. All 52 ledger entries have CPU fields. These are different quantities: eight concurrent one-hour tasks consume approximately eight summed task-wall hours while only one elapsed hour passes. Main batches used eight workers with single-thread BLAS.

These totals do not include every short interactive check, the earliest exceptional interruption, or all analysis, proof writing and publication work. The 24-hour experiment budget was defined on accumulated process/task wall time, not elapsed desktop time. The ledger includes archived/revised runs because they consumed experimental resources even though their results are not pooled in the final comparisons.

No private market data, original conversation attachments, credentials or environment secrets are part of this publication. All observations and true-risk evaluation fields in these records are synthetic.
