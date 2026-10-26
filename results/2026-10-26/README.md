# Executed results, 2026-10-26

- `summary.json`: five paired T/C rounds per held-out workload and five A/A
  boundary-load pairs; differences are paired percentages, not speedup claims.
- `plan.json` and `profile.json`: the measured, hardware/source-bound plan and
  seven supported cost keys. No key supports an eager switch beyond noise.
- `validation.json`: exact-token comparisons and observed decode-path coverage.
- `performance.json`: compact derived metrics/path evidence. Original request
  stream events remain in the separate local raw-evidence archive.
- `RAW_MANIFEST.json`: SHA-256 values of original files relative to the executed
  run root. These refer to the original bytes, not the path-redacted exports.

`${RUN}` replaces the private run directory in exported locators. Numeric
measurements, request identities, and source/model fingerprints are preserved.
These plans are not portable to a different GPU/software/model identity; rerun
calibration rather than editing a fingerprint. Full logs and the raw archive
are local deliverables, excluded from Git.
