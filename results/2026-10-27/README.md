# Reliability regression evidence — 2026-10-27

- [Current machine result](reliability-fixes.json): C1/C2/C3, CPU suites,
  six GPU smoke cases, exact outputs and real startup refusals.
- [Source CPU tests](cpu-wsl-final.txt) and [installed-wheel tests](package-tests.txt):
  80 passed each, zero skipped. Initial platform/fixture attempts remain in the
  private archive.
- [Original-file manifest](RAW_MANIFEST.json) and [archive receipt](archive-receipt.json):
  152 original files, verified by ZIP readback and SHA-256.
- [Private Drive handoff](https://drive.google.com/drive/folders/1KYqSDKNCW5RS7r3q-iMlYU4JsIUAvhCQ):
  original logs, SSE requests, native paths, telemetry, new one-candidate plan,
  malformed-profile fault fixtures, before/after reproducers and review inputs.

Runtime implementation: `07f58142d1fa4589c917da174f7a3fd9f34f46b8`.
The source-binding receipt explains the pre-commit GPU job start and verifies
all runtime modules against that implementation. The new plan is measured from
one predeclared smoke configuration; no old cost table is relabeled as calibrated.

Private raw records retain local machine paths; this public projection omits them.
The original `2026-10-26` data and root completion snapshot are unchanged.
This is a reliability regression, not a new performance evaluation.
