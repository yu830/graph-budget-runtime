# Project status

2026-10-26: **first runnable core complete; no performance adoption**.

| Claim | Result |
|---|---|
| Plan/profile, startup integration, dispatch and observations | Implemented and exercised |
| CPU checks | 57 passed, zero skipped, including clean installation/source checks |
| Model validation | 21 cases / 815 requests; all exact-token comparisons passed in the pinned deterministic mode |
| Native decode paths | Replay, 3-to-4 padding, cap overflow and forced-test eager observed |
| Strong-baseline evaluation | Five paired T/C rounds on each of three held-out loads; 2,078 requests completed without failure |
| Cost-policy value | Seven supported keys, zero measured eager switches; keep native dispatch |
| Default non-deterministic configuration | Strict token checks failed for 2/43 calibration requests; not adopted |
| Clean Git GPU replay | 43/43 exact outputs; actual plan application and missing-profile native fallback passed |
| Public push / upstream PR | Not performed |

[PROJECT_COMPLETION.json](../PROJECT_COMPLETION.json) is the delivery receipt.
The [technical report](TECHNICAL_REPORT.md) explains results and limits;
[derived results](../results/2026-10-26/README.md) link the reproducible data.
The complete original logs remain in the separate local raw-evidence archive.
No additional optimization route or background job is active.
