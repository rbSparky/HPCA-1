# Extended watchdog calibration (2026-07-29)

The earlier `paper_suite_v2_length_v2` queue was an execution calibration with
the historical 600-second per-item watchdog. Its terminal rows are preserved
unchanged and are not used as the complete baseline because several rows used
legacy or semantically rejected contracts.

The clean semantically preflighted baseline queue is
`results/paper_suite_v2_length_long_v1/run/`. It is initialized from the
immutable `results/paper_suite_v2_length_long_v1/jobs.json` job list and uses
the 13 kernel/architecture pairs that passed the constructor-level native
contract check. Each item has a 3,600-second watchdog, two attempts maximum,
15-second heartbeats, an isolated spawned worker, and atomic result
publication. The longer watchdog is an explicitly approved execution
deviation: it prevents an operational timeout from truncating a valid native
baseline while retaining process-tree termination and resumability.

The queue is run with two workers only. Each worker sets all BLAS/OpenMP and
Torch thread pools to one. No nested worker pool is used. Timeout, error and
valid mapping-failure statuses remain distinct and are never folded into one
another. This calibration is a prerequisite for complete paired FlowAdvantage
measurements; it is not itself a mapping-quality claim.

