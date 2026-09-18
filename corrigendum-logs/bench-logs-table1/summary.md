# Rebuilt Table 1 of arXiv:2603.07850, single machine

Both sides measured on the same machine so the ratio does not mix environments.
The v2 side comes from `v2.0.1` (published v2.0.0 architecture, two concurrency
fixes, no optimisations). Raw per-run output in `v1_*.log` and `v2_*.log`,
machine-readable data in `table1.csv`.

Worktree `/tmp/v200` at branch `release/v2.0.1`, `git describe --tags` = `v2.0.1`,
commit `cca9e71`. `main` never checked out or modified.

Hardware: RTX 5090, sm_120 (both binaries confirmed), CUDA 13.3.

## Invocations

| Binary | Command |
|---|---|
| v2.0.1 | `goldbach <N> --seg-size=200000000 --p-small=1000000 --batch-size=2000000` |
| v1 legacy | `goldbach_gpu3 <N> 200000000 1000000` |

`goldbach_gpu3` takes positional arguments, not flags, and has no batch-size
concept. See "Check 2" below.

## Table

| N | Binary | runs (ms) | mean (ms) | sd | rsd | fallbacks |
|---|---|---|---|---|---|---|
| 10^9 | v1 `goldbach_gpu3` | 1226.0, 1250.3, 1213.2, 1258.8, 1283.7 | **1246.4** | 27.8 | 2.23% | 0 |
| 10^9 | v2.0.1 `goldbach` | 230.6, 191.1, 197.5, 215.6, 195.9 | **206.1** | 16.5 | 8.01% | 0 |
| 10^10 | v1 `goldbach_gpu3` | 13538.1, 13482.3, 13819.8, 13787.2, 13157.8 | **13557.0** | 268.0 | 1.98% | 0 |
| 10^10 | v2.0.1 `goldbach` | 1023.6, 1035.3, 1032.0, 1037.8, 1027.1 | **1031.2** | 5.8 | 0.56% | 0 |

Five runs each, sample standard deviation (n-1). Zero Phase 2 fallbacks in all
20 runs; every run verified its full range.

## Speedup

| N | published | recomputed, same machine |
|---|---|---|
| 10^9 | not stated as a ratio | **6.05x** |
| 10^10 | 45.6x | **13.15x** |

## Why the ratio falls from 45.6x to 13.15x

Almost entirely the v2 side. The published table implies a v2 figure of
18056.5 / 45.6 = 396.0 ms at 10^10; the corrected v2.0.1 measures 1031.2 ms,
2.6x slower. That is the cost of the correction in `tiled_sieve_segment_kernel`,
which replaces an unsynchronised read-modify-write with atomicAnd.

The v1 side moved far less: 13557.0 ms here against 18056.5 ms published,
-24.9%, consistent with different hardware.

## Note on the v1 comparator and `build_prime_bitset`

`goldbach_gpu3` links `goldbach_lib`, so building it from this branch means it
picks up the word-aligned OpenMP slice fix in `build_prime_bitset`. That is
correct and intended: the v1 comparator was affected by the same defect, and
both sides of the table should be free of it.
