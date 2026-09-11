# GoldbachGPU v2.0.1 benchmark summary

Corrected performance figures for a corrigendum to arXiv:2603.07850.

Base `v2.0.0` (the release the paper describes) plus correctness fixes only.
Branch `release/v2.0.1`, built in a git worktree at /tmp/v200; `main` was never
checked out or modified. Full environment in `environment.txt`, raw per-run
output in `ladder_*.log`, machine-readable data in `ladder.csv`.

Parameters: `--seg-size=200000000 --p-small=1000000 --batch-size=2000000`.
RTX 5090, sm_120, CUDA 13.3.

## Fixes applied

| # | Fix | Location |
|---|---|---|
| 1 | `atomicAnd` for shared-memory marking | `tiled_sieve_segment_kernel`, `src/goldbach.cu` |
| 2 | Word-aligned OpenMP slices | `build_prime_bitset`, `src/prime_bitset.cpp` |
| 3 | *Not applicable* | v2.0.0 predates BPSW, so no strong Lucas test exists to widen |

No optimisation from v3.0.0 is present: byte-wide marking, transposed Phase 1,
bitset `d_verified`, the large-prime split, `SPLIT_THRESHOLD`, `TILE_ODDS`
changes, thread-index widening, VRAM validation and `--record-check` are all absent.

## Ladder

| N | runs (computation, s) | mean | sd | rsd | wall mean | x prev |
|---|---|---|---|---|---|---|
| 10^10 | 1.04, 1.03, 1.08, 1.07, 1.07 | **1.060** | 0.022 | 2.04% | 1.41 | — |
| 10^11 | 11.42, 11.33, 11.29, 11.18, 11.16 | **11.277** | 0.107 | 0.95% | 11.62 | 10.64x |
| 10^12 | 145.62, 145.27, 144.48, 144.45, 144.93 | **144.951** | 0.506 | 0.35% | 145.30 | 12.85x |
| 10^13 | 3248.00, 3240.88, 3248.05 | **3245.643** | 4.125 | 0.13% | 3245.99 | 22.39x |

Five runs at 10^10, 10^11 and 10^12; three at 10^13. Sample standard deviation (n-1).
**Zero Phase 2 fallbacks in all 18 runs, and every run verified its full range.**

## Scaling is strongly superlinear

Per-decade ratios are 10.64x, 12.85x and 22.39x, so the cost per decade grows
rather than staying flat. The mechanism is the size of the prime list the sieve
rescans for every tile, `small_high = max(sqrt(N), p_small)`:

| N | small_high | primes scanned per tile |
|---|---|---|
| 10^10 | 1,000,001 | 78,498 |
| 10^11 | 1,000,001 | 78,498 |
| 10^12 | 1,000,001 | 78,498 |
| 10^13 | 3,162,279 | **227,647** |

The published architecture scans the entire list in every tile, at three 64-bit
divisions per prime, whether or not that prime marks anything in the tile. Tile
count grows with N, so sieve cost goes as N x pi(sqrt(N)) rather than N. At 10^13
the list grows 2.90x at once, which alone predicts a 29x decade ratio if the sieve
were all of the runtime; the observed 22.39x implies the sieve is roughly 65% of
runtime at 10^12.

Between 10^11 and 10^12 the list does not change size, but the `p*p <= q_high`
guard stops rejecting primes once sqrt(N) reaches `p_small`: at 10^11 only 27,293
of the 78,498 primes mark anything, at 10^12 all of them do. That accounts for the
12.85x rather than 10x at that step.

This is the inefficiency the v3.0.0 large-prime split was written to remove, so
its presence here is expected for the published architecture.

## Note on the 10^13 expectation

The row was expected at 1500 to 1900 s per run and measured **3245.6 s**, roughly
1.7x to 2.2x higher. The measurement is reproducible, with a 7 s spread across
three runs (0.13% relative), the GPU held 100% utilisation at 2790 MHz and 64 C
with no throttling, and no other process shared the device. The expectation
appears to have been a near-linear extrapolation from 10^12, which this
architecture does not obey.

## Tests

| Test | Result |
|---|---|
| `test_bitset_race` (ported) | PASS at 4, 8 and 16 threads |
| `test_gpu_sieve` | Not ported. `tiled_sieve_segment_kernel` is defined inside `src/goldbach.cu` next to `main()`, so linking a separate test needs it extracted to a header, which is a structural change. |
