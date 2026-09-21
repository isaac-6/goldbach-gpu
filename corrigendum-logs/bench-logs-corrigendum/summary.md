# Corrigendum measurements, arXiv:2603.07850

Four independent items. Machine: RTX 5090, sm_120, CUDA 13.3.
Nothing committed. Worktrees: `/tmp/d1v200` (throwaway branch
`measure/d1-v200-throwaway`, never to be released) and `/tmp/v200`
(`release/v2.0.1`, clean tree).

---

## Item A: D1 quantified, with a denominator

v2.0.0's `tiled_sieve_segment_kernel` extracted byte-identically into a header
so `test_gpu_sieve` can link it, defect intact. Verified as a pure move by md5
of the 62-line body against v2.0.0 `src/goldbach.cu` lines 198-259.

NOTE: the verification command in the brief does not show this. Its `sed` range
restarts at the second occurrence of the kernel name (the launch site) and pulls
in most of the worker loop, so it always reports a large spurious diff.

Launch configuration: grid = ceil(num_odds / 32768) per range, block = 256,
shared = 4096 bytes. That is v2.0.0's sizing, (TILE_ODDS/64)*sizeof(uint64_t),
a 32768-bit bitmap rather than the byte array used in later versions.

CPU comparator: `src/segmented_sieve.cpp` via `goldbach_lib`.

| Quantity | Value |
|---|---|
| odd numbers compared | 12,202,140 |
| of which prime (CPU) | 1,196,250 |
| of which composite (CPU) | 11,005,890 |
| disagreements, mean of 5 runs | 299,984 |
| composite reported PRIME | 299,984 (100%) |
| prime reported COMPOSITE | 0 |
| rate vs all odds compared | 2.4585% |
| rate vs composites | 2.7257% |

Rates are computed from the five-run mean; run 1 alone gives 2.4599% and 2.7272%.

Five runs: 300,156 / 300,167 / 300,049 / 299,833 / 299,715.
Mean 299,984, sd 201.5, spread 452 (0.07%).

Every disagreement is one-directional, a composite left marked prime. That is
the signature of lost clears and the direction that makes verification wrongly
succeed.

Per-range, run 1, fixed ranges:

| Range | odds | composites | false primes | rate vs composites |
|---|---|---|---|---|
| [3, 1e6] | 500,000 | 421,503 | 6,243 | 1.481% |
| [999983, 2e6] | 500,010 | 429,574 | 9,653 | 2.247% |
| straddling 2^32 | 16,356 | 14,866 | 572 | 3.848% |
| [1e9, +2e5] | 100,001 | 90,396 | 2,778 | 3.073% |
| TILE_ODDS^2 boundary | 200,001 | 180,825 | 5,226 | 2.890% |
| [1e11, +4e5] | 200,001 | 184,167 | 4,145 | 2.251% |
| [1e12, +4e5] | 200,001 | 185,476 | 3,808 | 2.053% |

plus 75 randomized ranges.

CAUTION: this is ~300,000, not the 217,358 in the corrigendum, a factor of 1.38.
A disagreement count is a property of how much you compare. `test_gpu_sieve`
today uses 7 fixed and 75 randomized ranges; the earlier version used 4 fixed
and 50. Quote the rate (2.73% of composites), or state the range set with the
count.

---

## Item B: D2 boundary detail per thread count

Word-alignment fix reverted locally on `release/v2.0.1`, 250 builds per case.

| threads | odds_per_thread | mod 64 | wrong builds / 250 | wrong bits | max boundary distance |
|---|---|---|---|---|---|
| 4 | 125,000 | 8 | 23 | 33 | 51 |
| 8 | 62,500 | 36 | 33 | 54 | 51 |
| 16 | 31,250 | 18 | 87 | 164 | 55 |

Word-aligned controls: 0 wrong builds at every thread count.

The brief's "within 46 bits" is an understatement; the measured maximum is 55.

The contended region is not bounded by a single word offset, because the offset
of a boundary within its shared word varies with the boundary index. For
opt = 62500, boundary 1 at 62500 sits at word offset 36 and spans -36..+27, but
boundary 2 at 125000 sits at offset 8 and spans -8..+55. The observed maximum
came from that second boundary.

| threads | theoretical max over all boundaries | measured |
|---|---|---|
| 4 | 55 | 51 |
| 8 | 60 | 51 |
| 16 | 62 | 55 |

The safe general statement is "within 63 bit positions", since a boundary can
sit anywhere from offset 0 to 63 within its word.

Fix restored; tree clean; all thread counts PASS.

---

## Item C: profile of v2.0.1

Single GPU. This machine has one GPU, so only the single-GPU portion of the
paper's Table 3 can be replaced. The 2-GPU and 4-GPU columns cannot be
re-measured here and must stand as published or be withdrawn.

### N = 2e12 (the paper's limit). Run 360.635 s, 0 fallbacks.

| Kernel | Time % | Total | Instances | Avg |
|---|---|---|---|---|
| tiled_sieve_segment_kernel | 79.17 | 283.85 s | 5,000 | 56.77 ms |
| goldbach_phase1_kernel | 20.25 | 72.61 s | 5,000 | 14.52 ms |
| count_unverified_kernel | 0.58 | 2.09 s | 5,000 | 0.42 ms |

| Operation | Time % | Total | Count | Avg |
|---|---|---|---|---|
| [CUDA memset] | 81.8 | 536.51 ms | 10,000 | 53,650.7 ns |
| [CUDA memcpy HtoD] | 17.6 | 115.17 ms | 5,002 | 23,024.0 ns |
| [CUDA memcpy DtoH] | 0.6 | 4.20 ms | 5,000 | 839.2 ns |

Memset split (from the SQLite export, grouped by transfer size):

| Memset | Count | Size each | Total | Avg | Share |
|---|---|---|---|---|---|
| d_verified clear | 5,000 | 200 MB (one 199,999,999 B) | 535.08 ms | 107,016 ns | 99.73% |
| counter clear | 5,000 | 4 B | 1.425 ms | 285.1 ns | 0.27% |

The aggregated 53,650.7 ns is the arithmetic mean of 285 ns and 107,016 ns and
describes no operation that exists. The size column has the same defect: avg
100.000 MB with min 0.000 and max 200.000. Report the two rows separately.

### N = 1e13. Run 3216.83 s, 0 fallbacks.

| Kernel | Time % | Total | Instances | Avg |
|---|---|---|---|---|
| tiled_sieve_segment_kernel | 87.76 | 2814.62 s | 25,000 | 112.58 ms |
| goldbach_phase1_kernel | 11.91 | 382.13 s | 25,000 | 15.29 ms |
| count_unverified_kernel | 0.32 | 10.40 s | 25,000 | 0.42 ms |

| Operation | Time % | Total | Count | Avg |
|---|---|---|---|---|
| [CUDA memset] | 81.4 | 2580.77 ms | 50,000 | 51,615.3 ns |
| [CUDA memcpy HtoD] | 18.0 | 570.89 ms | 25,002 | 22,833.6 ns |
| [CUDA memcpy DtoH] | 0.7 | 20.64 ms | 25,000 | 825.8 ns |

Memset split: d_verified 24,999 x 200 MB + 1 x 199,999,999 B = 2573.80 ms
(99.73%, avg 102,952 ns); counter 25,000 x 4 B = 6.97 ms (0.27%, avg 278.7 ns).

### Sieve share, measured

| N | sieve | Phase 1 | count | kernel total |
|---|---|---|---|---|
| 2e12 | 79.17% | 20.25% | 0.58% | 358.5 s |
| 1e13 | 87.76% | 11.91% | 0.32% | 3207.1 s |

### The N * pi(sqrt(N)) account, measured rather than analytical

| | 2e12 | 1e13 | ratio |
|---|---|---|---|
| small_high | 1,414,214 | 3,162,278 | |
| primes rescanned per tile | 108,108 | 227,647 | 2.11x |
| segments | 5,000 | 25,000 | 5.00x |
| measured sieve time | 283.8 s | 2814.6 s | 9.92x |
| N * pi(sqrt(N)) prediction | | | 10.53x |
| N alone would predict | | | 5.00x |

Measured 9.92x against 10.53x predicted by N * pi(sqrt(N)), a 6% overshoot,
versus 5.00x for N alone, which is wrong by a factor of two. The account holds.

---

## Item D: matched cost of the correction

Same machine, same parameters, five runs each. Fixed column is the v2.0.1
figures measured previously.

| N | unfixed v2.0.0 | sd | v2.0.1 fixed | ratio |
|---|---|---|---|---|
| 1e10 | 0.4472 s | 0.0118 | 1.0600 s | 2.37x |
| 1e11 | 3.7961 s | 0.0063 | 11.2770 s | 2.97x |
| 1e12 | 42.3643 s | 0.0298 | 144.9510 s | 3.42x |

The correction replaces an unsynchronised read-modify-write with atomicAnd. The
measured cost is 2.4x to 3.4x and grows with N. The mechanism behind that cost
was not isolated.

THESE TIMINGS DO NOT MEASURE A VALID COMPUTATION. The unfixed binary leaves
2.73% of composites marked prime (Item A). These figures account for the change
in the reported timings; they are not a performance baseline.

---

## Item E: D3 observed, segment size above 2^32

Source: `itemE_d3_segsize.log`. v2.0.1 (commit 536ca1b), sm_120 build.

```
./bin/goldbach 9000000002 --seg-size=4500000000 --p-small=1000000 --batch-size=2000000
```

One segment: the run reports `Checking range : [4, 9000000002]` and
`Total numbers  : 4500000000`. It printed "All even numbers up to 9000000002
satisfy Goldbach.", 0 Phase 2 fallbacks, 1.09369 s, exit 0. The log notes the
same verdict from a default build, where `CMAKE_CUDA_ARCHITECTURES=native`
resolves to 75 on this device and the kernels run from JIT-compiled PTX.

The success is the point. With a segment of 4,500,000,000 the offsets at or
above 2^32 number 4,500,000,000 - 4,294,967,296 = 205,032,704, and 32-bit
thread-index arithmetic wraps, so none of them is ever assigned a thread. Those
even numbers were not checked, and the run still reported success. That count
follows from the arithmetic in the source; it is not measured by this run, which
emits no per-offset record. Captured before the startup guard added in v2.0.2.

---

## Item F: architecture differential, sm_120 against sm_75

Source: `itemF_arch_differential.log`. Same tree (release/v2.0.2 working tree),
built twice: `goldbach.1.sm_120.cubin` and, from a plain
`-DCMAKE_BUILD_TYPE=Release`, `goldbach.1.sm_75.cubin`. Phase 2 fallbacks is a
deterministic integer that depends only on sieve correctness, so the two builds
must agree exactly.

| Config | LIMIT | --p-small | fallbacks, sm_120 | fallbacks, sm_75 |
|---|---|---|---|---|
| T1 | 10^7 | 100 | 186,899 | 186,899 |
| T2 | 10^8 | 1000 | 3 | 3 |
| T3 | 1000040000000, --start=10^12 | 1000 | 894 | 894 |

Three runs per build per configuration; identical on every run. Independent
cross-check, `cpu_goldbach 10000000`: 4,999,999 even numbers checked, 0
failures. Register allocation differs between the builds, as expected from
separate code generation:

| Kernel | REG, sm_120 | REG, sm_75 |
|---|---|---|
| `goldbach_phase1_kernel` | 52 | 43 |
| `tiled_sieve_segment_kernel` | 34 | 38 |
| `count_unverified_kernel` | 8 | 6 |

The sm_75 build is the faster of the two at 10^10. That measurement is in
`itemF_arch_1e10.log`: five runs per build of
`goldbach 10000000000 --seg-size=200000000 --p-small=1000000 --batch-size=2000000`,
same tree, 0 Phase 2 fallbacks on all ten.

| Build | mean | sd (n-1) | runs |
|---|---|---|---|
| sm_120 | 1.09740 s | 0.01224 | 1.11439, 1.09352, 1.08692, 1.10563, 1.08652 |
| sm_75 | 0.58475 s | 0.01813 | 0.611155, 0.584806, 0.591888, 0.569132, 0.566784 |

The ratio is 1.88x, confirming the 1.9x that `itemF_arch_differential.log`
refers to; that log also records that the ratio does not reproduce at T3, making
it a function of total sieve work rather than of base-prime list size. The
reason is not established. It does not matter for the record: no reported figure
in this corrigendum depends on the sm_75 build. Every figure comes from an
sm_120 build, and the two builds agree on every value they were asked about.

---

## Item G: differential test of the corrected D1 kernel

Item A ran the differential harness only against the defective v2.0.0 kernel.
Item G runs the same harness against the corrected v2.0.2 kernel, with a control
run against the defective kernel in the same build.

The harness is byte-identical to Item A's:
`test_gpu_sieve.cu`, md5 `b89a71af991003c20b65eb58b10372d0`. Each build gets its
own directory holding its own `sieve_kernel.cuh`, so the harness source does not
change between them.

Provenance of the kernel under test. `itemA_source/sieve_kernel_fixed.cuh` is
Item A's archived header with the kernel body replaced by v2.0.2's. Lines 14-79
of it and lines 200-265 of v2.0.2 `src/goldbach.cu` both hash to
`c4e0fe44497ea2a935003db97ef62b10`: the kernel under test is byte-identical to
the kernel in the tagged release. Diffed against Item A's extraction it shows one
hunk, the D1 fix itself, `sh_tile[...] &= ~(1ULL << ...)` replaced by an
`atomicAnd` over the same word plus a three-line comment. Nothing else differs.

Build, machine A, both variants:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 \
     -I include -I /tmp/d1/$v \
     /tmp/d1/$v/test_gpu_sieve.cu src/segmented_sieve.cpp src/prime_bitset.cpp \
     -Xcompiler -fopenmp -lgomp \
     -o /tmp/d1/$v/test_gpu_sieve
```

`cuobjdump --list-elf` reports `test_gpu_sieve.1.sm_120.cubin` and
`test_gpu_sieve.2.sm_120.cubin` for both.

Control, defective kernel, same build (`itemG_d1_control_defective.log`): FAIL,
exit 1, 300,517 disagreements, all of them composite reported prime, none the
other way.

Corrected kernel, five runs (`itemG_d1_fixed_run1.log` to `run5.log`): PASS,
exit 0, zero disagreements in both directions on every run.

| Quantity | Control, defective | Corrected, all five runs |
|---|---|---|
| odd numbers compared | 12,202,140 | 12,202,140 |
| of which prime (CPU) | 1,196,250 | 1,196,250 |
| of which composite (CPU) | 11,005,890 | 11,005,890 |
| composite reported PRIME | 300,517 | 0 |
| prime reported COMPOSITE | 0 | 0 |
| verdict | FAIL | PASS |

The denominators are identical across all six runs, and identical to Item A's:
the corrected kernel was asked about the same numbers, not fewer.

Two notes for anyone reading the logs. The harness banner is hardcoded and
reads `=== D1 QUANTIFICATION (v2.0.0 kernel, defect intact) ===` in every log
including the five corrected-kernel runs; the header md5 on each log's prefix
line identifies which kernel actually ran, and the fixed logs carry an explicit
note to that effect. And the control's 300,517 lies slightly above the range of
Item A's five runs, 299,715 to 300,167, which is what a nondeterministic race
recompiled into a fresh build should do: the rate against composites is 2.7305%
here against Item A's 2.7257% mean, a difference in the fourth significant
figure.
