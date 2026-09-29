## [3.2.0] - Unreleased

### Fixed
- **False success when a segment had 2^32 or more unverified numbers.** The
  per-segment unverified count was 32-bit and wrapped to 0, so Phase 2 was
  skipped: `goldbach 8589934722 --seg-size=4294967360 --p-small=0` reported
  every even number verified after testing no prime. The count is now 64-bit,
  and the count kernel also counts verified numbers. Every segment must satisfy
  verified + unverified = its size, and its launch is checked; anything else
  ends the run with exit status 1.
- **Sieve overflow near 2^64.** The first multiple `(q_low + p - 1) / p * p`
  wrapped once q_low > 2^64 - p, so the prime was skipped and its multiples
  stayed marked prime (8,632 composites in one segment at N = 2^64 - 2^30). It
  is now computed as an offset from q_low that cannot overflow.
- **Segment counter wrap.** For N within 2*SEG*(G+1) of 2^64 the counter wrapped
  back to small numbers and the run never finished. N is now bounded (below).
- **Baillie-PSW rejected some primes (present since v2.1.0).** The Selfridge
  search for D stopped after 40 tries and called n a perfect square. Some
  primes need more. Below 2^32 there are 16, needing 43 to 49 tries:
  452980999, 505313251, 1143791191, 1272463669, 1373861479, 1582819291,
  2055693949, 2283397141, 2287905811, 2366713651, 2410622971, 3441877651,
  3703811101, 3823259311, 4131973231 and 4294405021. Device and host BPSW
  called them composite.

  Squares are now detected exactly before the search, and the search has no
  cap. Over every odd n below 2^32, BPSW now agrees with Miller-Rabin.

  The error ran one way only: a prime reported composite, never the reverse.
  It could therefore cost a Phase 2 fallback, or at worst a false "no
  partition" failure, but never a false success. `goldbach` reaches BPSW only
  under the default `--primetest=BPSW`, and only in two places:
  - `is_prime_q` in the scalar Phase 1 kernel, for q in
    (small_high, q_low), which is at most (P_SMALL, P_SMALL + 128]. Only
    segments starting in (P_SMALL + small_high, 2*P_SMALL + 128] get there.
  - Phase 2, for q > 10^8.

  No reported result depended on it. Every published run used
  `--p-small=1000000`, so the kernel path saw only q <= 1000128, and every run
  reported 0 Phase 2 fallbacks, so the host path was never taken.
  `single_check` uses its own Miller-Rabin and `big_check` uses GMP; neither
  was affected.
- **`--record-check` lost records above `--p-small`.** Numbers resolved by
  Phase 2 did not report p_min, so the maximum could be missed and non-records
  printed (n = 79,097,318, p_min = 1009, with `--p-small=1020`). Phase 2 now
  reports p_min. The flag is rejected with a `--start` other than 4, where its
  maxima are not records.
- **Parameter validation.** Rejected with exit status 1:
  - `--batch-size` of 0 (hung) or above 2^32;
  - `--p-small` below 3;
  - `--seg-size` that is 0, odd or at least 2^32;
  - N above 2^64 - 2^33, or N + 2*SEG*(G+1) at or above 2^64;
  - extra positional arguments and unknown options;
  - negative or malformed numbers, which used to wrap or abort on an uncaught
    exception.

  N = 4 and START = N are now accepted.
- **`big_check`.** `--p-max` is capped at 4e9. 2^64 - 1 used to crash, and an
  allocation failure now exits 1. q above 2^64 is reported as a probable
  prime, never as prime.
- **`single_check`.** Its kernel launch is now checked, and a counterexample
  now exits 2 instead of 0.
- Scalar Phase 1 reads `d_verified` with a relaxed atomic load, and the tiled
  sieve clears bytes with relaxed block-scope atomic stores, removing two formal
  data races. `TILE_ODDS` is checked at compile time, and the `--progress`
  thread no longer outlives a variable it read.

### Changed
- A Phase 2 failure reads "no partition with p ≤ 10^8 found for n = …": Phase 2
  searches p ≤ 10^8, so this is a search limit, not a counterexample.

### Added
- `--count-primes` and `--count-file`.
- Tests:
  - `test_bpsw_spsp`: every base-2 strong pseudoprime below 2^32 and the
    first ten published strong Lucas pseudoprimes, through BPSW and
    Miller-Rabin on device and host;
  - `test_phase2_fallback`: an exact Phase 2 count;
  - `test_record_check`: `--record-check` against brute-force records to 1e8,
    and a tie between two numbers in one segment;
  - `test_cli`: command-line validation;
  - `test_count_primes`, which now also runs from a non-default `--start`;
  - `test_phase1`, which now covers multiple segments and batch sizes 1, 7
    and 1000;
  - `test_sieve` and `test_bitset`, now registered with CTest.
- `tests/run_sanitizers.sh` (not part of ctest).

### Removed
- `tests/validation.sh` and `tests/validation_gpu.sh`, which checked outputs
  that no longer exist. `test_cli` covers their cases.

## [3.1.0] - 2026-10-26

### Changed
- **`TILE_ODDS` default 32768 to 16384.** 13-14% faster at 10^11, 10^12 and
  10^13 on an RTX 5090, measured as variable cost with the large-prime kernel
  present, and 1.13x faster at 10^14 end to end: 836.13 s wall against the
  v3.0.0 tag run's 942.3 s. The previous default was measured before that kernel
  existed, when a wider tile still amortised per-tile work that has since moved
  out of the tiled sieve. `SPLIT_THRESHOLD` was re-measured at the new tile width
  and stays at 65536. Occupancy does not predict the optimum: resident blocks per SM fall
  monotonically with tile width while runtime is U-shaped.

- **`big_check` reports the minimal p.** It now searches only p <= L, where
  L = min(`--p-max`, floor(n/2)), and returns the smallest such p with n - p a
  probable prime, for any thread count. Previously the first thread to reach a
  candidate past n/2 could end the search, so the reported p was not
  necessarily minimal.

- **`big_check` exit statuses.** 0 found, 3 no partition with p <= L, 1 invalid
  input. Exhausting the bound is reported as a search limit and no longer
  described as a counterexample.

- **`big_check` digit counts are exact.** Reported digit counts came from
  `mpz_sizeinbase(x, 10)`, which GMP documents as possibly one too large for any
  base that is not a power of two. q for n = 10^100 was reported as 101 digits
  where it has 100. The count now comes from the decimal string, and the same
  count sets the threshold below which q is printed in full.

### Added
- **`test_phase1` covers the segment-bitset branch of `is_prime_q`.** No range in
  the test reached it: the scalar kernel is selected only for
  `seg_even_start <= 2*p_small + 128`, and in the one range that qualified every
  complement stayed below `small_high`, so the lookup always answered from
  `d_small`. A counting probe recorded zero entries to that branch over a full
  run. The same range at `p_small = 1e4` reaches it.

- **CTest registration.** The five test binaries and a new `test_big_check` now
  run under `ctest --output-on-failure`. `test_big_check` checks `big_check`
  against the 48 published records, small-n edges, the search-limit exit,
  thread-count determinism, expression input and exact digit counts. It reads the
  record table out of `test_records.cpp` rather than copying it, and fails unless
  all 48 records are parsed and each is a plausible (n, p) pair.

- **`big_check` expression input.** n may be written `a^b`, `a^b+c` or `a^b-c`
  and is evaluated with GMP, so large cases are reproducible from their command
  lines. Evenness and n >= 4 are checked on the parsed value rather than on the
  last character of the string. Adds `--p-max` (default 1e7) and `--quiet`.

- **`big_check` summary line.** Every run ends with
  `RESULT status=<found|bound> p=<p> q_digits=<d> candidates=<i> prp_calls=<k> threads=<t> time_s=<s>`.
  `prp_calls` depends on scheduling and is reproducible only at a fixed thread
  count; `p`, `q_digits` and `candidates` do not.

## [3.0.0] - 2026-09-09

### Fixed
- **Segment sieve correctness.** `tiled_sieve_segment_kernel` marked composites
  in shared memory with a non-atomic read-modify-write. Threads within a block
  sieve distinct primes into the same 64-bit words, so concurrent updates were
  lost and some composites remained marked prime. Because `is_prime_q` consults
  this bitset, Phase 1 could accept a partition `n = p + q` in which `q` is not
  prime. Now uses `atomicAnd`. (Reported by an independent correspondent)

- **Parallel bitset construction.** `build_prime_bitset` gave each OpenMP thread
  a slice that was exclusive in bit index but not in 64-bit word, so adjacent
  threads raced on boundary words and lost each other's clears. A lost clear
  leaves the bit set, reporting a composite as prime. Measured at 90 of 250
  builds wrong with 16 threads. Slices are now word-aligned.

- **BPSW overflow above 2^63.** Intermediate sums in the strong Lucas test
  could exceed 64 bits and wrap, causing genuine primes to be reported as
  composite. Affected the default primality path introduced in v2.1.0 for
  `n > 2^63` only. The failure direction was safe: false negatives cause
  Phase 2 fallbacks (never false verification).

- **32-bit thread index overflow.** All kernels computed their global index as
  `blockIdx.x * blockDim.x + threadIdx.x`, a 32-bit product that wrapped past
  2^32 threads. Affected even numbers were skipped by the GPU and fell through
  to the single-threaded CPU path, presenting as a hang rather than an error.

- **VRAM validation.** The check compared a ~118 MB estimate against total
  device memory while real usage is ~573 MiB, the difference being the CUDA
  context, which was not modelled. It now compares against free memory after
  context creation, and counts `d_small_primes`, previously omitted.

- **CUDA architecture detection.** `native` silently resolved to `sm_75` on an
  `sm_120` device, producing a binary that ran via PTX JIT. Automatic detection
  is now verified against the GPU's reported compute capability.

### Added
- `test_gpu_sieve`: differential test comparing `tiled_sieve_segment_kernel`
  output against the CPU segmented sieve over fixed and randomized ranges,
  including the `q_low = 3` edge and ranges straddling 2^32.
- `test_primality`: cross-checks BPSW against the 12-base deterministic
  Miller-Rabin oracle, with emphasis on the region above 2^63.
- `test_phase1`: differential test of Phase 1 against a CPU reference, comparing
  across prime-list prefixes so the check resolves which prime succeeded rather
  than the saturated yes/no verdict.
- `test_bitset_race`: repeated parallel bitset construction against a
  single-threaded reference, at word-aligned and misaligned thread boundaries.
- `test_records`: minimal primes against 48 published record values from
  Oliveira e Silva's verification data.
- `--record-check`: reports each new maximum minimal prime during a run. Off by
  default; costs ~26% at 1e14. Records may be suppressed under `--gpus>1`.
- `analyze_pmin`: measures the distribution of the first successful prime
  index in Phase 1, per number and per warp.

### Changed
- Segment sieve marks composites as bytes rather than bits in shared memory,
  removing the atomic read-modify-write on contended words.
- Phase 1 processes 64 even numbers per thread as a single 64-bit word, since
  the complements of 64 consecutive even numbers for a fixed prime are 64
  consecutive odd numbers. Issued work falls from ~50 warp-iterations per even
  number to 1.85.
- Phase 1 verification state stored as a bitset: 200 MB to 25 MB per segment at
  `--seg-size=200000000`.
- Sieving primes above `SPLIT_THRESHOLD` handled by a separate kernel once per
  segment rather than rescanned per tile.
- `TILE_ODDS` default 16384 -> 32768; `SPLIT_THRESHOLD` introduced, default 65536.
- `--seg-size` derived from free VRAM when not given.
- Net effect at N=1e11: 11.09 s -> 0.81 s against a corrected v2.1.0 baseline.
  1e14 verifies in 933 s on a single RTX 5090.

### Removed
- `goldbach_gpu5a`, an experimental near-duplicate of `goldbach.cu` carrying its
  own copies of the sieve and primality code.
- `--async`, which was parsed and never used.

---


## v2.1.0 - 2026-03-11

### Enhanced Primality Testing (BPSW)
- **Implemented Enhanced Baillie–PSW (Method A\*)** ([arXiv:2006.14425](https://arxiv.org/abs/2006.14425)), replacing the previous 12‑base Miller–Rabin fallback, upon Baillie's suggestion.  
- **Deterministic for 64‑bit inputs**: This BPSW variant has no known counterexamples below \(2^{64}\).  
- **~60% faster Phase 2 primality checks** compared to the legacy Miller–Rabin implementation.  
- **No impact on wall‑clock time**: These primality tests are only invoked as *untriggered fallbacks* and in rare GPU overflow cases. For typical search ranges, overall runtime remains unchanged.

### CLI & Configuration
- **Added `--primetest` flag**: Selects the primality test (`mr` or `bpsw`) used by **both GPU Phase 1 primality checks** and **CPU Phase 2 fallback**. Default is bpsw.
- **Method A\* parameter selection**: Uses the \(P=5, Q=5\) override to avoid the pseudoprime class identified in recent research.

### Mathematical Correctness
- **Exact integer square root**: Replaced floating‑point `sqrt()` with a binary‑search integer square root to avoid precision issues near the \(2^{64}\) boundary.  
- **Strict Q‑check**: Added the \(V_{n+1} \equiv 2Q \pmod n\) verification step to the BPSW pipeline.

---

## v2.0.0 - 2026-03-05
### Multi‑GPU Execution & Concurrency
- Introduced a **lock‑free multi‑GPU worker pool** with dynamic load balancing, enabling heterogeneous GPUs to scale linearly.
- Added an asynchronous **progress monitor** and **thread‑safe logging**, eliminating console contention during high‑throughput runs.

### GPU Kernel & Pipeline Redesign
- Replaced CPU‑side segment generation with a **GPU‑native tiled sieve**, removing PCIe bottlenecks and dramatically increasing throughput.
- Added a **zero‑copy fast path** using device‑side reduction to avoid unnecessary host transfers.

### Mathematical Correctness & Overflow Safety
- Implemented strict **overflow guards** for 64‑bit boundary cases (e.g., safe handling of \(p \cdot p\) near \(10^{19}\)).
- Improved error handling: GPU failures now raise exceptions instead of terminating the process, ensuring clean shutdown of all worker threads.

### CPU Fallback (Phase 2) Improvements
- Added **eager, thread‑safe initialization** of fallback primes.
- Replaced the old exhaustive trial division with a **binary‑search‑based prime lookup** and **128‑bit Miller–Rabin**, yielding major speedups.

### Developer Experience & CLI Enhancements
- Added new CLI options (`--gpus`, `--start`, `--progress`) for fine‑grained control of hardware and UI behavior.
- Added **hardware pre‑validation** for VRAM and grid dimensions to prevent deep‑execution CUDA failures.

### Validation & Reliability
- Added a comprehensive **GPU Goldbach validation script** that cross‑checks GPU results against CPU and big‑int paths.

---

## v1.1.0 – 2026-03-02

**Added**
- Multi-GPU Goldbach verifier with work-stealing across devices, integrated as the new `goldbach_gpu3`.
- Multi-GPU benchmarking support and updated parameter tuning for modern GPUs (e.g., H100 SXM).

**Improved**
- CMake configuration now defaults to the native CUDA architecture and has cleaner, finalized targets.
- Repository layout: legacy tools and older GPU variants moved under legacy targets.

**Documentation**
- Expanded README with multi-GPU usage examples, including an 8× H100 cluster run.
- Updated build instructions, version badge, RESULTS log, and Zenodo DOIs in `README` and `CITATION.cff`.
- Added `CHANGELOG.md` to report version updates.