## [3.0.0] - 2026-09-09

### Fixed
- **Segment sieve correctness.** `tiled_sieve_segment_kernel` marked composites
  in shared memory with a non-atomic read-modify-write. Threads within a block
  sieve distinct primes into the same 64-bit words, so concurrent updates were
  lost and some composites remained marked prime. Because `is_prime_q` consults
  this bitset, Phase 1 could accept a partition `n = p + q` in which `q` is not
  prime. Now uses `atomicAnd`. (kudos to Kenzi Voyer for reporting)

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