## v2.0.2 - 2026-09-18

Corrigendum reference artifact for arXiv:2603.07850. Release v2.0.0 with
correctness fixes and nothing that changes its performance. Supersedes v2.0.1.

### Fixed
- Removed `src/goldbach_gpu5a.cu`, an auxiliary program not described in the
  paper, which carried a second unfixed copy of the segment-sieve defect
  corrected in v2.0.1. Its benchmark driver `benchmarks/bench_5.sh` is removed
  with it.
- `--seg-size` above 2^32 is now rejected at startup. All kernels compute their
  global thread index as a 32-bit product, which wraps past 2^32 threads; Phase
  1 and the counting kernel wrap identically, so affected even numbers were
  neither verified nor counted and the program reported success. The indexing
  itself is corrected in v3.0.0. No published run came near this: every run used
  `--seg-size=200000000`.

### Changed
- Withdrawn performance figures removed from `README.md` and the `goldbach.cu`
  header, and replaced with corrected ones. The historical figures in
  `RESULTS.md` are kept as a record and marked superseded in place.
- Build instructions now state `-DCMAKE_CUDA_ARCHITECTURES=120`, without which a
  default configure on this branch resolves to sm_75 and does not reproduce the
  documented timings.
- `CITATION.cff` corrected: it described v1.1.0 and carried the title of a
  different paper.
- Provenance notes and the machine B data added under `corrigendum-logs/`.
- `.gitignore`: added an exception for `corrigendum-logs/**/*.txt`, which the
  blanket `*.txt` rule had been silently excluding, and widened the build
  directory rule to `build*/`.

The measurements in `corrigendum-logs/` were taken at commit `cca9e71`.
`src/goldbach.cu` and `src/prime_bitset.cpp` are unchanged since then apart from
the header comment and the `--seg-size` guard, which cannot execute at the
parameters used, so the figures carry over.

## v2.0.1 - 2026-09-11

### Fixed
- **Segment sieve correctness.** `tiled_sieve_segment_kernel` marked composites
  in shared memory with a non-atomic read-modify-write. Threads within a block
  sieve distinct primes into the same 64-bit words, so concurrent updates were
  lost and some composites remained marked prime. Because `is_prime_q` consults
  this bitset, Phase 1 could accept a partition `n = p + q` in which `q` is not
  prime. Now uses `atomicAnd`. Reported by an independent correspondent, who
  identified it by reading the published source.
- **Parallel bitset construction.** `build_prime_bitset` gave each OpenMP thread
  a slice that was exclusive in bit index but not in 64-bit word, so adjacent
  threads raced on boundary words and lost each other's clears. A lost clear
  leaves the bit set, reporting a composite as prime. Slices are now
  word-aligned.

### Added
- `test_bitset_race`, a regression test for the above.
- `corrigendum-logs/`, the raw measurements supporting the corrigendum.

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