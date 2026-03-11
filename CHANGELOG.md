## v2.1.0 - 2026-03-11

### Enhanced Primality Testing (BPSW)
- **Implemented Enhanced Baillie–PSW (Method A\*)** ([arXiv:2006.14425](https://arxiv.org/abs/2006.14425)), replacing the previous 12‑base Miller–Rabin fallback, thanks to Baillie's suggestion.  
- **Deterministic for 64‑bit inputs**: This BPSW variant has no known counterexamples below \(2^{64}\), improving rigor for Goldbach verification.  
- **~60% faster Phase 2 primality checks** compared to the legacy Miller–Rabin implementation.  
- **No impact on wall‑clock time**: These primality tests are only invoked as *untriggered fallbacks* and in rare GPU overflow cases. For typical search ranges, overall runtime remains unchanged.

### CLI & Configuration
- **Added `--primetest` flag**: Selects the primality test (`mr` or `bpsw`) used by **both GPU Phase 1 primality checks** and **CPU Phase 2 fallback**. 
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