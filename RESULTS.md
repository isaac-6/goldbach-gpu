# Goldbach Verification Results Log

Hardware used unless otherwise noted:
- CPU: Intel i7-12700H, 20 threads, WSL2 Ubuntu 24
- GPU: NVIDIA RTX 3070, 8 GB VRAM
- RAM: 32 GB

---

## Log

### 2026 - Range verification (CPU baseline)
Tool: `goldbach` (cpu_goldbach.cpp)
Method: Simple sieve + sequential Goldbach check

| Date | Limit | Even n checked | Time | Failures |
|------|-------|----------------|------|----------|
| 2026 | 10^6  | 499,999        | 31ms | 0 |
| 2026 | 10^8  | 49,999,999     | 4,861ms | 0 |
| 2026 | 10^9  | 499,999,999    | 57,848ms | 0 |

---

### 2026 - Range verification (GPU byte array)
Tool: `goldbach_gpu` (goldbach_gpu.cu)
Method: Byte array prime table + GPU kernel, one thread per even n

| Date | Limit | Even n checked | Sieve (CPU) | Kernel (GPU) | Failures |
|------|-------|----------------|-------------|--------------|----------|
| 2026 | 10^6  | 499,999        | 6ms     | 0.91ms | 0 |
| 2026 | 10^8  | 49,999,999     | 1,042ms | 25ms   | 0 |
| 2026 | 10^9  | 499,999,999    | 12,928ms| 206ms  | 0 |

GPU speedup vs CPU: 27x at 10^6, 153x at 10^8, 208x at 10^9.
VRAM limit hit at 10^9 (953 MB byte array).

---

### 2026 - Single number verification (GPU Miller-Rabin)
Tool: `single_check` (single_check.cu)
Method: GPU Miller-Rabin, 12 witnesses, deterministic for all uint64_t

| Date | n | Partition found | Time |
|------|---|----------------|------|
| 2026 | 10^12 | 194,267 + 999,999,805,733 | 1.4s |
| 2026 | 10^13 | 90,059 + 9,999,999,909,941 | 1.4s |
| 2026 | 10^14 | 145,391 + 99,999,999,854,609 | 1.5s |
| 2026 | 10^15 | 152,639 + 999,999,999,847,361 | 1.4s |
| 2026 | 10^16 | 56,687 + 9,999,999,999,943,313 | 1.4s |
| 2026 | 10^17 | 91,967 + 99,999,999,999,908,033 | 1.6s |
| 2026 | 10^18 | 14,831 + 999,999,999,999,985,169 | 1.5s |
| 2026 | 10^19 | 226,283 + 9,999,999,999,999,773,717 | 1.5s |

Hard limit: uint64_t max (~1.8 x 10^19).

---

### 2026 - Range verification (GPU bitset)
Tool: `goldbach_gpu2` (goldbach_gpu2.cu)
Method: Compact prime bitset (1 bit per odd number) + GPU kernel

| Date | Limit | Even n checked | Sieve (CPU) | Kernel (GPU) | Total | Failures |
|------|-------|----------------|-------------|--------------|-------|----------|
| 2026 | 10^9  | 499,999,999    | 2,466ms   | 667ms    | 3,133ms   | 0 |
| 2026 | 10^10 | 4,999,999,999  | 31,238ms  | 7,893ms  | 39,131ms  | 0 |
| 2026 | 10^11 | 49,999,999,999 | 380,746ms | 95,123ms | 475,869ms | 0 |

Memory: 59MB at 10^9, 596MB at 10^10, 5,960MB at 10^11.
VRAM limit hit at 10^11 (5,960MB of 8,192MB available).
Sieve is 80% of total runtime, main bottleneck.

OpenMP parallel sieve (20 threads): 1.7x speedup on sieve.
Memory-bandwidth limited, adding cores does not scale linearly.

### 2026 - Range verification (GPU segmented bitset)
Tool: `goldbach_gpu3` (goldbach_gpu3.cu)
Method: Double sieve over n. For each segment [A, B]:
  Phase 1 (GPU): for each prime p <= P_SMALL, mark all even n
    where n-p is prime as verified. q checked via small bitset,
    segment bitset, or Miller-Rabin depending on where q falls.
  Phase 2 (CPU): exhaustive fallback for any unverified n.
    All odd p up to n/2 tested. Expected to never trigger.
Fixes the conceptual bug in gpu3 draft: q is no longer forced
into the current segment.

| Date | Limit | Even n checked | P_SMALL | SEG_SIZE | Total time | Failures | Phase 2 |
|------|-------|----------------|---------|----------|------------|----------|---------|
| 2026 | 10^10 | 4,999,999,999  | 1,000,000 | 10^9 | 44,803ms   | 0 | 0 |
| 2026 | 10^12 | 499,999,999,999| 2,000,000 | 5x10^8 | 5,742,130ms | 0 | 0 |

Parameters used for 10^12 run:
- LIMIT    = 10^12
- SEG_SIZE = 500,000,000 (5x10^8 even numbers per segment)
- P_SMALL  = 2,000,000
- P_BATCH  = 100,000
- GPU primes used: 148,933
- Segment buffer: 59 MB
- Verified buffer: 476 MB

Zero Phase 2 fallbacks confirms empirically that P_SMALL = 2,000,000
is sufficient for all even n up to 10^12. The adaptive multi-phase
optimization (planned) will exploit this to reduce runtime significantly.

---

### 2026 - Big integer single number verification (GMP)
Tool: `big_check` (big_check.cpp)
Method: CPU GMP arbitrary precision + Miller-Rabin (25 rounds)
No GPU used. No size limit, works for any even number.

| Date | n | Digits | p found | Time |
|------|---|--------|---------|------|
| 2026 | 10^50     | 51   | 383    | 43ms    |
| 2026 | 10^100    | 101  | 797    | 43ms    |
| 2026 | 10^200    | 204  | 113    | 43ms    |
| 2026 | 10^1000   | 1001  | 26,981 | 2,299ms  | 1 thread  |
| 2026 | 2x10^1000 | 1001  | 14,437 | 1,405ms  | 1 thread  |
| 2026 | 4x10^1000 | 1001  | 83     | 82ms     | 1 thread  |
| 2026 | 8x10^1000 | 1001  | 12,601 | 1,226ms  | 1 thread  |
| 2026 | 10^1000   | 1001  | 26,981| 363ms    | 20 threads|
| 2026 | 10^10000  | 10001 | 47,717 | 181s| 20 threads|

Timing depends primarily on which prime p gives a valid partition,
not on the digit count of n. p=83 finds instantly; p=26,981 takes 2.3s.
10^10000 aborted, estimated 30-60min single-threaded.

---

## Roadmap

- [x] CPU baseline range verifier
- [x] GPU range verifier, byte array (to 10^9)
- [x] GPU single number checker, Miller-Rabin uint64_t (to 1.8x10^19)
- [x] Compact prime bitset, 16x memory reduction
- [x] GPU range verifier, bitset (to 10^11)
- [x] OpenMP parallel sieve, 1.7x speedup (memory-bandwidth limited)
- [x] GMP big integer single number checker (verified to 10^1000)
- [x] Parallel big_check with OpenMP, 6x speedup, verified 10^10000 in 231s
- [x] GPU segmented verifier with correct double-sieve design (gpu3)
- [x] Range verification to 10^12 (500 billion even numbers, 96 minutes)
- [x] Big integer checker to 10^10000 (needs parallelization)
- [ ] Adaptive multi-phase p-search (target: 10^12 in ~5-10 minutes)
- [ ] Range verification to 10^13 with optimized gpu3
- [ ] Count partitions c(n) at scale