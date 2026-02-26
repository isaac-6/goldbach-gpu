# Goldbach Verification Results Log

All results were produced on the following fixed platform:

| Component | Specification |
|----------|---------------|
| **CPU** | Intel i7‑12700H, 20 logical threads |
| **GPU** | NVIDIA RTX 3070, 8 GB VRAM, 448 GB/s bandwidth |
| **RAM** | 32 GB |
| **OS** | WSL2, Ubuntu 24.04 |
| **CUDA Toolkit** | 13.1.115 |
| **CUDA Build Info** | cuda_13.1.r13.1/compiler.37061995_0 (Dec 16 2025) |
| **GCC** | 13.3.0 (Ubuntu 13.3.0‑6ubuntu2~24.04.1) |
| **OpenMP** | 4.5 |
| **GMP** | 6.3.0+dfsg‑2ubuntu6.1 |

All timings are wall‑clock time. All configurations are recorded exactly as run so results are fully reproducible.

---

## CPU baseline (`goldbach`)

Tool: `cpu_goldbach.cpp`
Method: Segmented sieve of Eratosthenes + sequential Goldbach scan.
No GPU. Serves as correctness oracle for all GPU tools.

| Limit | Even n checked | Sieve | Check | Total | Failures |
|-------|----------------|-------|-------|-------|----------|
| 10^7  | 4,999,999      | 27 ms | 89 ms | 117 ms | 0 |
| 10^8  | 49,999,999     | 768 ms | 1,052 ms | 1,820 ms | 0 |
| 10^9  | 499,999,999    | 9,836 ms | 12,786 ms | 22,622 ms | 0 |
| 10^10 | 4,999,999,999  | 119,947 ms | 188,388 ms | 308,335 ms | 0 |

---

## GPU range verifier -- byte array (`goldbach_gpu`)

Tool: `goldbach_gpu.cu`
Method: Byte array prime table (1 byte per number) + GPU kernel, one thread per even n.
Hard VRAM limit: ~10^9 (byte array requires 953 MB at 10^9).

| Limit | Even n checked | Sieve (CPU) | Kernel (GPU) | Total | Failures |
|-------|----------------|-------------|--------------|-------|----------|
| 10^6  | 499,999        | 6 ms    | 0.91 ms | 31 ms    | 0 |
| 10^8  | 49,999,999     | 1,042 ms | 25 ms  | 4,861 ms | 0 |
| 10^9  | 499,999,999    | 12,928 ms | 206 ms | 57,848 ms | 0 |

Note: These are pre-OpenMP sieve timings (historical). Current sieve is faster.

---

## GPU range verifier -- compact bitset (`goldbach_gpu2`)

Tool: `goldbach_gpu2.cu`
Method: Compact bitset (1 bit per odd number, 16x memory reduction) + GPU kernel.
VRAM usage: 59 MB at 10^9, 596 MB at 10^10, 5,960 MB at 10^11.
Hard VRAM limit: cannot exceed 10^11 on 8 GB GPU (10^12 would need 59 GB).
Sieve parallelized with OpenMP (20 threads, 1.7x speedup, memory-bandwidth limited).

| Limit | Even n checked | Sieve (CPU, OpenMP) | Kernel (GPU) | Total | Failures |
|-------|----------------|---------------------|--------------|-------|----------|
| 10^7  | 4,999,999      | 16 ms    | 6 ms     | 22 ms     | 0 |
| 10^8  | 49,999,999     | 46 ms    | 75 ms    | 121 ms    | 0 |
| 10^9  | 499,999,999    | 687 ms   | 698 ms   | 1,386 ms  | 0 |
| 10^10 | 4,999,999,999  | 17,923 ms | 7,520 ms | 25,443 ms | 0 |

GPU speedup over CPU baseline (total vs total): 16x at 10^9, 12x at 10^10.
At 10^8, kernel launch overhead causes GPU kernel time to exceed sieve time;
GPU advantage dominates from 10^9 upward.

---

## GPU segmented verifier (`goldbach_gpu3`)

Tool: `goldbach_gpu3.cu`
Method: Double sieve over n. For each segment [A, B]:
  - GPU Phase 1: for each prime p <= P_SMALL, mark all even n in [A, B]
    where n-p is prime. q = n-p checked via: small primes bitset (q <= P_SMALL),
    segment bitset (q in [A, B]), or Miller-Rabin (q elsewhere).
  - CPU Phase 2: exhaustive fallback, all odd p up to n/2. Never triggered.
No VRAM ceiling: each segment uses ~60 MB regardless of total range.

### Configuration used for all runs below
```
SEG_SIZE = 500,000,000  (5x10^8 even numbers per segment)
P_SMALL  = 2,000,000
P_BATCH  = 100,000
GPU primes used: 148,933
Segment buffer: 59 MB
Verified buffer: 476 MB
```

| Limit | Even n checked | GPU success | Phase 2 fallbacks | Total time | Failures |
|-------|----------------|-------------|-------------------|------------|----------|
| 10^9  | 499,999,999    | 100% | 0 | 3,979 ms      | 0 |
| 10^10 | 4,999,999,999  | 100% | 0 | 41,320 ms     | 0 |
| 10^11 | 49,999,999,999 | 100% | 0 | 488,478 ms    | 0 |
| 10^12 | 499,999,999,999 | 100% | 0 | 5,760,350 ms | 0 |

100% GPU success rate confirms p_min(n) <= 2,000,000 for all even n <= 10^12.
Predicted worst case H(10^12) ~ 2,000 (quadratic fit to Oliveira e Silva data).
Our bound exceeds the prediction by 3 orders of magnitude as a safety margin.

goldbach_gpu3 is slower than goldbach_gpu2 at shared limits (41,320 ms vs
25,443 ms at 10^10) due to per-segment overhead. This is an inherent trade-off:
goldbach_gpu3 has no VRAM ceiling; goldbach_gpu2 cannot exceed 10^11.

---

## GPU single number checker (`single_check`)

Tool: `single_check.cu`
Method: Deterministic Miller-Rabin, 12 witnesses, correct for all 64-bit integers.
Hard limit: uint64_t max (~1.8x10^19).
Note: returns a valid partition but not necessarily the minimal one
(concurrent GPU threads -- whichever completes first triggers atomic exit).

| n | Partition found | Time |
|---|----------------|------|
| 10^12 | 194,267 + 999,999,805,733 | 1.4 s |
| 10^13 | 90,059 + 9,999,999,909,941 | 1.4 s |
| 10^14 | 145,391 + 99,999,999,854,609 | 1.5 s |
| 10^15 | 152,639 + 999,999,999,847,361 | 1.4 s |
| 10^16 | 56,687 + 9,999,999,999,943,313 | 1.4 s |
| 10^17 | 91,967 + 99,999,999,999,908,033 | 1.6 s |
| 10^18 | 14,831 + 999,999,999,999,985,169 | 1.5 s |
| 10^19 | 226,283 + 9,999,999,999,999,773,717 | 1.5 s |

Timing is approximately constant across orders of magnitude, reflecting
consistent availability of small-prime Goldbach partitions.

---

## Arbitrary precision checker (`big_check`)

Tool: `big_check.cpp`
Method: GMP exact arithmetic + probabilistic Miller-Rabin (25 rounds,
false positive probability < 4^-25 ~ 10^-15). OpenMP parallelism with
batch-synchronised prime search (1,000 primes per batch) to prevent
thread runaway. Returns a valid partition; p is not guaranteed minimal
due to thread scheduling, but batch synchronisation ensures near-minimal.

Practical limit: each GMP primality test scales as O(d^3) in digit count d.
At 10^10000 (d=10,001) tests take ~ms each; at 10^100000 (d=100,001) tests
take ~seconds each, making exhaustive batch search infeasible.

| n | Digits | p found | Threads | Time |
|---|--------|---------|---------|------|
| 10^50    | 51    | 383    | 20 | 43 ms |
| 10^100   | 101   | 797    | 20 | 43 ms |
| 10^200   | 204   | 113    | 20 | 43 ms |
| 10^1000  | 1,001 | 26,981 | 1  | 2,299 ms |
| 10^1000  | 1,001 | 26,981 | 20 | 363 ms |
| 10^10000 | 10,001 | 47,717 | 20 | 182,000 ms |

Timing depends on index of first valid p found, not digit count.
p=113 (10^200) found in 43 ms; p=26,981 (10^1000) takes 363 ms with 20 threads.

---

## Roadmap

- [x] CPU baseline range verifier
- [x] GPU range verifier, byte array (to 10^9)
- [x] GPU single number checker, Miller-Rabin uint64_t (to 1.8x10^19)
- [x] Compact prime bitset, 16x memory reduction
- [x] GPU range verifier, compact bitset (to 10^11)
- [x] OpenMP parallel sieve, 1.7x speedup (memory-bandwidth limited)
- [x] GMP arbitrary precision single number checker (to 10^10000, 20 threads)
- [x] Batch-synchronised big_check (prevents thread runaway, 21% faster)
- [x] GPU segmented verifier, correct double-sieve design (gpu3)
- [x] Range verification to 10^12 (500 billion even numbers, 96 minutes)
- [ ] GPU-accelerated sieve construction (remove CPU sieve bottleneck)
- [ ] Oliveira-style bulk double sieve on GPU (O(N log log N), target 10^15)
- [ ] Goldbach partition counting c(n) at scale
