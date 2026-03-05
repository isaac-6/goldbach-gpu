# Goldbach Verification Results Log

All range verification results were produced on the following platform:

| Component | Specification |
|----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5090, 32 GB VRAM, Driver 580.95.05, CUDA 13.0 |
| **CPU** | Dual‑socket AMD EPYC (Engineering Sample 100‑000000897‑03), 128 logical cores; Effective: 13.6 vCPUs (cgroup quota: 1360000 / 100000) |
| **Memory** | 44GB |
| **Environment** | Ubuntu (containerized); GPU access: Full, non‑virtualized RTX 5090 |
| **OS** | Ubuntu 24.04 |
| **CUDA Toolkit** | 13.0 |
| **GCC** | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| **OpenMP** | 4.5 |
| **GMP** | 6.3.0+dfsg-2ubuntu6.1 |

All timings are wall‑clock time. All configurations are recorded exactly as run so results are fully reproducible.

---

## CPU baseline (`cpu_goldbach`)

Tool: `cpu_goldbach.cpp`
Method: Segmented sieve of Eratosthenes + sequential Goldbach scan.
No GPU. Serves as correctness oracle for all GPU tools.

| Limit | Even n checked | Total | Failures |
|-------|----------------|-------|----------|
| 10^8  | 49,999,999     | 1,515.2 ms | 0 |
| 10^9  | 499,999,999    | 19,183.7 ms | 0 |

---

## GPU range verifier -- current flagship (`goldbach`)

Tool: `goldbach.cu`
Method: Byte array prime table (1 byte per number) + GPU kernel, one thread per even n.

SEG_SIZE = 200,000,000
P_SMALL  = 1,000,000
P_BATCH  = 2,000,000

| Limit | Even n checked | Total | Failures |
|-------|----------------|-------|----------|
| 10^9  | 499,999,999    | 141.018 ms | 0 |
| 10^10  | 4,999,999,999    | 395.769 ms | 0 |
| 10^11  | 49,999,999,999    | 3,311.5 ms | 0 |
| 10^12  | 499,999,999,999    | 37,440 ms | 0 |

---

## GPU legacy verifier (`goldbach_gpu3`)

Tool: `goldbach_gpu3.cu`
Method: Double sieve over n. For each segment [A, B]:
  - GPU Phase 1: for each prime p <= P_SMALL, mark all even n in [A, B]
    where n-p is prime. q = n-p checked via: small primes bitset (q <= P_SMALL),
    segment bitset (q in [A, B]), or Miller-Rabin (q elsewhere).
  - CPU Phase 2: exhaustive fallback, all odd p up to n/2. Never triggered.
No VRAM ceiling: each segment uses ~60 MB regardless of total range.

### Configuration used for all runs below
```
SEG_SIZE = 10,000,000  (5x10^6 even numbers per segment)
P_SMALL  = 1,000,000
P_BATCH  = 100,000
GPU primes used: 78,498
Segment buffer: 59 MB
Verified buffer: 476 MB
```

| Limit | Even n checked | Total | Failures |
|-------|----------------|-------|----------|
| 10^8  | 49,999,999    | 451.657 ms | 0 |
| 10^9  | 499,999,999    | 1,867.67 ms | 0 |
| 10^10  | 4,999,999,999    | 18,056.5 ms | 0 |

100% Phase 1 confirms p_min(n) <= 1,000,000 for all even n <= 10^12 (from prefious run with another system).
Predicted worst case H(10^12) ~ 2,000 (quadratic fit to Oliveira e Silva data).
Our bound exceeds the prediction by 3 orders of magnitude as a safety margin.

goldbach_gpu3 has no VRAM ceiling; old goldbach_gpu2 cannot exceed 10^11.

---

All results below were produced on the following fixed platform:

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
- [x] Range verification to 10^12 (500 billion even numbers, 41 minutes)
- [x] Multi-GPU adaptation (tested on 8x H100)
- [x] GPU-accelerated sieve construction (remove CPU sieve bottleneck)
- [ ] Goldbach partition counting c(n) at scale
