# Goldbach Verification Results

## Hardware

- **CPU:** AMD/Intel, 20 threads (WSL2, Ubuntu 24)
- **GPU:** NVIDIA RTX 3070 (8 GB VRAM)
- **RAM:** 32 GB

---

## Background

**Goldbach's conjecture** (1742): every even integer greater than 2
can be expressed as the sum of two prime numbers.

Example: 100 = 3 + 97 = 11 + 89 = 17 + 83 = 29 + 71 = 41 + 59 = 47 + 53

It remains one of the oldest unsolved problems in mathematics.
Never formally proved, but verified computationally up to **4 × 10^18**
by Tomás Oliveira e Silva (University of Aveiro) using a distributed
computing project running over many years on many machines.

Reference: https://sweet.ua.pt/tos/goldbach.html

**No counterexample has ever been found.**

---

## Our approach

We implemented two complementary verification strategies:

### 1. Range verification
Verify every even n in [4, LIMIT] using a GPU-accelerated
segmented sieve combined with a compact prime bitset.

- CPU builds a bitset (1 bit per odd number = 16x compression
  vs byte array)
- Bitset is copied to GPU VRAM once
- CUDA kernel launches one thread per even n
- Each thread scans primes p up to n/2, checks if n-p is prime
  via bitset lookup
- Batched processing avoids VRAM overflow

### 2. Single number verification
Verify one specific large even n using Miller-Rabin primality
test on GPU — no lookup table needed, works up to uint64_t max
(~1.8 × 10^19).

- CPU generates primes up to n/2 in chunks
- GPU tests each prime p: is n-p also prime? (Miller-Rabin)
- Reports first valid partition (p, q) with p ≤ q, p + q = n
- Deterministic: uses 12 witnesses, correct for all 64-bit integers

---

## Range verification results

All even numbers in [4, LIMIT] verified. Zero failures found.

| Limit | Even numbers checked | Sieve (CPU) | Kernel (GPU) | Total |
|-------|---------------------|-------------|--------------|-------|
| 10^6  | 499,999             | 6ms         | 0.91ms       | 31ms  |
| 10^8  | 49,999,999          | 1,042ms     | 25ms         | 4,861ms |
| 10^9  | 499,999,999         | 12,928ms    | 206ms        | 57,848ms |
| 10^10 | 4,999,999,999       | 31,238ms    | 7,893ms      | 39,131ms |
| 10^11 | 49,999,999,999      | 380,746ms   | 95,123ms     | 475,869ms |

**Memory usage at 10^11:**
- Bitset: 5,960 MB (vs ~95 GB for byte array — 16x compression)
- This approaches the 8 GB VRAM limit of the RTX 3070
- Going beyond 10^11 requires segmented bitset processing

---

## Single number verification results

| n | Partition found | Time |
|---|----------------|------|
| 10^12 | 194,267 + 999,999,805,733 | 1.4s |
| 10^13 | 90,059 + 9,999,999,909,941 | 1.4s |
| 10^14 | 145,391 + 99,999,999,854,609 | 1.5s |
| 10^15 | 152,639 + 999,999,999,847,361 | 1.4s |
| 10^16 | 56,687 + 9,999,999,999,943,313 | 1.4s |
| 10^17 | 91,967 + 99,999,999,999,908,033 | 1.6s |
| 10^18 | 14,831 + 999,999,999,999,985,169 | 1.5s |
| 10^19 | 226,283 + 9,999,999,999,999,773,717 | 1.5s |

**Notes:**
- Timing is essentially constant (~1.5s) regardless of n
- Partitions are always found with small primes — this is expected
- The smallest prime in a Goldbach partition of n grows roughly
  as (log n)^2, which is very slow
- Limited to uint64_t max (~1.8 × 10^19)

---

## Context and comparison

| Source | Method | Verified up to |
|--------|--------|----------------|
| Oliveira e Silva (2013) | Distributed CPU | 4 × 10^18 |
| This project — range | GPU bitset (RTX 3070) | 10^11 |
| This project — single n | GPU Miller-Rabin (RTX 3070) | 1.8 × 10^19 |

Our range verification (10^11) is well below the published frontier
(4 × 10^18). Reaching the frontier requires:
- Segmented bitset to remove the VRAM ceiling
- Faster CPU sieve (currently 80% of total runtime)
- Potentially cloud GPU (A100 with 80 GB VRAM)

Our single-number checker reaches beyond the published frontier
for individual numbers, but this is not equivalent to exhaustive
range verification.

---

## Known theoretical results

- **Ternary Goldbach conjecture** (Helfgott, 2013): every odd
  number > 5 is the sum of three primes. Formally proved.
- **Binary Goldbach** (our problem): every even number > 2 is
  the sum of two primes. Not yet proved.
- Best unconditional result: every sufficiently large even number
  has a Goldbach partition (Vinogradov, 1937) — but "sufficiently
  large" is not bounded usefully for computation.

---

## Roadmap

- [x] CPU baseline verifier
- [x] GPU range verifier with byte array (to 10^9)
- [x] Compact prime bitset (16x memory reduction)
- [x] GPU range verifier with bitset (to 10^11)
- [x] Single number checker with Miller-Rabin (to 1.8 × 10^19)
- [ ] Parallel CPU sieve with OpenMP (target: 8-16x sieve speedup)
- [ ] Segmented bitset (removes VRAM ceiling, target: 10^12+)
- [ ] Count partitions c(n) at scale
- [ ] Cloud GPU run (A100, target: push range frontier)
- [ ] Write-up for arXiv or JOSS
