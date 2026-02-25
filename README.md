![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Language: C++](https://img.shields.io/badge/Language-C%2B%2B-blue.svg)
![CUDA: 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)

# goldbach-gpu

GPU-accelerated verification of Goldbach's conjecture using CUDA and GMP.

**Goldbach's conjecture** states that every even integer greater than 2 can be
expressed as the sum of two prime numbers. First proposed in 1742, it remains
one of the oldest unsolved problems in mathematics. Verified computationally up
to 4×10¹⁸ by distributed computing [[1]](#references), but never formally proved.

This project provides an open source, reproducible framework for exhaustive
range verification and arbitrary precision single-number checking on consumer
hardware. **No counterexamples have been found in any computation.**

---

## Headline result
Figure shows times to check even numbers up to N.
<img width="2814" height="1132" alt="image" src="https://github.com/user-attachments/assets/91c4da61-11fb-40d7-ae1b-5fb756d836e0" />

> **All even integers up to 10¹² verified on a single NVIDIA RTX 3070 (8 GB VRAM)
> in 96 minutes. Zero failures. Zero fallbacks.**

This is achieved by `goldbach_gpu3`, a segmented double-sieve verifier that
removes the VRAM ceiling of earlier designs. See [Architecture](#architecture)
and [Results](#results-summary) below.

---

## Project goals

- Verify Goldbach's conjecture to the largest range feasible on consumer hardware
- Demonstrate GPU-accelerated number theory with rigorous correctness guarantees
- Benchmark and compare CPU vs GPU architectures for sieve-based computation
- Provide fully reproducible scientific results on a fixed hardware platform
- Serve as a reference implementation for GPU Goldbach verification

---

## Architecture

The project contains five tools of increasing capability, each validating
against the previous:

| Tool | Method | Range | Hard limit |
|------|--------|-------|------------|
| `goldbach` | CPU sieve + sequential scan | exhaustive | time |
| `goldbach_gpu` | GPU, byte array prime table | exhaustive | VRAM (~10⁹) |
| `goldbach_gpu2` | GPU, compact bitset (1 bit/odd) | exhaustive | VRAM (~10¹¹) |
| `goldbach_gpu3` | GPU, segmented double-sieve | exhaustive | time only |
| `single_check` | GPU Miller-Rabin | single number | uint64_t (~1.8×10¹⁹) |
| `big_check` | GMP + OpenMP | single number | time only |

### Design evolution

**goldbach_gpu** stores primality as one byte per number. At 10⁹ this requires
953 MB VRAM -- already approaching limits. Not usable beyond 10⁹.

**goldbach_gpu2** encodes one bit per odd number, a 16× memory reduction.
This makes 10¹¹ feasible (5,960 MB) but 10¹² impossible (would need 59 GB).
The VRAM ceiling is architectural -- no amount of runtime can overcome it.

**goldbach_gpu3** removes the ceiling entirely via a segmented design. For each
segment [A, B] of even numbers, it iterates over candidate primes p and marks
all n in [A, B] where n-p is prime as verified. Crucially, q = n-p is checked
globally (small bitset, segment bitset, or Miller-Rabin) -- not forced into
the current segment, which would miss valid partitions. Each segment uses only
~60 MB VRAM regardless of total range. An exhaustive CPU fallback guarantees
rigor for any number the GPU phase does not resolve; in all runs to date,
this fallback has never been triggered.

**big_check** handles numbers of arbitrary digit count using GMP exact
arithmetic and probabilistic Miller-Rabin (25 rounds, false positive
probability < 10⁻¹⁵). OpenMP parallelism with batch-synchronised prime
search prevents thread runaway for large inputs.

---

## Results summary

Full benchmark log: [RESULTS.md](RESULTS.md)

### Range verification -- exhaustive, all even n in [4, N]

| Tool | Limit | Even n checked | Total time | Failures |
|------|-------|----------------|------------|----------|
| `goldbach` (CPU) | 10⁹ | 499,999,999 | 22,622 ms | 0 |
| `goldbach` (CPU) | 10¹⁰ | 4,999,999,999 | 308,335 ms | 0 |
| `goldbach_gpu2` | 10⁹ | 499,999,999 | 1,386 ms | 0 |
| `goldbach_gpu2` | 10¹⁰ | 4,999,999,999 | 25,443 ms | 0 |
| `goldbach_gpu3` | 10¹² | 499,999,999,999 | 5,760,350 ms | 0 |

GPU speedup over CPU baseline: **16× total at 10⁹, 12× at 10¹⁰**.
goldbach_gpu3 reaches 10¹² which no single-GPU implementation can reach
with a global bitset design on 8 GB VRAM.

### Single number verification

| Tool | n | Partition | Time |
|------|---|-----------|------|
| `single_check` | 10¹⁸ | 14,831 + 999,999,999,999,985,169 | 1.5 s |
| `single_check` | 10¹⁹ | 226,283 + 9,999,999,999,999,773,717 | 1.5 s |
| `big_check` | 10¹⁰⁰⁰ | p = 26,981 | 363 ms (20 threads) |
| `big_check` | 10¹⁰⁰⁰⁰ | p = 47,717 | 182 s (20 threads) |

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Dependencies:** CUDA toolkit, GMP, OpenMP.

```bash
sudo apt install libgmp-dev    # GMP
# CUDA toolkit: https://developer.nvidia.com/cuda-downloads
```

---

## Usage

### CPU range verifier
```bash
./goldbach
```
Edit `LIMIT` in `src/cpu_goldbach.cpp`. Useful as correctness baseline.
Expected time: ~23 s at 10⁹, ~5 min at 10¹⁰.

### GPU range verifier -- compact bitset (recommended up to 10¹¹)
```bash
./goldbach_gpu2
```
Edit `LIMIT` in `src/goldbach_gpu2.cu`. Requires VRAM ≥ N/16 bytes.
Expected time: ~1.4 s at 10⁹, ~25 s at 10¹⁰, ~8 min at 10¹¹.

### GPU segmented verifier (recommended for 10¹¹ and beyond)
```bash
./goldbach_gpu3
```
Edit `LIMIT`, `SEG_SIZE`, and `P_SMALL` in `src/goldbach_gpu3.cu`.
No VRAM ceiling. Expected time: ~96 min at 10¹².

To reproduce the 10¹² result:
```cpp
// in goldbach_gpu3.cu
const uint64_t LIMIT    = 1'000'000'000'000ULL;
const uint64_t SEG_SIZE = 500'000'000ULL;
const uint64_t P_SMALL  = 2'000'000ULL;
const uint64_t P_BATCH  = 100'000ULL;
```

### GPU single number checker (up to ~1.8×10¹⁹)
```bash
./single_check                         # default: 10¹²
./single_check 999999999999999998      # any even uint64_t
```
Uses deterministic Miller-Rabin with 12 witnesses. Returns a valid
partition but not necessarily the minimal one (concurrent threads).

### Arbitrary precision checker (any size)
```bash
./big_check                            # default: 10^50
./big_check "$(python3 -c "print('1' + '0'*1000)")"    # 10^1000
./big_check "$(python3 -c "print('1' + '0'*10000)")"   # 10^10000
./big_check "123456789012345678901234567890"            # arbitrary even number
```

Candidate primes p are generated deterministically by a standard Sieve
of Eratosthenes up to 10^7 (664,579 primes). For each p, q = n - p is
computed exactly using GMP arbitrary precision arithmetic, then tested
for primality using GMP's Miller-Rabin with 25 rounds (false positive
probability < 4^-25 ≈ 10^-15 per test). Since p is sieve-generated,
only the test on q is probabilistic.

Primes are processed in batches of 1,000 with a synchronisation barrier
between batches. This prevents thread runaway: without batching, fast
threads on small-p tests could race far ahead of slow threads doing
expensive GMP tests, causing a large non-minimal p to be returned first.
With batching, the winning p is the smallest in the first batch containing
a valid partition.

Practical limit: GMP primality tests scale roughly as O(d^3) in digit
count d. At 10^10000 (d = 10,001) each test takes milliseconds; beyond
~10^50000 each test takes seconds and exhaustive search becomes
infeasible in practice.

### Validation tests
```bash
./test_bitset    # validates prime bitset encoding correctness
./test_sieve     # validates segmented sieve correctness
```

---

## Reproducibility

All results were produced on the following fixed platform:

| Component | Specification |
|-----------|---------------|
| CPU | Intel i7-12700H, 20 logical threads |
| GPU | NVIDIA RTX 3070, 8 GB VRAM, 448 GB/s bandwidth |
| RAM | 32 GB |
| OS | WSL2, Ubuntu 24.04 |
| CUDA | Record your version: `nvcc --version` |
| GCC | Record your version: `gcc --version` |
| GMP | Record your version: `apt show libgmp-dev` |

To record your environment:
```bash
nvcc --version
gcc --version
apt show libgmp-dev 2>/dev/null | grep Version
```

All benchmark configurations (LIMIT, SEG_SIZE, P_SMALL, thread counts)
are documented in [RESULTS.md](RESULTS.md) for each run.

---

## Project structure

```
src/
  cpu_goldbach.cpp      # CPU range verifier (baseline)
  goldbach_gpu.cu       # GPU range verifier (byte array, historical)
  goldbach_gpu2.cu      # GPU range verifier (compact bitset, to 10^11)
  goldbach_gpu3.cu      # GPU segmented verifier (no VRAM ceiling, to 10^12+)
  single_check.cu       # GPU single number checker (Miller-Rabin, uint64_t)
  big_check.cpp         # Arbitrary precision checker (GMP + OpenMP)
  prime_bitset.hpp      # Compact prime bitset (1 bit per odd number)
  prime_bitset.cpp      # Bitset construction (segmented sieve + OpenMP)
  segmented_sieve.cpp   # Segmented sieve of Eratosthenes
  test_bitset.cpp       # Bitset correctness tests
  test_sieve.cpp        # Sieve correctness tests
```

---

## How to cite

If you use this work, please cite:

```
Llorente-Saguer, I. (2026). An Open Source GPU-Accelerated Framework for
Verification of Goldbach's Conjecture. arXiv preprint [forthcoming].
https://github.com/isaac-6/goldbach-gpu
```

A preprint will be submitted to arXiv shortly.

---

## References

[1] T. Oliveira e Silva, S. Herzog, S. Pardi, "Empirical verification of the
even Goldbach conjecture and computation of prime gaps up to 4×10¹⁸",
*Mathematics of Computation*, 83(288):2033–2060, 2014.

---

## License

MIT