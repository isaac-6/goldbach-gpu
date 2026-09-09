[![DOI](https://zenodo.org/badge/1162873542.svg)](https://doi.org/10.5281/zenodo.18786328)
[![arXiv](https://img.shields.io/badge/arXiv-2603.07850_(RangeVerification)-b31b1b.svg)](https://arxiv.org/abs/2603.07850)
[![arXiv](https://img.shields.io/badge/arXiv-2603.02621_(BigCheck)-b31b1b.svg)](https://arxiv.org/abs/2603.02621)

[![Release](https://img.shields.io/github/v/release/isaac-6/goldbach-gpu)](https://github.com/isaac-6/goldbach-gpu/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Language: C++](https://img.shields.io/badge/Language-C%2B%2B-blue.svg)
![CUDA: 12.x / 13.x](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green.svg)

# GoldbachGPU

Exhaustive GPU verification of Goldbach's conjecture: every even integer in a
range is checked for a representation as the sum of two primes.

Verifies every even number from 4 to 10<sup>14</sup> in **15 minutes 33 seconds**
on a single RTX 5090, with no counterexamples found. On four GPUs, 10<sup>13</sup>
takes 20.9 seconds at 99.96% parallel efficiency.

This does not approach the research frontier: Oliveira e Silva, Herzog and Pardi
verified to 4×10<sup>18</sup> in 2014 using a distributed CPU cluster over several
years. The contribution here is that a comparable class of computation runs on
hardware an individual can own, in minutes rather than machine-years.

---

## Results

### Single GPU

RTX 5090, CUDA 13.3, Ubuntu 26.04 (WSL2). Five runs per limit, mean ± sample
standard deviation. Zero Phase 2 fallbacks throughout.

| Limit | Computation | Wall clock |
|---|---|---|
| 10<sup>10</sup> | 0.087 ± 0.001 s | 0.56 s |
| 10<sup>11</sup> | 0.809 ± 0.002 s | 1.27 s |
| 10<sup>12</sup> | 8.44 ± 0.03 s | 8.89 s |
| 10<sup>13</sup> | 88.79 ± 0.14 s | 89.24 s |
| 10<sup>14</sup> | 932.8 s | 933.3 s |

Every row above is a plain verification run: no `--record-check`, no profiler.
At 10<sup>10</sup> the computation is only 0.087 s against 0.56 s wall clock, so
most of the wall time is fixed startup. That row should not be read as a
throughput figure.

Scaling stays close to linear across the whole ladder, including the last decade:
10<sup>13</sup> → 10<sup>14</sup> is 10.51×. Normalised per segment the cost is
3.24, 3.38, 3.55 and 3.73 ms at 10<sup>11</sup> through 10<sup>14</sup>; 5%
rise over the final decade, not a regime change.

There is a real effect underneath: above 10<sup>12</sup> the sieve bound √N
overtakes `--p-small`, so the number of sieving primes grows (78,498 → 227,647 →
664,579) and the large-prime kernel's share of GPU time rises from 8% at
10<sup>11</sup> to 29% at 10<sup>14</sup>. Phase 1 is unaffected, since its prime
list stays capped at `--p-small`. The sieve is absorbing that growth well so far,
but it is the term that will dominate first at larger limits.

### Multiple GPUs

4× RTX 5090, CUDA 12.8, Ubuntu 24.04, AMD EPYC 7302P. Five runs per
configuration. Efficiency is η<sub>k</sub> = T₁ / (k · T<sub>k</sub>) on
computation time, with T₁ measured on the same node.

| Limit | GPUs | Computation | Speedup | Efficiency |
|---|---|---|---|---|
| 10<sup>12</sup> | 1 | 7.825 s | — | — |
| | 2 | 3.910 s | 2.00× | ~100% |
| | 4 | 1.969 s | 3.97× | 99.4% |
| 10<sup>13</sup> | 1 | 83.724 s | — | — |
| | 2 | 41.740 s | 2.01× | ~100% |
| | 4 | 20.941 s | 4.00× | 99.96% |

The 2-GPU rows measure marginally above 100%, which is measurement noise rather
than superlinear scaling. On wall clock the 4-GPU 10<sup>13</sup> figure is
22.60 s against 20.94 s of computation: the ~1.6 s of startup is fixed
regardless of GPU count, so wall-clock efficiency is 92.7% where computation
efficiency is ~100%.

Work is distributed by a lock-free atomic counter (each GPU claims the next
segment when it finishes the previous one) so devices of different speeds
balance automatically and no GPU waits on another.

---

## Building

Requires a CUDA-capable GPU, CUDA 12.x or newer, CMake 3.18+, a C++17 compiler,
GMP and OpenMP.

```bash
sudo apt install -y cmake g++ libgmp-dev libomp-dev
git clone https://github.com/isaac-6/goldbach-gpu.git
cd goldbach-gpu && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

If CMake cannot determine your GPU's compute capability it will stop and ask you
to pass it explicitly:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
```

On a host with glibc 2.41 or newer, CUDA 12.x headers conflict with the C23
declarations of `cospi`, `sinpi` and `rsqrt`. Use CUDA 13.x there.

---

## Running

```bash
./bin/goldbach 1000000000000
```

Verifies every even number from 4 to the given limit. Useful options:

| Option | Effect |
|---|---|
| `--gpus=N` | Use N GPUs (`-1` for all). Default 1. |
| `--start=N` | Begin at N rather than 4, for splitting work across machines. |
| `--seg-size=N` | Even integers per segment. Derived from free VRAM if omitted. |
| `--p-small=N` | Prime search bound for the GPU phase. Default 10<sup>6</sup>. |
| `--batch-size=N` | Primes uploaded per Phase 1 kernel launch. Default 10<sup>5</sup>. |
| `--record-check` | Print each new maximum p<sub>min</sub> as it is found. |
| `--progress` | Live throughput and estimated completion. |
| `--primetest=bpsw\|mr` | Primality fallback. Default BPSW. |

A representative invocation:

```bash
./bin/goldbach 10000000000000 --seg-size=200000000 --p-small=1000000 \
               --batch-size=2000000 --gpus=4 --progress
```

---

## How it works

For each even *n*, the program searches small primes *p* in ascending order for
one where *n − p* is also prime. Almost every even number has such a partition
with a very small *p*: the largest minimal prime below 10<sup>14</sup> is 4909,
at *n* = 76903574497118, found by the `--record-check` run and matching the
published record table. That is a factor of 200 below the 10<sup>6</sup> search
bound, which is why the CPU fallback is never reached.

The range is processed in segments. For each segment the GPU builds a bitset of
the primes needed to answer those queries, then verifies every even number in the
segment against it. Both steps run entirely on the device; the host sends a
prime list and receives a 4-byte count per segment.

Two ideas account for most of the performance:

**Byte-wide marking during sieving.** Clearing a bit is a read-modify-write, so
concurrent threads sieving different primes into the same 64-bit word lose each
other's updates unless the operation is atomic, and the atomic serialises them.
Giving each candidate its own byte during construction makes marking a plain
store of a constant, which needs no atomic because concurrent stores of the same
value cannot conflict. The result is packed back to bits before leaving shared
memory, so the stored representation is unchanged.

**Transposed verification.** For 64 consecutive even numbers and a fixed prime
*p*, the complements *n − p* are 64 consecutive odd numbers: exactly one 64-bit
window of the prime bitset. One shifted load and one OR therefore resolves 64
numbers against a prime at once. Because a thread continues until all 64 of its
numbers are settled, and hard numbers are rare, widening the group costs far less
than it saves: measured warp-iterations per even number fall from about 50 to
1.85.

Sieving primes are also split by size, so that the large majority which mark at
most one position per tile are handled once per segment rather than rescanned for
every tile.

A CPU fallback exists for any number the GPU phase does not resolve. It has not
been reached at any limit tested.

---

## Correctness

Verification is only as good as its checks, and a verifier that is silently wrong
produces exactly the same output as one that is right. The repository therefore
carries five tests, each comparing a component against an independent
implementation rather than against itself:

| Test | What it checks |
|---|---|
| `test_gpu_sieve` | GPU segment sieve against an independently written CPU sieve, over fixed and randomised ranges including boundary cases. |
| `test_phase1` | GPU verification against a CPU reference. Compares across prime-list prefixes, so the comparison resolves *which* prime succeeded rather than the saturated yes/no verdict. |
| `test_primality` | Baillie–PSW against the 12-base deterministic Miller–Rabin, with emphasis above 2<sup>63</sup>. |
| `test_bitset_race` | Repeated parallel bitset construction against a single-threaded reference, at both word-aligned and misaligned thread boundaries. |
| `test_records` | Minimal primes against 48 published record values computed independently by Oliveira e Silva. |

```bash
./bin/test_gpu_sieve && ./bin/test_phase1 && ./bin/test_primality \
  && ./bin/test_bitset_race && ./bin/test_records
```

The `--record-check` flag extends this to a live run. It reports each new maximum
minimal prime as it is found; a separate 10<sup>14</sup> run with the flag set
emitted 22 such records, all matching the published table, six of them above
10<sup>13</sup> where the CPU-side test does not reach. The flag costs 26.5% at
10<sup>14</sup> (43.6% at 10<sup>11</sup>, falling as N grows because the
tracking is proportional to Phase 1, whose share of runtime shrinks), so it is
off by default and the timings above are measured without it.

Under `--gpus>1` segments complete out of order, so a later segment can raise the
running maximum and permanently suppress a genuine earlier record. The surviving
set is scheduling-dependent. **Use a single GPU when the record sequence is being
used for validation**; multi-GPU runs remain correct for verification itself.

Primality is decided by a bitset lookup wherever possible, and otherwise by
Baillie–PSW or a 12-base deterministic Miller–Rabin. The Miller–Rabin base set is
*proved* deterministic for all *n* < 2<sup>64</sup>; Baillie–PSW is verified to
have no counterexample below 2<sup>64</sup> but has no such proof, which is why
`test_primality` cross-checks it against Miller–Rabin rather than trusting it.

---

## Tuning

`TILE_ODDS` and `SPLIT_THRESHOLD` are compile-time constants, overridable with
`-DCMAKE_CUDA_FLAGS="-DSPLIT_THRESHOLD=65536"`. The defaults were chosen by
sweeping at 10<sup>13</sup> on an RTX 5090.

The optimum for `SPLIT_THRESHOLD` depends on how many sieving primes there are,
which grows as √N above 10<sup>12</sup>, so a different limit may prefer a
different value. Measured at 10<sup>13</sup>, relative to the optimum of 65536:

| Step from optimum | Value | Cost |
|---|---|---|
| 1 below | 32768 | +9.9% |
| 1 above | 131072 | +1.3% |
| 2 above | 262144 | +13.6% |
| 3 above | 524288 | +40.7% |
| 4 above | 1048576 | +95.2% |

The curve is flat between 65536 and 131072 and rises steeply outside that
range. The optimum depends on how many sieving primes there are, which grows
as sqrt(N) above 1e12, so a different limit may prefer a different value.

`--seg-size` mainly trades memory for parallelism and is flat near the default.

---

## Repository layout

```
src/goldbach.cu           Main verifier
include/sieve_kernel.cuh  Segment sieve (tiled and large-prime kernels)
include/phase1_kernel.cuh Verification kernels (transposed and scalar)
include/primality.cuh     Miller-Rabin and Baillie-PSW, device and host
src/test_*.c*             The five tests described above
src/analyze_pmin.cpp      Distribution of minimal primes; used to size the
                          search bound and to predict kernel cost
```

Also included, from earlier stages of the project: `cpu_goldbach` (a CPU
verifier), `big_check` (single very large numbers via GMP, beyond 64 bits),
`single_check` (one number on the GPU), and `legacy/goldbach_gpu3.cu`, the
previous host-coupled implementation, retained for comparison and built with
`-DBUILD_LEGACY=ON`.

---

## How to cite

If you use this software in academic work, please cite the archived release:

```
Llorente-Saguer, I. (2026). GoldbachGPU (v3.0.0) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.XXXXXXXX
```

For the latest version, see the concept DOI:
https://doi.org/10.5281/zenodo.18786328

If you reference the scientific description of the range verification
(`goldbach`), please cite:

```
Llorente-Saguer, I. (2026). A Lock-Free, Fully GPU-Resident Architecture for
the Verification of Goldbach's Conjecture. https://arxiv.org/abs/2603.07850
```

If you reference the single large number verification method (`big_check`) or
the previous CPU-GPU hybrid implementation (`goldbach_gpu3`), please cite:

```
Llorente-Saguer, I. (2026). GoldbachGPU: High-performance Goldbach verification
on GPUs. https://arxiv.org/abs/2603.02621
```

**Note on versions.** The preprints above describe release v2.0.0. Two
concurrency defects have since been found and fixed in that code (see
[CHANGELOG.md](CHANGELOG.md)) and a corrigendum reporting corrected timings is
in preparation. The performance figures on this page come from the current
release and are not comparable to those in the preprints, which describe a
different implementation. A manuscript covering the optimisations is also in
preparation.

---

## References

[1] T. Oliveira e Silva, S. Herzog, S. Pardi, "Empirical verification of the even
Goldbach conjecture and computation of prime gaps up to 4×10¹⁸",
*Mathematics of Computation*, 83(288):2033–2060, 2014.

[2] T. Oliveira e Silva, *Goldbach conjecture verification*, record data file
`https://sweet.ua.pt/tos/goldbach/t0.txt.gz`, linked from
https://sweet.ua.pt/tos/goldbach.html (retrieved 2026-09-09). The record table
is in the data file, not on the page itself. Used by `test_records`.

[3] R. Baillie, S. S. Wagstaff Jr., "Lucas pseudoprimes", *Mathematics of
Computation*, 35(152):1391–1417, 1980.

---

## License

MIT
