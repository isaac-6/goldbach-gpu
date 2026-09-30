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

Verifies every even number from 4 to 10<sup>14</sup> in **14 minutes 17 seconds**
(857 s) on a single RTX 5090, with no counterexamples found; this run also counts
the primes up to 10<sup>14</sup> and checks the record sequence, and its count,
3,204,941,750,802, equals primecount's. Windows of up to 10<sup>15</sup> even
numbers starting at 4·10<sup>18</sup> have been verified and prime-counted the
same way (see Results). On four GPUs, measured on a separate node at an earlier
commit, 10<sup>13</sup> takes 20.9 seconds at 99.95% parallel efficiency.

This does not approach the research frontier: Oliveira e Silva, Herzog and Pardi
verified to 4×10<sup>18</sup> in 2014 using a distributed CPU cluster over several
years. The contribution here is that a comparable class of computation runs on
hardware an individual can own, in minutes rather than machine-years.

---

## Results

### Single GPU

v3.2.0, RTX 5090 (driver 610.88), CUDA 13.3, Ubuntu 26.04 (WSL2), Ryzen 7 9800X3D.
Wall clock from `/usr/bin/time`, mean ± sample standard deviation, zero Phase 2
fallbacks throughout, prime counting on (the default). Flags:
`--seg-size=200000000 --p-small=1000000 --batch-size=2000000`. The full method,
including the interleaved comparison with v2.0.2, is in
[RESULTS.md](RESULTS.md).

| Limit | v3.2.0 wall clock | Runs | v2.0.2 wall clock | Speedup (variable cost) |
|---|---|---|---|---|
| 10<sup>11</sup> | 1.160 ± 0.007 s | 5 | 11.348 ± 0.018 s | 15.97× |
| 10<sup>12</sup> | 7.978 ± 0.052 s | 5 | 143.890 ± 0.662 s | 19.13× |
| 10<sup>13</sup> | 79.523 ± 0.067 s | 3 | 3,204.710 ± 4.523 s | 40.54× |
| 10<sup>14</sup> | 857 s | 1 | not run | |

The v2.0.2 and v3.2.0 rows at each limit were run interleaved in one session.
Speedup is the ratio of variable costs, that is of mean wall clock at the limit
minus mean wall clock at 10<sup>6</sup> of the same build (0.488 s for v2.0.2,
0.480 s for v3.2.0), which removes the fixed startup. The 10<sup>14</sup> row is
a single run made with `--record-check`. Results for v3.1.0 are in RESULTS.md,
section 3.

Absolute times on this machine moved by about 2% between sessions, with both
builds moving together, so ratios are quoted only within a session.

Mean wall clock grows by a factor of 9.97 from 10<sup>12</sup> to 10<sup>13</sup>
for v3.2.0, against 22.27 for v2.0.2. Part of the growth is the sieving prime
count: above 10<sup>12</sup> the sieve bound √N overtakes `--p-small`, and the run
at 10<sup>14</sup> reports 664,579 sieving primes (`small_prime_count`).
Phase 1 is unaffected, since its prime list stays capped at `--p-small`, which is
78,498 primes.

### Windows at 4·10<sup>18</sup>

`--start=4000000000000000000 --window-max`, with N = 4·10<sup>18</sup> + W. Every
even number in the closed interval [4·10<sup>18</sup>, N] is checked, and the
count is of primes q with 4·10<sup>18</sup> < q ≤ N. Each window was run twice
with different segment and prime-bound parameters (runs A and B, listed in
RESULTS.md, section 2.3); the two runs agree exactly, and the prime count equals
the difference of primecount 8.8 values.

| W | Numbers checked | Primes in (4·10<sup>18</sup>, N] | Window maximum p<sub>min</sub> at n | Wall clock A / B |
|---|---|---|---|---|
| 10<sup>12</sup> | 500,000,000,001 | 23,346,564,662 | 6,073 at 4,000,000,402,622,391,632 | 13.54 s / 12.64 s |
| 10<sup>13</sup> | 5,000,000,000,001 | 233,465,239,142 | 6,421 at 4,000,005,756,560,737,472 | 113.05 s / 106.39 s |
| 10<sup>14</sup> | 50,000,000,000,001 | 2,334,657,870,388 | 7,103 at 4,000,048,374,322,888,936 | 1,110.87 s / 1,044.38 s |
| 10<sup>15</sup> | 500,000,000,000,001 | 23,346,512,560,823 | 7,487 at 4,000,400,837,678,526,154 | 11,125.3 s / 10,404.2 s |

All eight runs report 0 Phase 2 fallbacks. The 10<sup>15</sup> runs took 3.09 h and
2.89 h. During them the GPU sat at a median SM clock of 2,760 MHz with no throttle
reason active in any of 359 samples. The window maximum is the largest
p<sub>min</sub> over the window and is not a p-record. These runs cover windows
that start at 4·10<sup>18</sup>; they make no claim about the range between
10<sup>14</sup> and 4·10<sup>18</sup>.

### Multiple GPUs

4× RTX 5090, CUDA 12.8, Ubuntu 24.04, AMD EPYC 7302P. Five runs per
configuration. Efficiency is η<sub>k</sub> = T₁ / (k · T<sub>k</sub>) on
computation time, with T₁ measured on the same node.

| Limit | GPUs | Computation | Speedup | Efficiency |
|---|---|---|---|---|
| 10<sup>12</sup> | 1 | 7.8252 ± 0.0156 s | — | — |
| | 2 | 3.9096 ± 0.0042 s | 2.0015× | 100.08% |
| | 4 | 1.9686 ± 0.0012 s | 3.9750× | 99.38% |
| 10<sup>13</sup> | 1 | 83.7240 ± 0.0504 s | — | — |
| | 2 | 41.7403 ± 0.0687 s | 2.0058× | 100.29% |
| | 4 | 20.9406 ± 0.0221 s | 3.9982× | 99.95% |

Values transcribed from the session output of a rented node; raw logs were not
retained.

The 2-GPU rows measure marginally above 100%, which is measurement noise rather
than superlinear scaling. On wall clock the 4-GPU 10<sup>13</sup> figure is
22.5960 s against 20.9406 s of computation: the ~1.66 s of startup is fixed
regardless of GPU count, so computation is 92.67% of wall clock where computation
efficiency against one GPU is 99.95%.

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
| `--start=N` | Begin at N rather than 4, for splitting work across machines. At most the limit. |
| `--seg-size=N` | Even integers per segment: even, below 2<sup>32</sup>. Derived from free VRAM if omitted. |
| `--p-small=N` | Prime search bound for the GPU phase, 3 to 4·10<sup>9</sup>. Default 10<sup>6</sup>. |
| `--batch-size=N` | Primes uploaded per Phase 1 kernel launch, 1 to 2<sup>32</sup>. Default 10<sup>5</sup>. |
| `--record-check` | Print each new maximum p<sub>min</sub> as it is found, and the overall maximum at the end. Requires the default `--start`. |
| `--window-max` | Print the largest p<sub>min</sub> over [start, limit] and the smallest *n* attaining it, for any `--start`. Numbers resolved by the CPU fallback are included. A window maximum, not a p-record. |
| `--no-count-primes` | Do not count primes. By default every run also prints π(limit), counted from the segment sieve, or, with a `--start` above the small-prime bound (about √limit), the count of primes in (start, limit]. Counting costs 0.2% at 10<sup>12</sup> and 0.4% at 10<sup>13</sup>. `--count-primes` is still accepted and does nothing. |
| `--count-file=F` | Write each segment's range and prime count to F. Needs counting on. |
| `--progress` | Live throughput and estimated completion. |
| `--primetest=mr\|bpsw` | Primality test in the CPU fallback (Phase 2) for q above 10<sup>8</sup>. Default MR, the proved 12-base Miller–Rabin. |

Invalid values, including negative numbers and unknown options, are rejected
with exit status 1 before any work starts.

### Supported range

The limit N may be any integer from 4 to 2<sup>64</sup> − 2<sup>33</sup> =
18,446,744,065,119,617,024; an odd N verifies up to N − 1. The bound keeps every
segment boundary, sieve bound and primality-test operand below 2<sup>64</sup>.
With G GPUs, N + 2·`--seg-size`·(G + 1) must also stay below 2<sup>64</sup>, so
that the shared segment counter cannot wrap; the program checks both and refuses
anything outside them. Time, not arithmetic, is the practical limit.

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
at *n* = 76903574497118, found by the `--record-check` run of RESULTS.md,
section 2.2. That is a factor of 200 below the 10<sup>6</sup> search
bound, which is why the CPU fallback is never reached.

The range is processed in segments. For each segment the GPU builds a bitset of
the primes needed to answer those queries, then verifies every even number in the
segment against it. Both steps run entirely on the device; the host sends a
prime list and receives two counts per segment, verified and unverified, which
must add up to the segment's size.

Two ideas account for most of the performance:

**Byte-wide marking during sieving.** Clearing a bit is a read-modify-write, so
concurrent threads sieving different primes into the same 64-bit word lose each
other's updates unless the operation is atomic, and the atomic serialises them.
Giving each candidate its own byte during construction makes marking a store of
a constant rather than a read-modify-write. Two threads can still store to the
same byte, and under the CUDA memory model two plain stores to one location
form a data race even when they store the same value. Each mark is therefore a
relaxed, block-scoped atomic byte store (`st.relaxed.cta.shared.b8`), which is
race-free and compiles to the same single-byte store instruction, with no
read-modify-write. The result is packed back to bits before leaving shared
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

A CPU fallback (Phase 2) handles any number the GPU phase does not resolve. It
searches primes p ≤ 10<sup>8</sup>, so it is exhaustive for n ≤ 2·10<sup>8</sup>;
if it finds nothing, the run ends with "no partition with p ≤ 10^8 found for
n = …" and exit status 1, a search-limit result rather than a counterexample. It
has not been reached at any limit tested with the default `--p-small`. Every
segment must account for all of its numbers as verified or unverified before
the run can succeed; any shortfall, or any CUDA error, ends the run with exit
status 1.

---

## Correctness

Verification is only as good as its checks, and a verifier that is silently wrong
produces exactly the same output as one that is right. The repository therefore
carries fifteen tests, most comparing a component against an independent
implementation or published data rather than against itself:

| Test | What it checks |
|---|---|
| `test_gpu_sieve` | GPU segment sieve against an independently written CPU sieve, over fixed and randomised ranges including boundary cases. |
| `test_phase1` | GPU verification against a CPU reference. Compares across prime-list prefixes, so the comparison resolves *which* prime succeeded rather than the saturated yes/no verdict. Walks ranges in segments with the verifier's own segment geometry, including a partial last segment and a switch between the two kernels, at batch sizes down to one prime per launch. Also pins the p<sub>min</sub> maximum's tie rule: an equal p<sub>min</sub> at a smaller *n* must replace the stored one. |
| `test_primality` | Baillie–PSW against the 12-base deterministic Miller–Rabin, device and host, with emphasis above 2<sup>63</sup>; plus inputs with known verdicts (confirmed with sympy): the smallest number that is a strong pseudoprime to every prime base up to 31, and primes near 2<sup>64</sup> whose parameter search needs up to 82 steps; and the V<sub>n+1</sub> ≡ 2Q check against the Lucas-V pseudoprimes tabulated by Baillie, Fiori and Wagstaff. |
| `test_bpsw_spsp` | Every base-2 strong pseudoprime below 2<sup>32</sup> (2,314, generated independently) and the first ten strong Lucas pseudoprimes published by Baillie, Fiori and Wagstaff must be rejected by Baillie–PSW on device and host, and 126,897 primes accepted. |
| `test_mr_spsp` | The Miller–Rabin half of the same controls, as its own test because Miller–Rabin is the default in Phase 2: every base-2 strong pseudoprime below 2<sup>32</sup> and the ten strong Lucas pseudoprimes rejected, the 126,897 primes accepted, on device and host. |
| `test_bitset_race` | Repeated parallel bitset construction against a single-threaded reference, at both word-aligned and misaligned thread boundaries. |
| `test_sieve`, `test_bitset` | The CPU segmented sieve and the prime bitset against known π(n). |
| `test_records` | The CPU definition of p<sub>min</sub> against 48 published record values computed independently by Oliveira e Silva. |
| `test_record_check` | `goldbach --record-check` to 10<sup>8</sup> at three parameter sets against a brute-force record list: the output must be a subsequence and include the maximum. A fourth run pins tie-breaking: two numbers share a segment's maximum p<sub>min</sub>, and the smaller must be reported. |
| `test_window` | Window mode at 4·10<sup>18</sup> and 10<sup>18</sup>: the prime count of a 10<sup>8</sup>-wide window against `primesieve`, and its largest p<sub>min</sub> against a GMP brute force; plus three windows whose largest p<sub>min</sub> is tied, on the transposed, scalar and CPU-fallback paths, where the smallest *n* must be reported. |
| `test_count_primes` | `goldbach`'s prime count, on by default, against known π(N), over many segments and from a non-default `--start`. |
| `test_phase2_fallback` | With `--p-small=3`, exactly 421,501 numbers to 10<sup>6</sup> must reach the CPU fallback, and the run must still succeed; and 89,098 in a range above 10<sup>8</sup>, where the fallback tests q with Miller–Rabin or, with `--primetest=bpsw`, Baillie–PSW. |
| `test_cli` | Every invalid command line of `goldbach`, `big_check` and `single_check` exits 1 with its message; edge cases still run. |
| `test_big_check` | `big_check` against the same 48 published records, plus small-*n* edges, the search-limit exit, thread-count determinism and expression input. Reads the record table out of `test_records.cpp` rather than copying it. |

All are registered with CTest, so the whole suite runs with:

```bash
ctest --output-on-failure
```

or individually:

```bash
./bin/test_gpu_sieve && ./bin/test_phase1 && ./bin/test_primality \
  && ./bin/test_bpsw_spsp bpsw && ./bin/test_bpsw_spsp mr && ./bin/test_bitset_race && ./bin/test_records
```

The `--record-check` flag extends this to a live run. It reports each new maximum
minimal prime as it is found; a separate 10<sup>14</sup> run with the flag set
emitted 22 such records. The 16 below 10<sup>13</sup> form a subsequence of the
published table; the six above 10<sup>13</sup>, where the CPU-side test does not
reach, were not compared with a table (RESULTS.md, section 2.2). At v3.0.0 the flag
cost 26.6% at 10<sup>14</sup>, and at v3.2.0 as first written 36% at
10<sup>12</sup>: every GPU thread ended with an atomic maximum on one address.
Skipping the atomic when it cannot raise the stored value brings `--record-check`
and `--window-max` to 2.2% at 10<sup>12</sup> (7.53 s against 7.37 s, five runs
each). Both are off by default, and the timings above are measured without them,
except the 10<sup>14</sup> run of the Results section, which used
`--record-check`.

The flag reports at most one record per segment, so its output is a
subsequence of the true records. Numbers resolved by the CPU fallback contribute
their p<sub>min</sub> too, so records above `--p-small` are not lost. Records are
defined from 4, so the flag requires the default `--start`.

Under `--gpus>1` segments complete out of order, so a later segment can raise the
running maximum and permanently suppress a genuine earlier record. The surviving
set is scheduling-dependent. **Use a single GPU when the record sequence is being
used for validation**; multi-GPU runs remain correct for verification itself.

Primality is decided by a bitset lookup wherever possible. The GPU phase needs
nothing else: every complement q it queries lies in one of its two prime
bitsets, and a q outside both would end the run as an internal error. The CPU
fallback (Phase 2) looks q up in a table below 10<sup>8</sup>; above that it
uses, by default, a 12-base deterministic Miller–Rabin, whose base set is
*proved* deterministic for all *n* < 2<sup>64</sup>. `--primetest=bpsw` selects
Baillie–PSW instead, in the strengthened form Baillie, Fiori and Wagstaff
recommend (*Math. Comp.* 90, 2021; arXiv:2006.14425, Section 6): a strong
base-2 test, Method A* parameters, the strong Lucas test, and two further
checks, V<sub>n+1</sub> ≡ 2Q and Euler's criterion for Q, both mod n.
Baillie–PSW has no such proof, but it is exact below
2<sup>64</sup> by computation: Baillie, Fiori and Wagstaff report that none of
the 118,968,378 base-2 pseudoprimes below 2<sup>64</sup> is a Lucas pseudoprime
for Method A*, the parameter choice used here, and in a one-off run against
that list (Feitsma and Galway) this implementation rejects every one, on
device and host. The Miller–Rabin implementation passed the same run, and
agreed with an independent sieve on every odd number below 2<sup>32</sup>.
`test_bpsw_spsp` and `test_mr_spsp` keep both tests honest below
2<sup>32</sup>, and `test_primality` cross-checks them above it.

---

## Scope and limits

A successful run means that every even number in the stated range was written
as a sum of two primes, each decided by a sieve or by a deterministic test. The
GPU phase uses sieved bitsets only. The CPU fallback (Phase 2) uses a table
below 10<sup>8</sup> and, above it, a Miller–Rabin test with the first twelve
prime bases, which is proved deterministic below 3.19·10<sup>23</sup> and so
for every 64-bit input. Baillie–PSW, selected with `--primetest=bpsw`, is not a
primality proof; below 2<sup>64</sup> it is exact only by computation (see
Correctness).

Phase 2 searches p ≤ 10<sup>8</sup> only. If it finds no partition for some n,
the run stops with the message "no partition with p ≤ 10^8 found for n = …"
and exit status 1. This is a search-limit outcome, not a counterexample claim.
Exit status 1 is also used for invalid input and internal errors, so the
message identifies the case. Such an n can be examined further with
`single_check`, whose search extends to every p ≤ n/2 with a deterministic test
for any n below 2<sup>64</sup>, or with `big_check --p-max=…`.

`single_check` stops at the first partition it finds and reports a
counterexample (exit status 2) only after exhausting every p ≤ n/2. The search
is exhaustive by construction for every 64-bit n, but a full search near
2<sup>64</sup> would take far longer than any practical run and has not been
performed. `big_check` accepts n of any size, but decides q with GMP's
`mpz_probab_prime_p`: for q of 2<sup>64</sup> or more the result is a probable
prime, and the output says so.

With more than one GPU, the sequence printed by `--record-check` depends on
segment scheduling by design; `--window-max` does not. The default CMake flags
include `-march=native`, which ties the host binaries to CPUs with the build
machine's instruction set. This limits portability, not correctness.

---

## Tuning

`TILE_ODDS` and `SPLIT_THRESHOLD` are compile-time constants, overridable with
`-DCMAKE_CUDA_FLAGS="-DSPLIT_THRESHOLD=65536"`. Both defaults were measured on an
RTX 5090 with the large-prime kernel present. Figures below are variable cost:
mean wall clock at N minus mean wall clock at 10<sup>6</sup>, which removes the
fixed setup. Five runs at 10<sup>11</sup> and 10<sup>12</sup>, three at
10<sup>13</sup>, builds interleaved, 0 Phase 2 fallbacks throughout.

### Tile width

`TILE_ODDS` is the number of odd numbers per sieve tile, and shared memory per
block is `TILE_ODDS` bytes. It trades occupancy against the fixed per-tile cost
of looping over every small prime.

| `TILE_ODDS` | 10<sup>11</sup> | 10<sup>12</sup> | 10<sup>13</sup> |
|---|---|---|---|
| 4096 | — | 10.760 s | — |
| 8192 | 0.726 s | 7.946 s | 83.969 s |
| **16384** | **0.702 s** | **7.398 s** | **78.670 s** |
| 32768 | 0.800 s | 8.418 s | 88.957 s |

16384 is fastest at every limit measured, by 13–14% over 32768, a margin stable
across two decades of N rather than a crossover. Occupancy does not predict this:
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives 6, 6, 5 and 3 resident
blocks per SM for 4096, 8192, 16384 and 32768 at 256 threads, with 36 registers
in every case, so shared memory alone sets it. Occupancy falls monotonically with
tile width while runtime is U-shaped.

### Split threshold

Primes below `SPLIT_THRESHOLD` go to the tiled kernel, the rest to
`large_prime_sieve_kernel`. Measured at 10<sup>13</sup> with `TILE_ODDS=16384`:

| `SPLIT_THRESHOLD` | Variable cost | Relative |
|---|---|---|
| 32768 | 85.991 s | +9.28% |
| **65536** | **78.686 s** | optimum |
| 131072 | 81.369 s | +3.41% |

65536 remains best after the tile width dropped to 16384, so the two constants
did not have to be retuned together.

The optimum for `SPLIT_THRESHOLD` depends on how many sieving primes there are,
which grows as √N above 10<sup>12</sup>, so a different limit may prefer a
different value. Only powers of two were sampled.

`--seg-size` mainly trades memory for parallelism and is flat near the default.

---

## Repository layout

```
src/goldbach.cu           Main verifier
include/sieve_kernel.cuh  Segment sieve (tiled and large-prime kernels)
include/phase1_kernel.cuh Verification kernels (transposed and scalar)
include/primality.cuh     Miller-Rabin and Baillie-PSW, device and host
include/segment_geometry.hpp  Segment bounds and sieved range, shared with test_phase1
include/parse_u64.hpp     Strict decimal parsing, shared by goldbach and single_check
src/test_*.c*             The test binaries described above
tests/test_*.sh           The shell-driven tests described above
tests/run_sanitizers.sh   compute-sanitizer runner (not part of ctest)
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

If you use this software, please cite the archived release on Zenodo:

```
Llorente-Saguer, I. (2026). GoldbachGPU [Software]. Zenodo.
https://doi.org/10.5281/zenodo.18786328
```

This DOI resolves to the latest version. The DOI of each release is listed
on the Zenodo record.

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

**Note on versions.** The preprints above describe release v2.0.0. 
Their performance figures describe that implementation and are not comparable 
to those on this page, which come from the current release. 
A manuscript describing the current release is in preparation.

### Versions

| Version | Released | Scope | DOI |
|---|---|---|---|
| v2.0.2 | 2026-09-18 | The v2.0.0 architecture of arXiv:2603.07850 with two concurrency defects in the sieve corrected and `--seg-size` bounded; a reference artifact for the corrected figures. | [10.5281/zenodo.22831383](https://doi.org/10.5281/zenodo.22831383) |
| v3.0.0 | 2026-09-09 | Byte-wide sieve marking, transposed Phase 1, large-prime kernel, bitset verification state, `--record-check`, and tests of the sieve, Phase 1, primality and bitset construction. | [10.5281/zenodo.22678175](https://doi.org/10.5281/zenodo.22678175) |
| v3.1.0 | 2026-09-26 | `TILE_ODDS` 16384; `big_check` reports the minimal p, with expression input and distinct exit statuses; tests registered with CTest. | [10.5281/zenodo.22980241](https://doi.org/10.5281/zenodo.22980241) |
| v3.2.0 | 2026-09-30 | Prime counting by default, `--window-max` and verification windows at any `--start`, folded prime count, faster p<sub>min</sub> tracking, Baillie–PSW as specified by Baillie, Fiori and Wagstaff, fail-closed segment accounting, stricter input validation, and an extended test suite. | assigned at release |

The concept DOI [10.5281/zenodo.18786328](https://doi.org/10.5281/zenodo.18786328)
resolves to the latest version; each release also has its own DOI, listed
above. Details of each version are in [CHANGELOG.md](CHANGELOG.md).

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
