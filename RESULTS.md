# Goldbach Verification Results

Results for earlier versions and tools are in the corresponding tagged releases.

---

## 1. Platform

| | |
|---|---|
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| GPU | NVIDIA GeForce RTX 5090, 32,606 MiB, compute capability 12.0 (`sm_120`) |
| Driver | 610.88 |
| CUDA | 13.3 (nvcc V13.3.73) |
| OS | Ubuntu 26.04 LTS under WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |
| GCC | 15.2.0 |
| GMP | 6.3.0 |
| RAM | 30.1 GiB |

Everything below was measured on this machine, except section 3, which was
measured on a separate four-GPU node described there.

All runs use `--seg-size=200000000 --p-small=1000000 --batch-size=2000000`,
built with `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120`, and
report 0 Phase 2 fallbacks. Wall clock is from `/usr/bin/time -f %e`.

---

## 2. Range verification

Every even number from 4 to N is checked. Variable cost is mean wall clock at N
minus mean wall clock at 10<sup>6</sup>, which removes the fixed startup; the
10<sup>6</sup> baseline is 0.4613 s, pooled over 15 runs. Picoseconds per even
number are computed from variable cost, so they measure verification throughput
rather than total runtime.

The 10<sup>11</sup>, 10<sup>12</sup> and 10<sup>13</sup> rows pool every run at
`TILE_ODDS=16384` and `SPLIT_THRESHOLD=65536` across the tuning sweeps, which is
the shipped configuration: 5 runs at 10<sup>11</sup>, 5 at 10<sup>12</sup> and 6
at 10<sup>13</sup> (three from the tile-width sweep and three from the
split-threshold sweep). Pooling is sound because the device code is identical:
`cuobjdump -sass` output for the v3.1.0 build and for a v3.0.0 build forced to
`TILE_ODDS=16384` matches byte for byte.

| N | Even numbers | Wall clock | Runs | Variable cost | ps / even number |
|---|---|---|---|---|---|
| 10<sup>10</sup> | 4,999,999,999 | 0.540 ± 0.017 s | 5 | 0.079 s | 15.7 |
| 10<sup>11</sup> | 49,999,999,999 | 1.162 ± 0.028 s | 5 | 0.701 s | 14.0 |
| 10<sup>12</sup> | 499,999,999,999 | 7.858 ± 0.075 s | 5 | 7.397 s | 14.8 |
| 10<sup>13</sup> | 4,999,999,999,999 | 79.140 ± 0.032 s | 6 | 78.679 s | 15.7 |
| 10<sup>14</sup> | 49,999,999,999,999 | 836.13 s | 1 | 835.67 s | 16.7 |

The 10<sup>10</sup> row is dominated by startup: 0.079 s of verification against
0.540 s of wall clock, so it is not a throughput figure. Throughput is flat to
within 20% across four decades, drifting up from 14.0 to 16.7 ps per even number
as the sieve bound √N overtakes `--p-small` and the number of sieving primes
grows.

The 10<sup>14</sup> run took 836.13 s wall, 835.66 s computation, 0 Phase 2
fallbacks, in a single run without `--record-check`. Device memory required was
reported as 132 MB against 30,927 MB free; the program logs its pre-flight
estimate rather than measured peak usage.

At 10<sup>14</sup> this is 1.13× faster than v3.0.0, whose tag run took 942.3 s
wall against 836.13 s here. The gain is the `TILE_ODDS` default moving from 32768
to 16384; the device code is otherwise unchanged.

**Record check.** A separate 10<sup>14</sup> run with `--record-check`, measured
at v3.0.0, emitted 22 minimal-prime records, the largest being p_min = 4909 at
n = 76,903,574,497,118. All 22 match the published p-records of Oliveira e Silva
et al., which are verified below 4·10<sup>18</sup>. The emitted set is a
subsequence of the 54 published records below 10<sup>14</sup>, not all of them:
the mechanism reports at most one record per segment, so a record sharing a
segment with a larger one is masked. Every emitted record is nonetheless genuine,
which is what makes the check meaningful — it is an external cross-check of
minimal primes, not a completeness claim.

---

## 3. Multi-GPU

Measured on a separate node: 4× RTX 5090, AMD EPYC 7302P, CUDA 12.8, driver
580.126.09, commit 3afcf07, `TILE_ODDS` 32768. Five runs per configuration.
Efficiency is η<sub>k</sub> = T₁ / (k · T<sub>k</sub>) on computation time, with
T₁ measured on the same node.

| N | GPUs | Computation | Speedup | Efficiency |
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

Work is distributed by a lock-free atomic counter, each GPU claiming the next
segment when it finishes the previous one, so devices of different speeds balance
automatically and no GPU waits on another.

---

## 4. Single large numbers (`big_check`)

Tool: `src/big_check.cpp`

Finds the **smallest** prime p <= L with n - p a probable prime, where
L = min(`--p-max`, floor(n/2)). The result does not depend on thread count or
scheduling: batches of candidates are processed in ascending order, threads
narrow a shared best index downwards and skip only indices above it, and the
search stops after the first batch containing a hit. No index below the winner
is ever skipped, so the reported p is minimal for the given bound.

Primality of q is decided by GMP 6.3 `mpz_probab_prime_p(q, 25)`, which performs
trial division, then a Baillie-PSW probable-prime test, then `reps - 24`
Miller-Rabin rounds -- one round beyond BPSW at this setting. A return of 2 means
proven prime and is reported as "prime"; 1 means probable prime and is reported
as "probable prime (BPSW)".

Three exit statuses:

| Exit | Meaning |
|---|---|
| 0 | a partition was found; p and q are printed |
| 3 | no partition with p <= L. A search-limit result, not a counterexample; raising `--p-max` continues the search |
| 1 | invalid input |

n may be given as a decimal string or as `a^b`, `a^b+c` or `a^b-c`, evaluated
with GMP, so the numbers below are reproducible directly from their command
lines. Evenness and n >= 4 are checked on the parsed value.

Every run ends with one machine-readable line:

```
RESULT status=<found|bound> p=<p> q_digits=<d> candidates=<i> prp_calls=<k> threads=<t> time_s=<s>
```

`candidates` is the index of p plus one, and `prp_calls` is the number of
`mpz_probab_prime_p` calls. `prp_calls` depends on scheduling, because threads
stop issuing tests once a hit lands at a lower index; it is reproducible only at
a fixed thread count. `p`, `q_digits` and `candidates` do not.

Practical limit: each GMP primality test scales steeply in digit count.

| n | q digits | p found | candidates | Threads | Time |
|---|---|---|---|---|---|
| 10^100 | 100 | 797 | 139 | 16 | 0.027 ± 0.004 s |
| 10^1000 | 1000 | 26,981 | 2,959 | 16 | 0.246 ± 0.002 s |
| 10^5000 | 5000 | 4,967 | 664 | 16 | 4.914 ± 0.007 s |
| 10^10000 | 10000 | 47,717 | 4,920 | 16 | 96.071 ± 0.129 s |
| 2^1000 | 302 | 16,607 | 1,921 | 16 | 0.029 ± 0.003 s |
| 2^10000 | 3011 | 227 | 49 | 16 | 1.360 ± 0.004 s |

AMD Ryzen 7 9800X3D, 16 threads. Mean ± sample standard deviation over three runs.

Every q above was confirmed prime with PARI/GP 2.17.3 `ispseudoprime`, an
independent Baillie-PSW implementation.

Most candidates are rejected by trial division inside `mpz_probab_prime_p` at
small cost; the few whose complement reaches BPSW account for nearly all of the
running time.

---

## 5. Roadmap

Done:
- Segmented GPU range verifier: a double sieve with a CPU fallback
- GPU sieve construction with compact prime bitsets
- Tiled shared-memory sieve with a separate large-prime kernel
- Transposed Phase 1 verification
- Multi-GPU scheduling, at 99.95% parallel efficiency on four GPUs at 10^13
- Range verification to 10^14 on a single RTX 5090
- A record check against the published p-records of Oliveira e Silva et al.
- An arbitrary-precision single-number checker (minimal p, up to 10,001 digits)
- A test suite under ctest, checked by fault injection

Planned:
- Range verification to 10^15 on multiple GPUs
- Goldbach partition counting c(n) at scale
