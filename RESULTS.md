# Goldbach Verification Results

Sections 2 and 3 give the v3.2.0 and v3.1.0 range-verification results. Results
for other earlier versions and tools are in the corresponding tagged releases.

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
| RAM | 30.1 GiB for the v3.1.0 measurements; 47 GiB visible to WSL2 for the v3.2.0 session |

Driver, CUDA toolkit and WSL2 kernel are the same for the v3.1.0 and v3.2.0
sessions.

Everything below was measured on this machine, except section 4, which was
measured on a separate four-GPU node described there.

Unless a run states otherwise, the runs use
`--seg-size=200000000 --p-small=1000000 --batch-size=2000000`, built with
`-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120`, and report 0
Phase 2 fallbacks. Wall clock is from `/usr/bin/time -f %e` and is the primary
figure; computation time is the program's own report and excludes process start
and the prime-table build.

**Measurement method.** Builds that are compared are run interleaved, with the
order alternating from round to round, so that a change in machine state affects
both. Standard deviations are sample standard deviations (n − 1). Variable cost
is the mean wall clock at N minus the mean wall clock at 10<sup>6</sup> of the
same build in the same session, which removes the fixed startup. Absolute wall
times on this machine moved by about 2% between sessions, with both builds of a
comparison moving together and no GPU throttle reason active in the sampled
sessions; the cause was not established. Only interleaved comparisons within one
session are therefore quoted as ratios.

**GPU state.** During the 10<sup>15</sup> window runs, `nvidia-smi` was sampled
every 60 s (359 samples, 358 of them under load): median SM clock 2,760 MHz
(2,745 to 2,760 under load), median board power 401 W (limit 575 W), maximum
temperature 73 °C, and no clock throttle reason active in any sample. During the
fixed 10<sup>12</sup> benchmark of the timing check the SM clock was 2,775 to
2,835 MHz, power 290 to 310 W and temperature 55 to 61 °C, again with no
throttle reason.

---

## 2. Range verification at v3.2.0

The logs, scripts and summary tables behind this section are archived as a
separate dataset at DOI [10.5281/zenodo.23090649](https://doi.org/10.5281/zenodo.23090649).

### 2.1 Comparison with v2.0.2

The v2.0.2 tag (a corrected release of the v2.0.0 architecture) and the v3.2.0
tree were built from source and run interleaved, with alternating order, at the
flags of section 1. v3.2.0 ran with prime counting on, its default. Five runs
each at 10<sup>6</sup>, 10<sup>11</sup> and 10<sup>12</sup>, three at
10<sup>13</sup>; all 36 runs succeeded with 0 Phase 2 fallbacks.

| N | Build | Runs | Mean wall clock | sd | Variable cost | Speedup, variable cost | Speedup, wall clock |
|---|---|---|---|---|---|---|---|
| 10<sup>6</sup> | v2.0.2 | 5 | 0.488 s | 0.013 s | | | |
| 10<sup>6</sup> | v3.2.0 | 5 | 0.480 s | 0.014 s | | | |
| 10<sup>11</sup> | v2.0.2 | 5 | 11.348 s | 0.018 s | 10.860 s | | |
| 10<sup>11</sup> | v3.2.0 | 5 | 1.160 s | 0.007 s | 0.680 s | 15.97× | 9.78× |
| 10<sup>12</sup> | v2.0.2 | 5 | 143.890 s | 0.662 s | 143.402 s | | |
| 10<sup>12</sup> | v3.2.0 | 5 | 7.978 s | 0.052 s | 7.498 s | 19.13× | 18.04× |
| 10<sup>13</sup> | v2.0.2 | 3 | 3,204.710 s | 4.523 s | 3,204.222 s | | |
| 10<sup>13</sup> | v3.2.0 | 3 | 79.523 s | 0.067 s | 79.043 s | 40.54× | 40.30× |

The mean wall clock grows by a factor of 12.68 from 10<sup>11</sup> to 10<sup>12</sup>
and 22.27 from 10<sup>12</sup> to 10<sup>13</sup> for v2.0.2, and by 6.88 and 9.97
for v3.2.0. The speedup therefore widens with N over this range. The v3.2.0 arm
at 10<sup>12</sup> sat in the slower of the two session bands described in
section 1 (7.978 s here; 7.82 to 7.87 s in other sessions), which affects its
speedup by under 2%.

### 2.2 Certified run, 4 to 10<sup>14</sup>

```
goldbach 100000000000000 --seg-size=200000000 --p-small=1000000 \
         --batch-size=2000000 --record-check
```

One GPU, prime counting on (default). Wall clock 857 s (14 min 17 s), computation
856.84 s, 0 Phase 2 fallbacks, maximum resident set size 182,400 kB (178 MiB).
The run reports the program's own count π(10<sup>14</sup>) = 3,204,941,750,802,
equal to the value from primecount 8.8. `small_prime_count` is 664,579.

`--record-check` printed 22 records (at most one per segment). The 16 below
10<sup>13</sup> form a subsequence of the published table of Oliveira e Silva
et al.; the six above 10<sup>13</sup> lie beyond the range of the CPU-side
`test_records` and were not compared with a table:

| n | p<sub>min</sub> |
|---|---|
| 335,070,838 | 1,427 |
| 721,013,438 | 1,789 |
| 1,847,133,842 | 1,861 |
| 7,473,202,036 | 1,877 |
| 11,001,080,372 | 1,879 |
| 12,703,943,222 | 2,029 |
| 21,248,558,888 | 2,089 |
| 35,884,080,836 | 2,803 |
| 105,963,812,462 | 3,061 |
| 244,885,595,672 | 3,163 |
| 599,533,546,358 | 3,457 |
| 3,132,059,294,006 | 3,463 |
| 3,620,821,173,302 | 3,529 |
| 4,438,327,672,994 | 3,613 |
| 5,320,503,815,888 | 3,769 |
| 8,342,945,544,436 | 3,917 |
| 10,591,605,900,482 | 4,003 |
| 12,982,270,197,518 | 4,027 |
| 15,197,900,994,218 | 4,057 |
| 28,998,050,650,046 | 4,327 |
| 46,878,442,766,282 | 4,519 |
| 76,903,574,497,118 | 4,909 |

The window maximum over [4, 10<sup>14</sup>] is 4,909 at n = 76,903,574,497,118.
The flag reports at most one record per segment, so its output is a subsequence
of the true records, not all of them. It is a cross-check of minimal primes and
not a completeness claim.

### 2.3 Windows at 4·10<sup>18</sup>

`--start=4000000000000000000 --window-max`, N = 4·10<sup>18</sup> + W, prime
counting on. Run A: `--seg-size=200000000 --p-small=1000000 --batch-size=2000000`.
Run B: `--seg-size=300000000 --p-small=2000000 --batch-size=500000`. In every
window runs A and B agree exactly on the count, the window maximum and the
smallest n attaining it, and report 0 Phase 2 fallbacks.

| W | Numbers checked | Primes in (4·10<sup>18</sup>, N], equal to the primecount difference | Window maximum p<sub>min</sub> at n | Wall clock A / B | Host memory |
|---|---|---|---|---|---|
| 10<sup>12</sup> | 500,000,000,001 | 23,346,564,662 | 6,073 at 4,000,000,402,622,391,632 | 13.54 s / 12.64 s | 1,041 MiB |
| 10<sup>13</sup> | 5,000,000,000,001 | 233,465,239,142 | 6,421 at 4,000,005,756,560,737,472 | 113.05 s / 106.39 s | 1,041 MiB |
| 10<sup>14</sup> | 50,000,000,000,001 | 2,334,657,870,388 | 7,103 at 4,000,048,374,322,888,936 | 1,110.87 s / 1,044.38 s | 1,041 MiB |
| 10<sup>15</sup> | 500,000,000,000,001 | 23,346,512,560,823 | 7,487 at 4,000,400,837,678,526,154 | 11,125.3 s / 10,404.2 s | 1,137 / 1,091 MiB |

primecount 8.8 (16 threads, about 6.3 s per value):

| x | π(x) |
|---|---|
| 4·10<sup>18</sup> | 95,676,260,903,887,607 |
| 4·10<sup>18</sup> + 10<sup>12</sup> | 95,676,284,250,452,269 |
| 4·10<sup>18</sup> + 10<sup>13</sup> | 95,676,494,369,126,749 |
| 4·10<sup>18</sup> + 10<sup>14</sup> | 95,678,595,561,757,995 |
| 4·10<sup>18</sup> + 10<sup>15</sup> | 95,699,607,416,448,430 |

For the 10<sup>15</sup> window, run A took 11,125.3 s wall (11,122.6 s
computation, 2.31 s to build the prime table) and run B 10,404.2 s wall
(10,401.9 s computation, 2.01 s). The wall clock of successive windows grows by
factors of 8.35, 9.83 and 10.01 (run A). At 4·10<sup>18</sup> the sieve bound is about 2·10<sup>9</sup> and there are
98,234,059 sieving primes (`small_prime_count`), against 664,579 in the run of
section 2.2. The cost per number is higher than in that run: 1,110.87 s (run A)
for the 5.0·10<sup>13</sup> numbers of the 10<sup>14</sup> window, against 857 s
for the 5.0·10<sup>13</sup> numbers of section 2.2. The two runs differ in start,
in `--window-max` against `--record-check`, and in sieve bound; the comparison
is not a controlled one. The window
maximum of each window is at least that of the smaller windows it contains; that
of the 10<sup>15</sup> window lies beyond its 10<sup>14</sup> sub-window.

The windows were run from a build of the frozen v3.2.0 tree. No range below
4·10<sup>18</sup> other than [4, 10<sup>14</sup>] is claimed to be covered by
these runs.

### 2.4 Interval semantics

`--start=S` and N are even. The run checks every even n in the closed interval
[S, N], that is N/2 − S/2 + 1 numbers. The prime count printed with a START above
the small-prime bound (about √N) is over the primes q with S < q ≤ N, so the
count of a window is π(N) − π(S). `--window-max` reports the largest p<sub>min</sub>
over the numbers checked and the smallest n attaining it. It is a window maximum
and not a p-record.

---

## 3. Range verification at v3.1.0

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
n = 76,903,574,497,118. The emitted set is a subsequence of the 54 published
records below 10<sup>14</sup>: the mechanism reports at most one record per
segment, so a record sharing a segment with a larger one is masked. The run was
made on one GPU, from 4, with `--p-small` = 10<sup>6</sup>, far above the largest
p_min, so no number reached the CPU fallback. It is an external cross-check of
minimal primes, not a completeness claim. The same check at v3.2.0 is in
section 2.2. From v3.2.0 the fallback reports p_min, and `--record-check` is
rejected with any `--start` other than 4.

---

## 4. Multi-GPU

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

## 5. Single large numbers (`big_check`)

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
proven prime; q is reported as "prime" only then and only if q < 2^64. Otherwise
it is reported as "probable prime (BPSW)", and the verdict reads "Goldbach holds
for this n if q is prime".

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

Every q above was confirmed as a probable prime with PARI/GP 2.17.3
`ispseudoprime`, an independent Baillie-PSW implementation. That is a second
probable-prime test, not a proof of primality.

Most candidates are rejected by trial division inside `mpz_probab_prime_p` at
small cost; the few whose complement reaches BPSW account for nearly all of the
running time.

---

## 6. Roadmap

Done:
- Segmented GPU range verifier: a double sieve with a CPU fallback
- GPU sieve construction with compact prime bitsets
- Tiled shared-memory sieve with a separate large-prime kernel
- Transposed Phase 1 verification
- Multi-GPU scheduling, at 99.95% parallel efficiency on four GPUs at 10^13
- Range verification to 10^14 on a single RTX 5090
- Verification windows of up to 10^15 numbers at 4·10^18, with prime counts checked against primecount
- A record check against the published p-records of Oliveira e Silva et al.
- An arbitrary-precision single-number checker (minimal p, up to 10,001 digits)
- A test suite under ctest. Two injected faults, a count kernel that reports
  nothing unverified and a strong Lucas step that always passes, each fail it

Planned:
- Range verification to 10^15 on multiple GPUs
- Goldbach partition counting c(n) at scale
