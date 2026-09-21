# Benchmark logs

Raw measurement logs supporting the corrigendum to arXiv:2603.07850. All
produced from this tag (v2.0.1) unless noted.

- `bench-logs-v201/` corrected single-GPU ladder, 10^10 to 10^13.
- `bench-logs-table1/` Table 1 rebuild: v2.0.1 and the v1 legacy binary
  measured on one machine at 10^9 and 10^10.
- `bench-logs-corrigendum/` quantification of both defects (Items A and B),
  Nsight Systems profiles at 2x10^12 and 10^13 (Item C), and the matched cost
  of the sieve correction (Item D).

Machine: RTX 5090, CUDA 13.3, Ubuntu 26.04 under WSL2. The multi-GPU figures in
the corrigendum came from a rented 2x RTX 5090 node with a different driver and
CPU; that environment is stated in the corrigendum.

`bench-logs-corrigendum/itemA_source/` holds a deliberately unfixed extraction
of the v2.0.0 sieve kernel, used only to quantify the defect. It is not part of
this release.

Omitted for size: the Nsight profile for 10^13, whose `--stats` text output is
included and contains every figure quoted, and the SQLite exports, which are
regenerable with `nsys export --type sqlite`. The 2x10^12 profile is retained
because the corrected memset breakdown was derived from its SQLite export and
cannot be re-derived from the text output alone.

## Provenance notes

Every figure in this directory comes from a build configured with
`-DCMAKE_CUDA_ARCHITECTURES=120`, confirmed by the `sm_120` cubin names in
`bench-logs-table1/build.log` and recorded in each `summary.md`. A default
configure on this branch resolves to sm_75 and produces different timings.

`bench-logs-v201/cmake.log` records a build directory `/tmp/v211`. That worktree
was created for earlier work and later checked out to `release/v2.0.1`; the
directory name was not changed. The binary is v2.0.x, not v2.1.0: every ladder
log prints "All even numbers up to <LIMIT> satisfy Goldbach", whereas v2.1.0
prints "All even numbers from <START> up to <LIMIT>". The file is kept as
captured rather than regenerated.

`bench-logs-v201/environment.txt` is the environment referred to by
`bench-logs-v201/summary.md`. It was absent from the original commit because
`.gitignore` excluded `*.txt`; that rule now carries an exception for this
directory.

`bench-logs-v201/setup_cost.log` bounds the setup carried inside the program's
own timer: CUDA context creation and device allocation happen after the timer
starts, so every reported computation time includes them.

`bench-logs-corrigendum/itemE_d3_segsize.log` records the behaviour of v2.0.1 at
a segment size above 2^32, where 32-bit thread-index arithmetic wraps. Captured
before the startup guard was added in v2.0.2.

`bench-logs-corrigendum/itemF_arch_differential.log` compares an sm_120 build
against a default (sm_75, JIT) build of the same tree. Phase 2 fallback counts
are identical at all three test points, so the two sieves agree on every value
they were asked about and the timing difference between the builds is code
generation, not correctness. `bench-logs-corrigendum/itemF_arch_1e10.log` adds
the 10^10 comparison that the first log refers to but does not contain: five
runs per build, fallbacks identical, the sm_75 build faster by 1.88x.

`bench-logs-corrigendum/itemG_*` runs Item A's differential harness,
unchanged, against the corrected v2.0.2 kernel, with a control run against the
defective kernel in the same build. These were added after the v2.0.2 tag, on
the `release/v2.0.2` branch.

`bench-logs-machineB/` holds the two-GPU figures. That node was a rented
ephemeral pod and the session was not captured to file, so the values were
transcribed from the terminal afterwards. They are weaker evidence than
everything else here, which is machine-captured, and the corrigendum says so.
The node had four GPUs; only the 1- and 2-GPU configurations were exercised,
because two earlier pods failed with a container GPU passthrough fault and a
third four-GPU node was not available in time.
