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
