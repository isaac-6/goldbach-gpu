#!/bin/bash

BIN=../build/bin/goldbach_gpu3
LOG=bench_gpu3.log

echo "=== GPU3 Benchmark: $(date) ===" | tee -a "$LOG"

# LIMITS=(1000000000 5000000000 10000000000 50000000000)
# SEGS=(100000000 500000000)
# PSMALLS=(1000000 2000000 5000000)

LIMITS=(1000000000 5000000000 10000000000)
SEGS=(100000 1000000 10000000)
PSMALLS=(1000000)

for L in "${LIMITS[@]}"; do
  for S in "${SEGS[@]}"; do
    for P in "${PSMALLS[@]}"; do
      echo -e "\n=== LIMIT=$L SEG_SIZE=$S P_SMALL=$P ===" | tee -a "$LOG"
      $BIN $L $S $P 2>&1 | tee -a "$LOG"
    done
  done
done

echo -e "\n=== Benchmark complete ===" | tee -a "$LOG"