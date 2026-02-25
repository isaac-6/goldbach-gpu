#!/bin/bash

# ---------------------------------------------------------
# Goldbach GPU Benchmark Suite
# ---------------------------------------------------------
# Output is appended to benchmarks.log
# ---------------------------------------------------------

LOGFILE="benchmarks.log"
echo "=== Benchmark run: $(date) ===" | tee -a "$LOGFILE"

BIN=./build/bin

# ---------------------------------------------------------
# CPU baseline: 10^6 → 10^9
# ---------------------------------------------------------
CPU_LIMITS=(1000000 10000000 100000000 1000000000)

echo -e "\n--- CPU Baseline ---" | tee -a "$LOGFILE"
for N in "${CPU_LIMITS[@]}"; do
    echo -e "\n[CPU] N=$N" | tee -a "$LOGFILE"
    $BIN/cpu_goldbach $N 2>&1 | tee -a "$LOGFILE"
done

# ---------------------------------------------------------
# GPU2 (global bitset): 10^8 → 10^10
# ---------------------------------------------------------
GPU2_LIMITS=(100000000 1000000000 10000000000)

echo -e "\n--- GPU2 (global bitset) ---" | tee -a "$LOGFILE"
for N in "${GPU2_LIMITS[@]}"; do
    echo -e "\n[GPU2] N=$N" | tee -a "$LOGFILE"
    $BIN/goldbach_gpu2 $N 2>&1 | tee -a "$LOGFILE"
done

# ---------------------------------------------------------
# GPU3 (segmented): 10^8 → 10^12
# ---------------------------------------------------------
# GPU3_LIMITS=(100000000 1000000000 10000000000 1000000000000)
GPU3_LIMITS=(100000000 1000000000 10000000000)

echo -e "\n--- GPU3 (segmented) ---" | tee -a "$LOGFILE"
for N in "${GPU3_LIMITS[@]}"; do
    echo -e "\n[GPU3] N=$N" | tee -a "$LOGFILE"
    $BIN/goldbach_gpu3 $N 2>&1 | tee -a "$LOGFILE"
done

echo -e "\n=== Benchmark complete ===\n" | tee -a "$LOGFILE"