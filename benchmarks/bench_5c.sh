#!/usr/bin/env bash

BIN=../build/bin/goldbach_gpu5c
LIMIT=1000000000   # 1e9 for quick tests
OUT=results_$(date +%Y%m%d_%H%M%S).txt

SEG_SIZES=(10000000 20000000 50000000)
P_SMALLS=(200000 500000 1000000)
STREAMS=(1 2)
# COPY_MODES=("full" "failures" "none")
COPY_MODES=("full")

echo "timestamp,seg_size,p_small,streams,copy_mode,total_ms,sieve_ms,kernel_ms,copy_ms,phase2_ms" >> "$OUT"

for seg in "${SEG_SIZES[@]}"; do
  for ps in "${P_SMALLS[@]}"; do
    for st in "${STREAMS[@]}"; do
      for cm in "${COPY_MODES[@]}"; do

        echo "Running SEG_SIZE=$seg P_SMALL=$ps streams=$st copy=$cm"

        # Run and capture output
        RAW=$($BIN $LIMIT --seg-size=$seg --p-small=$ps --streams=$st --copy-mode=$cm 2>&1)

        # Extract timing numbers using grep + sed
        total=$(echo "$RAW" | grep "Total time" | sed 's/.*: //; s/ ms//')
        sieve=$(echo "$RAW" | grep "GPU sieve total" | sed 's/.*: *//; s/ ms//')
        kernel=$(echo "$RAW" | grep "GPU kernel total" | sed 's/.*: *//; s/ ms//')
        copy=$(echo "$RAW" | grep "Copy verified" | sed 's/.*: *//; s/ ms//')
        phase2=$(echo "$RAW" | grep "CPU Phase 2 total" | sed 's/.*: *//; s/ ms//')

        # Append to log
        echo "$(date +%Y-%m-%dT%H:%M:%S),$seg,$ps,$st,$cm,$total,$sieve,$kernel,$copy,$phase2" >> "$OUT"

      done
    done
  done
done

echo "Results saved to $OUT"