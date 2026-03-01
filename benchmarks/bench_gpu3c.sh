#!/bin/bash

BIN="../build/bin/goldbach_gpu3c"
OUT="bench_gpu3c_results.csv"

LIMITS=(100000000000)
SEG_SIZES=(10000000 100000000)
P_SMALL=1000000

# VERIFIED=("byte" "bitset")
# SMALL=("on" "off")
VERIFIED=("byte")
SMALL=("on")

echo "LIMIT,SEG_SIZE,P_SMALL,verified,small,total_ms,phase2,vram_MB" > "$OUT"

for L in "${LIMITS[@]}"; do
  for S in "${SEG_SIZES[@]}"; do
    for V in "${VERIFIED[@]}"; do
      for SM in "${SMALL[@]}"; do

        echo "Running: LIMIT=$L SEG_SIZE=$S verified=$V small=$SM"

        CMD="$BIN $L $S $P_SMALL --verified=$V"
        if [ "$SM" = "off" ]; then
          CMD="$CMD --no-small"
        fi

        RESULT=$($CMD 2>&1)

        total_ms=$(echo "$RESULT" | grep "Total time" | grep -oE '[0-9]+\.[0-9]+')
        phase2=$(echo "$RESULT" | grep "Phase 2 fallbacks" | grep -oE '[0-9]+$')
        vram=$(echo "$RESULT" | grep "Verified buffer" | grep -oE '[0-9]+' | head -n1)

        echo "$L,$S,$P_SMALL,$V,$SM,$total_ms,$phase2,$vram" >> "$OUT"

      done
    done
  done
done

echo "Benchmark complete → $OUT"