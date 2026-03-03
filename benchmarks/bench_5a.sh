#!/usr/bin/env bash

LIMIT="$1"
if [ -z "$LIMIT" ]; then
    echo "Usage: $0 LIMIT"
    exit 1
fi

BIN="../build/bin/goldbach_gpu5a"
OUT="bench_5a_results.csv"

echo "async,batch,copy,time_ms" > "$OUT"

# ASYNC=("off" "on")
ASYNC=("off")
BATCH=(25000 50000 100000 200000)
COPY=("full" "failures")

for a in "${ASYNC[@]}"; do
    for b in "${BATCH[@]}"; do
        for c in "${COPY[@]}"; do

            CMD="$BIN $LIMIT --batch-size=$b --copy-mode=$c"
            if [ "$a" = "on" ]; then
                CMD="$CMD --async"
            fi

            echo "Running: $CMD"

            LINE=$($CMD 2>/dev/null | grep -E "Total time[[:space:]]*:")
            TIME=$(echo "$LINE" | awk '{print $4}')

            if [ -z "$TIME" ]; then
                TIME="FAIL"
            fi

            echo "$a,$b,$c,$TIME" >> "$OUT"
        done
    done
done

echo "Done. Results saved to $OUT"