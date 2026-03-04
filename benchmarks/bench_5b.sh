#!/usr/bin/env bash

LIMIT="$1"
if [ -z "$LIMIT" ]; then
    echo "Usage: $0 LIMIT"
    exit 1
fi

BIN="../build/bin/goldbach_gpu5b"
OUT="bench_5b_results.csv"

echo "async,batch,copy,time_ms" > "$OUT"

# ASYNC=("off" "on")
ASYNC=("off")
BATCH=(100000)
# COPY=("full" "failures")
COPY=("full")
STREAMS=(1 2 4) # Test with more streams for better concurrency

for a in "${ASYNC[@]}"; do
    for b in "${BATCH[@]}"; do
        for c in "${COPY[@]}"; do
            for s in "${STREAMS[@]}"; do

                CMD="$BIN $LIMIT --batch-size=$b --copy-mode=$c --streams=$s"
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
done

echo "Done. Results saved to $OUT"