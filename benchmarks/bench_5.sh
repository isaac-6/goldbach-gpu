#!/usr/bin/env bash
# Goldbach GPU Finetuning Benchmark Script
# For paper "Fine-tuning" section
# Tests all meaningful combinations of SEG_SIZE, P_SMALL, BATCH_SIZE across multiple LIMITs

BIN="../build/bin/goldbach_gpu5a"
OUT="finetune_results_$(date +%Y%m%d_%H%M%S).csv"

# ============== CONFIGURATION ==============
LIMITS=(1000000000 10000000000 100000000000)        # 1e9, 1e10, 1e11
SEG_SIZES=(20000000 50000000 100000000)             # 20M, 50M, 100M
P_SMALLS=(500000 1000000 5000000)
BATCH_SIZES=(500000 1000000 2000000)

echo "timestamp,limit,seg_size,p_small,batch_size,total_ms,sieve_ms,kernel_ms,copy_ms,phase2_ms,small_gpu_kb,seg_buf_mb,ver_buf_mb" > "$OUT"

echo "Starting fine-tuning benchmark. This will take a while..."
echo "Results will be saved to $OUT"

for limit in "${LIMITS[@]}"; do
  for seg in "${SEG_SIZES[@]}"; do
    for ps in "${P_SMALLS[@]}"; do
      for batch in "${BATCH_SIZES[@]}"; do

        echo "Running: LIMIT=$limit  SEG=$seg  P_SMALL=$ps  BATCH=$batch"

        RAW=$($BIN $limit --seg-size=$seg --p-small=$ps --batch-size=$batch 2>&1)

        # Extract values robustly
        total=$(echo "$RAW"  | sed -n 's/.*Total time\s*:\s*\([0-9.]*\).*/\1/p')
        sieve=$(echo "$RAW"  | sed -n 's/.*GPU sieve total:\s*\([0-9.]*\).*/\1/p')
        kernel=$(echo "$RAW" | sed -n 's/.*GPU kernel total:\s*\([0-9.]*\).*/\1/p')
        copy=$(echo "$RAW"   | sed -n 's/.*Copy verified → CPU total:\s*\([0-9.]*\).*/\1/p')
        phase2=$(echo "$RAW" | sed -n 's/.*CPU Phase 2 total:\s*\([0-9.]*\).*/\1/p')

        small_kb=$(echo "$RAW" | grep -oE 'Small primes in GPU \([0-9]+' | grep -oE '[0-9]+')
        seg_mb=$(echo "$RAW"   | grep -oE 'Segment buffer: [0-9]+' | grep -oE '[0-9]+')
        ver_mb=$(echo "$RAW"   | grep -oE 'Verified buffer: [0-9]+' | grep -oE '[0-9]+')

        # Fallback to 0 if parsing fails
        total=${total:-0}
        sieve=${sieve:-0}
        kernel=${kernel:-0}
        copy=${copy:-0}
        phase2=${phase2:-0}
        small_kb=${small_kb:-0}
        seg_mb=${seg_mb:-0}
        ver_mb=${ver_mb:-0}

        timestamp=$(date +%Y-%m-%dT%H:%M:%S)

        echo "$timestamp,$limit,$seg,$ps,$batch,$total,$sieve,$kernel,$copy,$phase2,$small_kb,$seg_mb,$ver_mb" >> "$OUT"

      done
    done
  done
done

echo "Benchmark completed!"
echo "Results saved to: $OUT"
echo "Total combinations tested: $(( ${#LIMITS[@]} * ${#SEG_SIZES[@]} * ${#P_SMALLS[@]} * ${#BATCH_SIZES[@]} ))"