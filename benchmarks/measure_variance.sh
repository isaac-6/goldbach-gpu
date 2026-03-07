#!/bin/bash

LIMIT=1000000000
GPUS=1
RUNS=20
OUTFILE="variance_results.csv"

echo "Run,Time_Seconds" > $OUTFILE

# Warm-up run (GPU context creation, JIT, etc.)
../build/bin/goldbach --gpus=$GPUS $LIMIT > /dev/null

for i in $(seq 1 $RUNS); do
    output=$(../build/bin/goldbach --gpus=$GPUS $LIMIT  --seg-size=200000000 --p-small=1000000 --batch-size=2000000)
    time_sec=$(echo "$output" | grep "Total computation time" | awk '{print $5}')
    echo "$i,$time_sec" >> $OUTFILE
done

awk -F, '
    NR>1 { sum+=$2; sumsq+=($2*$2); n++ }
    END {
        mean = sum/n;
        stddev = sqrt((sumsq - (sum*sum/n)) / (n-1));
        cv = (stddev/mean)*100;
        printf "Mean   : %.4f seconds\n", mean;
        printf "StdDev : %.4f seconds\n", stddev;
        printf "CV (%%) : %.4f %%\n", cv;
    }
' $OUTFILE