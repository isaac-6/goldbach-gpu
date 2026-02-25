#!/bin/bash

LOGFILE="benchmarks.log"
OUTFILE="benchmarks/plot_data.tex"

echo "% Auto-generated PGFPlots data" > "$OUTFILE"
echo "% Generated on $(date)" >> "$OUTFILE"
echo "" >> "$OUTFILE"

# Helper: extract pairs (N, time_ms) for a given tag
extract_points() {
    local tag="$1"
    local label="$2"
    local style="$3"

    echo "\\addplot[$style] coordinates {" >> "$OUTFILE"

    # Find lines like:
    # [CPU] N=100000000
    # Total time : 1970.92 ms
    awk -v tag="$tag" '
        $0 ~ "\\["tag"\\]" {
            # extract N
            match($0, /N=([0-9]+)/, a)
            N=a[1]
            getline
            while ($0 !~ /Total time/) getline
            match($0, /Total time[^0-9]*([0-9.]+)/, b)
            T=b[1]
            printf("    (%s, %s)\n", N, T)
        }
    ' "$LOGFILE" >> "$OUTFILE"

    echo "};" >> "$OUTFILE"
    echo "\\addlegendentry{$label}" >> "$OUTFILE"
    echo "" >> "$OUTFILE"
}

# CPU baseline
extract_points "CPU" "CPU baseline" "mark=square*, color=black, dashed, thick"

# GPU2
extract_points "GPU2" "\\texttt{goldbach\\_gpu2} (global bitset)" "mark=o, color=blue, thick"

# GPU3
extract_points "GPU3" "\\texttt{goldbach\\_gpu3} (segmented)" "mark=triangle*, color=red, thick"

echo "Generated $OUTFILE"