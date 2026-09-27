#!/usr/bin/env bash
# test_record_check.sh
# End-to-end test of goldbach --record-check up to 10^8.
#
# The reference is every p-record below 10^8: n whose minimal prime p_min(n)
# exceeds that of every smaller even number. It was computed by brute force (a
# plain sieve and an ascending scan over every even n <= 10^8, sharing no code
# with this project) and agrees with the first 29 entries of the published
# table in src/test_records.cpp.
#
# The verifier reports at most one record per segment, so its output must be
# a SUBSEQUENCE of this list, in ascending order, and must include the
# maximum, n = 60119912 with p_min = 1093. Three configurations:
#   default parameters               one segment
#   --seg-size=1000000 --batch-size=7 --p-small=1100
#                                    50 segments, ~27 launches per segment
#   --seg-size=100000 --p-small=1020 --batch-size=1
#                                    500 segments, one launch per prime; the
#                                    records 1039 and 1093 exceed P_SMALL and
#                                    come from Phase 2. Before Phase 2 reported
#                                    p_min this printed the non-record
#                                    79097318 1009 and missed 1093.
#
# Usage: test_record_check.sh <path-to-goldbach>
# Exit 0 = all configurations pass.

set -u
GOLDBACH="${1:?usage: $0 <goldbach binary>}"
[ -x "$GOLDBACH" ] || { echo "FAIL: $GOLDBACH is not executable"; exit 1; }

REFERENCE="4 2
6 3
12 5
30 7
98 19
220 23
308 31
556 47
992 73
2642 103
5372 139
7426 173
43532 211
54244 233
63274 293
113672 313
128168 331
194428 359
194470 383
413572 389
503222 523
1077422 601
3526958 727
3807404 751
10759922 829
24106882 929
27789878 997
37998938 1039
60119912 1093"
MAXIMUM="60119912 1093"

fails=0
check() {
    local out rc got
    out=$("$GOLDBACH" 100000000 --record-check "$@" 2>&1); rc=$?
    got=$(sed -n 's/^\[record\] n=\([0-9]*\) p_min=\([0-9]*\)$/\1 \2/p' <<<"$out")
    local bad=0 why=""
    [ "$rc" -eq 0 ] || { bad=1; why="exit $rc"; }
    grep -q "satisfy Goldbach" <<<"$out" || { bad=1; why="$why; no success line"; }
    [ -n "$got" ] || { bad=1; why="$why; no records printed"; }
    # Subsequence: every printed pair is a reference pair, and n ascends.
    local extra
    extra=$(grep -vxF -f <(printf '%s\n' "$REFERENCE") <<<"$got")
    [ -z "$extra" ] || { bad=1; why="$why; not records: $(tr '\n' ',' <<<"$extra")"; }
    sort -n -k1,1 -c <<<"$got" 2>/dev/null || { bad=1; why="$why; not ascending"; }
    grep -qxF "$MAXIMUM" <<<"$got" || { bad=1; why="$why; maximum $MAXIMUM missing"; }
    if [ "$bad" -eq 0 ]; then
        echo "  ok   [$*] $(wc -l <<<"$got") records, all genuine, maximum present"
    else
        echo "  FAIL [$*]$why"; fails=$((fails+1))
    fi
}

check
check --seg-size=1000000 --batch-size=7 --p-small=1100
check --seg-size=100000 --p-small=1020 --batch-size=1

if [ "$fails" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "FAIL ($fails)"; exit 1
