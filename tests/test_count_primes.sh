#!/usr/bin/env bash
# test_count_primes.sh
# Functional test for goldbach --count-primes.
#
# Three cases:
#   1 1e9          pi(10^9)        = 50,847,534   (OEIS A006880)
#   2 123456789    pi(123,456,789) = 7,027,260    (primesieve 12.12)
#   3 segments     123,456,789 again with --seg-size=1000000, so the count is
#                  assembled from 62 segments instead of 1; the --count-file
#                  lines must tile [5, N] and sum, with 2 and 3, to pi(N)
#
# At the default segment size case 2 is a single segment, so only case 3
# exercises the segment-boundary arithmetic (each segment counting its own odd
# range and not the P_SMALL overlap below it). N = 123,456,789 is odd and not
# a segment boundary in either layout.
#
# Usage: test_count_primes.sh <path-to-goldbach>
# Exit 0 = all cases pass.

set -u

GOLDBACH="${1:?usage: $0 <goldbach binary>}"
[ -x "$GOLDBACH" ] || { echo "FAIL: $GOLDBACH is not executable"; exit 1; }

fails=0
ok()  { printf '  ok   %s\n' "$*"; }
bad() { printf '  FAIL %s\n' "$*"; fails=$((fails+1)); }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Runs goldbach --count-primes and checks the single "pi(N) = v" line.
# $1 = N, $2 = expected pi(N), remaining args passed through.
check_pi() {
    local n="$1" want="$2"; shift 2
    local out rc lines got
    out=$("$GOLDBACH" "$n" --count-primes "$@" 2>&1); rc=$?
    if [ "$rc" -ne 0 ]; then
        bad "N=$n $*: exit code $rc"; printf '%s\n' "$out" | tail -5; return
    fi
    lines=$(grep -c "^pi($n) = " <<<"$out")
    if [ "$lines" -ne 1 ]; then
        bad "N=$n $*: expected exactly one 'pi($n) = ' line, found $lines"; return
    fi
    got=$(sed -n "s/^pi($n) = \([0-9]*\)$/\1/p" <<<"$out")
    if [ "$got" = "$want" ]; then ok "pi($n) = $got $*"; else bad "pi($n) = '$got', expected $want $*"; fi
}

echo "[1/3] pi(10^9)"
check_pi 1000000000 50847534

echo "[2/3] pi(123456789)"
check_pi 123456789 7027260

echo "[3/3] pi(123456789) from 62 segments, and the --count-file"
check_pi 123456789 7027260 --seg-size=1000000 --count-file="$TMP/counts.txt"
if [ -f "$TMP/counts.txt" ]; then
    # Each line "A B count": A = 5 on the first line, then previous B + 2 (the
    # ranges are adjacent over the odd numbers); last B = N; sum + 2 = pi(N).
    res=$(awk 'NR==1 && $1!=5 {b++} NR>1 && $1!=prev+2 {b++} {prev=$2; s+=$3}
               END {print NR, b+0, prev, s+2}' "$TMP/counts.txt")
    read -r nlines breaks lastb total <<<"$res"
    if [ "$nlines" -eq 62 ] && [ "$breaks" -eq 0 ] && [ "$lastb" = 123456789 ] \
       && [ "$total" = 7027260 ]; then
        ok "count file: 62 lines, contiguous to 123456789, sum + 2 = 7027260"
    else
        bad "count file: lines=$nlines breaks=$breaks last B=$lastb sum+2=$total"
    fi
else
    bad "count file was not written"
fi

if [ "$fails" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "FAIL ($fails)"; exit 1
