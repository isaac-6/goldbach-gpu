#!/usr/bin/env bash
# test_window.sh
# Window mode: --count-primes with a START above the small-prime bound, and
# --window-max, which reports the largest p_min over [START, N] and the
# smallest n attaining it.
#
# Expected values were computed outside the project:
#   - prime counts with primesieve 12.16:
#       primesieve 4000000000000000001 4000000000100000000 --count   -> 2335857
#       primesieve 1000000000000000001 1000000000100000000 --count   -> 2414886
#   - window maxima by brute force over every even n, with GMP's
#     mpz_probab_prime_p (deterministic below 2^64);
#   - the tie windows by a plain byte sieve.
#
# Cases:
#   1 [4e18, 4e18 + 1e8]   count and window maximum, 50,000,001 even numbers
#   2 [1e18, 1e18 + 1e8]   the same at 1e18
#   3 [3000128, 3000254]   one 64-number group on the transposed path whose
#                          maximum p_min, 107, occurs at 3000196 and 3000238:
#                          the smaller must be reported
#   4 [90, 130]            p_min 19 at 98 and 128, on the scalar path
#   5 [90, 130]            the same through Phase 2 (--p-small=17)
#
# Usage: test_window.sh <path-to-goldbach>
# Exit 0 = all cases pass.

set -u
GOLDBACH="${1:?usage: $0 <goldbach binary>}"
[ -x "$GOLDBACH" ] || { echo "FAIL: $GOLDBACH is not executable"; exit 1; }

fails=0
# check <label> <expected "p n" or -> <expected count or -> <goldbach args...>
check() {
    local label="$1" want_max="$2" want_count="$3"; shift 3
    local out rc got_max got_count why=""
    out=$("$GOLDBACH" "$@" 2>&1); rc=$?
    got_max=$(sed -n 's/^Window maximum p_min *: *\([0-9]*\) at n = \([0-9]*\) .*/\1 \2/p' <<<"$out")
    got_count=$(sed -n 's/^primes in ([0-9]*, [0-9]*\] = \([0-9]*\)$/\1/p' <<<"$out")
    [ "$rc" -eq 0 ] || why="$why exit $rc;"
    grep -q "satisfy Goldbach" <<<"$out" || why="$why no success line;"
    [ "$want_max" = - ] || [ "$got_max" = "$want_max" ] || why="$why window max '$got_max', expected '$want_max';"
    [ "$want_count" = - ] || [ "$got_count" = "$want_count" ] || why="$why count '$got_count', expected '$want_count';"
    if [ -z "$why" ]; then echo "  ok   $label"; else echo "  FAIL $label:$why"; fails=$((fails+1)); fi
}

check "4e18 + 1e8: count and window max" "3631 4000000000077765314" 2335857 \
      4000000000100000000 --start=4000000000000000000 --count-primes --window-max
check "1e18 + 1e8: count and window max" "3769 1000000000066133978" 2414886 \
      1000000000100000000 --start=1000000000000000000 --count-primes --window-max
check "transposed tie [3000128, 3000254]" "107 3000196" - 3000254 --start=3000128 --window-max
check "scalar tie [90, 130]"              "19 98"       - 130 --start=90 --window-max
check "Phase 2 tie [90, 130]"             "19 98"       - 130 --start=90 --window-max --p-small=17

if [ "$fails" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "FAIL ($fails)"; exit 1
