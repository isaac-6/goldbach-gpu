#!/usr/bin/env bash
# test_phase2_fallback.sh
# Negative control for the per-segment verified/unverified count and Phase 2.
#
# At the default --p-small every number is resolved by Phase 1, so a count
# kernel that reported "0 unverified" regardless would pass every other test.
# Here --p-small=3 leaves most numbers unresolved, and the number that must
# reach Phase 2 is known exactly:
#
#   With p in {2, 3}, Phase 1 verifies n = 4 (2 + 2) and every n = q + 3 for
#   an odd prime q (p = 2 gives an even q, prime only for n = 4). Up to 10^6
#   the odd primes q <= 999997 number pi(10^6) - 1 = 78497, so 78498 of the
#   499999 even numbers in [4, 10^6] are verified and 421501 fall back.
#
# The count must be exact, and the run must still succeed (Phase 2 verifies
# them all), at the default segment size and across many small segments.
#
# Those complements stay below 10^8, where Phase 2 looks q up in its table. A
# second range, [10^8, 10^8 + 2*10^5] with --p-small=3, puts q above 10^8, so
# Phase 2 tests it with Miller-Rabin (the default) or BPSW (--primetest=bpsw).
# There the 100001 even numbers include 10903 with n - 3 prime
# (pi(100199997) - pi(99999996), from sympy.primepi), so 89098 fall back.
#
# Usage: test_phase2_fallback.sh <path-to-goldbach>
# Exit 0 = all cases pass.

set -u
GOLDBACH="${1:?usage: $0 <goldbach binary>}"
[ -x "$GOLDBACH" ] || { echo "FAIL: $GOLDBACH is not executable"; exit 1; }

EXPECTED=421501
fails=0

for args in "" "--seg-size=1000" "--seg-size=64 --batch-size=1"; do
    out=$("$GOLDBACH" 1000000 --p-small=3 $args 2>&1); rc=$?
    fb=$(sed -n 's/^Phase 2 fallbacks *: *\([0-9]*\)$/\1/p' <<<"$out")
    if [ "$rc" -ne 0 ]; then
        echo "  FAIL [$args] exit $rc"; printf '%s\n' "$out" | tail -3; fails=$((fails+1))
    elif ! grep -q "satisfy Goldbach" <<<"$out"; then
        echo "  FAIL [$args] no success line"; fails=$((fails+1))
    elif [ "$fb" != "$EXPECTED" ]; then
        echo "  FAIL [$args] Phase 2 fallbacks = '${fb}', expected $EXPECTED"; fails=$((fails+1))
    else
        echo "  ok   [$args] exit 0, Phase 2 fallbacks = $fb"
    fi
done

for pt in mr bpsw; do
    out=$("$GOLDBACH" 100200000 --start=100000000 --p-small=3 --primetest=$pt 2>&1); rc=$?
    fb=$(sed -n 's/^Phase 2 fallbacks *: *\([0-9]*\)$/\1/p' <<<"$out")
    if [ "$rc" -ne 0 ] || ! grep -q "satisfy Goldbach" <<<"$out" || [ "$fb" != 89098 ]; then
        echo "  FAIL [above 1e8, --primetest=$pt] exit $rc, Phase 2 fallbacks = '${fb}', expected 89098"
        fails=$((fails+1))
    else
        echo "  ok   [above 1e8, --primetest=$pt] exit 0, Phase 2 fallbacks = $fb"
    fi
done

if [ "$fails" -eq 0 ]; then echo "PASS"; exit 0; fi
echo "FAIL ($fails)"; exit 1
