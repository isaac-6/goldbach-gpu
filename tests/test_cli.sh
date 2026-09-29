#!/usr/bin/env bash
# test_cli.sh
# Command-line validation for goldbach, big_check and single_check.
#
# Every input below that breaks an invariant the code relies on must be
# rejected with exit 1 and a message naming the problem, before any work is
# done. Several used to be accepted: a batch size of 0 hung; P_SMALL = 0 with
# a 2^32 + 64 segment wrapped the unverified count and printed a false
# success; LIMIT near 2^64 wrapped the segment counter or the sieve; negative
# values wrapped through std::stoull; malformed values aborted on an uncaught
# exception. The accepted edge cases at the end must still run and succeed.
#
# Usage: test_cli.sh <goldbach> <big_check> <single_check>
# Exit 0 = all cases pass.

set -u
GOLDBACH="${1:?usage: $0 <goldbach> <big_check> <single_check>}"
BIG_CHECK="${2:?usage: $0 <goldbach> <big_check> <single_check>}"
SINGLE_CHECK="${3:?usage: $0 <goldbach> <big_check> <single_check>}"

fails=0; n=0

# reject <binary> <expected message substring> <args...>
reject() {
    local bin="$1" want="$2"; shift 2
    local out rc
    out=$(timeout 120 "$bin" "$@" 2>&1); rc=$?
    n=$((n+1))
    if [ "$rc" -eq 1 ] && grep -qF -- "$want" <<<"$out"; then
        echo "  ok   $(basename "$bin") $*"
    else
        echo "  FAIL $(basename "$bin") $*: exit $rc, expected 1 with '$want'"
        printf '%s\n' "$out" | grep -v '^ *$' | tail -2 | sed 's/^/         /'
        fails=$((fails+1))
    fi
}

# accept <expected output substring> <goldbach args...>
accept() {
    local want="$1"; shift
    local out rc
    out=$(timeout 300 "$GOLDBACH" "$@" 2>&1); rc=$?
    n=$((n+1))
    if [ "$rc" -eq 0 ] && grep -qF -- "$want" <<<"$out"; then
        echo "  ok   goldbach $* (accepted)"
    else
        echo "  FAIL goldbach $*: exit $rc, expected 0 with '$want'"; fails=$((fails+1))
    fi
}

G="$GOLDBACH"
echo "[goldbach: rejected]"
reject "$G" "--batch-size must be between 1 and"   1000000 --batch-size=0
reject "$G" "--batch-size must be between 1 and"   1000000 --batch-size=4294967297
reject "$G" "--batch-size must be between 1 and"   1000000 --batch-size=2305843009213693953
reject "$G" "P_SMALL must be between 3 and"        1000000 --p-small=2
reject "$G" "P_SMALL must be between 3 and"        1000000 --p-small=0
reject "$G" "P_SMALL must be between 3 and"        1000000 --p-small=4000000001
reject "$G" "P_SMALL must be between 3 and"        1000000 5000 2
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 1000000 --seg-size=0
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 1000000 --seg-size=1000001
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 1000000 --seg-size=4294967296
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 1000000 --seg-size=18446744073709551614
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 1000000 0
reject "$G" "SEG_SIZE must be even, > 0 and < 2^32" 8589934722 --seg-size=4294967360 --p-small=0
reject "$G" "LIMIT must be <= 18446744065119617024" 18446744065119617025
reject "$G" "LIMIT must be <= 18446744065119617024" 18446744073709551614 --start=18446744073709551000 --seg-size=128
reject "$G" "is out of range"                       18446744073709551616
reject "$G" "LIMIT + 2*SEG_SIZE*(GPUs+1) exceeds"   18446744065119617024 --start=18446744065119617000 --seg-size=4294967294
reject "$G" "LIMIT must be >= 4"                    3
reject "$G" "too many positional arguments"         1000000 5000 1000 7
reject "$G" "unknown option"                        1000000 --p-smal=5
reject "$G" "must be a non-negative decimal integer" 1000000 --batch-size=-1
reject "$G" "must be a non-negative decimal integer" 1000000 --seg-size=-2
reject "$G" "must be a non-negative decimal integer" 1000000 --p-small=-1
reject "$G" "must be a non-negative decimal integer" 1000000 --start=-4
reject "$G" "must be non-negative decimal integers"  -1000000
reject "$G" "must be a non-negative decimal integer" 1000000 --gpus=abc
reject "$G" "--gpus must be -1 (all) or between 1"   1000000 --gpus=0
reject "$G" "must be a non-negative decimal integer" 1000000 --gpus=-2
reject "$G" "must be a non-negative decimal integer" 1000000 --start=
reject "$G" "must be a non-negative decimal integer" 1000000 --batch-size=1e5
reject "$G" "must be a non-negative decimal integer" 1e6
reject "$G" "START must be <= LIMIT"                1000000 --start=1000002
reject "$G" "contains no even number to check"     1000001 --start=1000001
reject "$G" "contains no even number to check"     5 --start=5
reject "$G" "unknown --primetest value"             1000000 --primetest=XX
reject "$G" "LIMIT is required"                     --seg-size=1000000
reject "$G" "--count-file requires --count-primes"  1000000 --count-file=x.txt
reject "$G" "--record-check requires --start=4"     2000000 --record-check --start=1100000

echo "[goldbach: accepted edge cases]"
accept "All even numbers from 4 up to 4 satisfy"             4
accept "All even numbers from 4 up to 4 satisfy"             5
accept "All even numbers from 1000000 up to 1000000 satisfy" 1000000 --start=1000000
accept "All even numbers from 1000000 up to 1000000 satisfy" 1000000 --start=999999
accept "All even numbers from 4 up to 2000000 satisfy"       2000000 --record-check --start=3
accept "satisfy Goldbach"                                    18446744065119617024 --start=18446744065119616000 --seg-size=64
# The batch buffer is sized by the primes actually used, not by --batch-size:
# 2^32 used to ask for 32 GiB of device memory for 78,498 primes.
accept "All even numbers from 4 up to 1000000 satisfy"       1000000 --batch-size=4294967296

echo "[big_check]"
reject "$BIG_CHECK" "--p-max must be <="  "10^30" --p-max=18446744073709551615
reject "$BIG_CHECK" "--p-max must be <="  "10^30" --p-max=4000000001
reject "$BIG_CHECK" "N must be an even integer" 7

echo "[single_check]"
reject "$SINGLE_CHECK" "n must be an even integer" 1000000000001
reject "$SINGLE_CHECK" "Number is too large"       100000000000000000000
reject "$SINGLE_CHECK" "n must be >= 4"            2
# Strict parsing: std::stoull ran "-2" as 2^64 - 2 and "12abc" as 12.
S1="$SINGLE_CHECK"
reject "$S1" "must be a non-negative decimal integer" -2
reject "$S1" "must be a non-negative decimal integer" 12abc
reject "$S1" "must be a non-negative decimal integer" +12
reject "$S1" "must be a non-negative decimal integer" " 12"
reject "$S1" "must be a non-negative decimal integer" 0x10
reject "$S1" "must be a non-negative decimal integer" ""
reject "$S1" "expected one argument"                  100 200

echo
if [ "$fails" -eq 0 ]; then echo "test_cli: all $n cases PASS"; exit 0; fi
echo "test_cli: $fails of $n cases FAILED"; exit 1
