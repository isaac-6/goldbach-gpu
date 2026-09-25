#!/usr/bin/env bash
# test_big_check.sh
# Functional test for big_check.
#
# Seven cases:
#   1 records      every published p-record below 1e13 is reproduced by big_check
#   2 edges        n = 4, 6, 8
#   3 bound        n = 128 --p-max=17 must exit 3 (no partition within the bound)
#   4 determinism  the 1e12 record gives the same p on 1 thread and on all threads
#   5 expression   10^30 and its decimal string give identical RESULT lines
#   6 invalid      malformed and out-of-domain inputs exit 1
#   7 digits       reported digit counts are exact, and a 100-digit q prints in full
#
# The record pairs are not duplicated here: they are read out of
# src/test_records.cpp, which carries the provenance for that table, so the two
# tests cannot drift apart.
#
# Usage: test_big_check.sh <path-to-big_check> <path-to-source-root>
# Exit 0 = all cases pass.

set -u

BIG_CHECK="${1:?usage: $0 <big_check binary> <source root>}"
SRC_ROOT="${2:?usage: $0 <big_check binary> <source root>}"

# Number of published p-records below 1e13 in src/test_records.cpp. The parse
# below must yield exactly this many. Without that guard a sed pattern that
# silently stopped matching would reduce this case to testing nothing at all
# while still reporting PASS.
EXPECTED_RECORDS=48

# Overridable so the guard can be exercised against deliberately damaged copies
# without touching the real table.
RECORDS_SRC="${RECORDS_SRC:-$SRC_ROOT/src/test_records.cpp}"

[ -x "$BIG_CHECK" ]    || { echo "FAIL: $BIG_CHECK is not executable"; exit 1; }
[ -f "$RECORDS_SRC" ]  || { echo "FAIL: cannot read $RECORDS_SRC"; exit 1; }

fails=0
cases=0

note() { printf '  %s\n' "$*"; }
ok()   { printf '  ok   %s\n' "$*"; }
bad()  { printf '  FAIL %s\n' "$*"; fails=$((fails+1)); }

# Extract "status=... p=..." fields from a RESULT line.
field() { sed -n 's/.*[[:space:]]'"$1"'=\([^[:space:]]*\).*/\1/p' <<<"$2"; }

# ---------------------------------------------------------------
# Case 1: published p-records
# ---------------------------------------------------------------
echo "[1/7] published p-records below 1e13"
cases=$((cases+1))
mapfile -t RECS < <(sed -n '/^static const Record RECORDS\[\]/,/^};/p' "$RECORDS_SRC" \
                    | sed -n 's/^[[:space:]]*{[[:space:]]*\([0-9]\+\)ULL[[:space:]]*,[[:space:]]*\([0-9]\+\)[[:space:]]*}.*/\1 \2/p')
# Guard 1: the parse must recover every record, not merely some of them.
if [ "${#RECS[@]}" -ne "$EXPECTED_RECORDS" ]; then
    echo "FAIL: parsed ${#RECS[@]} records from $RECORDS_SRC, expected $EXPECTED_RECORDS" >&2
    exit 1
fi

# Guard 2: every record must be a plausible (n, p_min) pair -- two positive
# integers with n even and n >= 4. This catches a parse that still yields the
# right number of lines but picks up the wrong fields, which guard 1 cannot see.
for line in "${RECS[@]}"; do
    rn=${line%% *}; rp=${line##* }
    if ! [[ "$rn" =~ ^[0-9]+$ ]] || ! [[ "$rp" =~ ^[0-9]+$ ]] \
       || [ "$rp" -lt 1 ] || [ "$rn" -lt 4 ] || [ $(( rn % 2 )) -ne 0 ]; then
        echo "FAIL: malformed record parsed from $RECORDS_SRC: '$line'" >&2
        echo "      expected two positive integers 'n p' with n even and n >= 4" >&2
        exit 1
    fi
done

note "${#RECS[@]} records parsed and validated"
rec_fail=0
for line in "${RECS[@]}"; do
    n=${line% *}; want=${line#* }
    out=$("$BIG_CHECK" "$n" --quiet 2>&1)
    rc=$?
    got=$(field p "$out")
    if [ "$rc" -ne 0 ] || [ "$got" != "$want" ]; then
        bad "n=$n expected p=$want got p=${got:-<none>} (exit $rc)"
        rec_fail=$((rec_fail+1))
        [ "$rec_fail" -ge 5 ] && { note "... stopping after 5 record failures"; break; }
    fi
done
[ "$rec_fail" -eq 0 ] && ok "all ${#RECS[@]} records reproduced"

# ---------------------------------------------------------------
# Case 2: small even numbers
# ---------------------------------------------------------------
echo "[2/7] edge cases n = 4, 6, 8"
cases=$((cases+1))
edge_fail=0
for pair in "4 2" "6 3" "8 3"; do
    n=${pair% *}; want=${pair#* }
    out=$("$BIG_CHECK" "$n" --quiet 2>&1); rc=$?
    got=$(field p "$out")
    if [ "$rc" -ne 0 ] || [ "$got" != "$want" ]; then
        bad "n=$n expected p=$want got p=${got:-<none>} (exit $rc)"; edge_fail=1
    fi
done
[ "$edge_fail" -eq 0 ] && ok "n = 4, 6, 8 give p = 2, 3, 3"

# ---------------------------------------------------------------
# Case 3: search-limit result
# ---------------------------------------------------------------
echo "[3/7] n = 128 --p-max=17 exits 3"
cases=$((cases+1))
out=$("$BIG_CHECK" 128 --p-max=17 --quiet 2>&1); rc=$?
st=$(field status "$out")
if [ "$rc" -ne 3 ]; then
    bad "expected exit 3, got $rc"
elif [ "$st" != "bound" ]; then
    bad "expected status=bound, got status=${st:-<none>}"
elif grep -qi 'counterexample' <<<"$out"; then
    bad "output still mentions 'counterexample'"
else
    ok "exit 3, status=bound, no 'counterexample' wording"
fi

# ---------------------------------------------------------------
# Case 4: determinism in thread count
# ---------------------------------------------------------------
echo "[4/7] determinism at the 1e12 record"
cases=$((cases+1))
N_DET=3132059294006
out1=$(OMP_NUM_THREADS=1 "$BIG_CHECK" "$N_DET" --quiet 2>&1); rc1=$?
outN=$("$BIG_CHECK" "$N_DET" --quiet 2>&1); rcN=$?
p1=$(field p "$out1"); pN=$(field p "$outN")
c1=$(field candidates "$out1"); cN=$(field candidates "$outN")
t1=$(field threads "$out1"); tN=$(field threads "$outN")
if [ "$rc1" -ne 0 ] || [ "$rcN" -ne 0 ]; then
    bad "n=$N_DET exit codes $rc1 / $rcN"
elif [ "$p1" != "$pN" ] || [ "$c1" != "$cN" ]; then
    bad "1 thread gave p=$p1 candidates=$c1; $tN threads gave p=$pN candidates=$cN"
else
    ok "p=$p1 candidates=$c1 on $t1 thread and on $tN threads"
fi

# ---------------------------------------------------------------
# Case 5: expression input
# ---------------------------------------------------------------
echo "[5/7] 10^30 matches its decimal string"
cases=$((cases+1))
DEC=1000000000000000000000000000000
# Pinned to one thread on both sides. prp_calls counts tests actually issued,
# and threads skip candidates above the current best as soon as a hit lands, so
# that count legitimately varies with scheduling. Fixing the thread count makes
# every field of the RESULT line except time_s deterministic, which is what lets
# the two spellings of the same n be compared line for line.
oute=$(OMP_NUM_THREADS=1 "$BIG_CHECK" "10^30" --quiet 2>&1); rce=$?
outd=$(OMP_NUM_THREADS=1 "$BIG_CHECK" "$DEC"  --quiet 2>&1); rcd=$?
# compare the RESULT lines with time_s stripped
stripe=$(sed 's/ time_s=[^ ]*//' <<<"$oute")
stripd=$(sed 's/ time_s=[^ ]*//' <<<"$outd")
if [ "$rce" -ne 0 ] || [ "$rcd" -ne 0 ]; then
    bad "exit codes $rce / $rcd"
elif [ "$stripe" != "$stripd" ]; then
    bad "RESULT lines differ:"; note "expr: $stripe"; note "dec : $stripd"
else
    ok "identical apart from time_s: $stripe"
fi

# ---------------------------------------------------------------
# Case 6: invalid input
# ---------------------------------------------------------------
echo "[6/7] invalid input exits 1"
cases=$((cases+1))
inv_fail=0
for bad_in in "7" "2" "0" "abc" "10^" "^5" "12x" "-4"; do
    "$BIG_CHECK" "$bad_in" --quiet >/dev/null 2>&1
    rc=$?
    if [ "$rc" -ne 1 ]; then
        bad "input '$bad_in' expected exit 1, got $rc"; inv_fail=1
    fi
done
[ "$inv_fail" -eq 0 ] && ok "odd, too-small and malformed inputs all exit 1"

# ---------------------------------------------------------------
# Case 7: exact digit counts
# ---------------------------------------------------------------
# mpz_sizeinbase(x, 10) may return one more than the true digit count, so it
# cannot be used for a reported figure. 10^100 - 797 has exactly 100 digits and
# is the case that exposed it; 2^1000 - 16607 is a control, since the old code
# was already right whenever the leading digit did not roll over.
echo "[7/7] exact digit counts and full-q printing"
cases=$((cases+1))
dig_fail=0

out=$("$BIG_CHECK" "10^100" --quiet 2>&1); rc=$?
dp=$(field p "$out"); dq=$(field q_digits "$out")
if [ "$rc" -ne 0 ] || [ "$dp" != "797" ] || [ "$dq" != "100" ]; then
    bad "10^100 expected p=797 q_digits=100, got p=${dp:-<none>} q_digits=${dq:-<none>} (exit $rc)"
    dig_fail=1
fi

# 100 digits is within the <= 100 threshold, so q must appear in full rather
# than as the leading/trailing 20 digits joined by an ellipsis.
vout=$("$BIG_CHECK" "10^100" 2>&1)
qline=$(grep -m1 '^q = ' <<<"$vout")
qval=${qline#q = }
if [ "${#qval}" -ne 100 ] || [[ "$qval" == *...* ]]; then
    bad "10^100: q not printed in full (${#qval} chars: '${qval:0:40}')"
    dig_fail=1
fi

out=$("$BIG_CHECK" "2^1000" --quiet 2>&1); rc=$?
dq=$(field q_digits "$out")
if [ "$rc" -ne 0 ] || [ "$dq" != "302" ]; then
    bad "2^1000 expected q_digits=302, got ${dq:-<none>} (exit $rc)"
    dig_fail=1
fi

[ "$dig_fail" -eq 0 ] && ok "10^100 -> p=797, q_digits=100, q printed in full; 2^1000 -> q_digits=302"

echo
if [ "$fails" -eq 0 ]; then
    echo "test_big_check: all $cases cases PASS"
    exit 0
fi
echo "test_big_check: $fails failure(s) across $cases cases"
exit 1
