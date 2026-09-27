#!/usr/bin/env bash
# run_sanitizers.sh
# Runs compute-sanitizer over the production GPU kernels and saves every log.
#
# Tools:   memcheck, initcheck, synccheck, racecheck (--racecheck-report all)
# Targets: test_gpu_sieve   segment sieve (tiled + large-prime kernels)
#          test_phase1      segment sieve + both Phase 1 kernels
#          sieve_driver     reduced driver, built here from the source below:
#                           the segment sieve on one small range, [3, 1e6], and
#                           one above 1e11, [1e11+1, 1e11+400001], where the
#                           large-prime kernel has work. For racecheck when the
#                           full test_gpu_sieve is too slow.
#
# Not part of ctest: it needs the CUDA debugger interface, which is not
# available everywhere. On WSL2 it must first be enabled from Windows by
# running EnableDebuggerInterface.bat as administrator. Without it every tool
# fails with "Failed to initialize WDDM debugger interface" and "Device not
# supported", and the summary line then counts those two tool errors as if
# they were findings. The preflight below detects that and stops.
#
# Usage: tests/run_sanitizers.sh <build-dir> [out-dir]
#   SANITIZER_TIMEOUT  per-run timeout, default 4h (racecheck is slow)
#   SANITIZER_TARGETS  space-separated subset of the targets above
# Exit 0 = every run clean; 1 = findings or failed runs (see summary.txt);
#      2 = sanitizer unavailable or setup error.

set -u

BUILD="${1:?usage: $0 <build-dir> [out-dir]}"
SRC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${2:-$BUILD/sanitizer-logs/$(date +%Y%m%d-%H%M%S)}"
TIMEOUT="${SANITIZER_TIMEOUT:-4h}"
TARGETS="${SANITIZER_TARGETS:-test_gpu_sieve test_phase1 sieve_driver}"
SAN="${COMPUTE_SANITIZER:-compute-sanitizer}"

command -v "$SAN" >/dev/null || { echo "ERROR: $SAN not found"; exit 2; }
mkdir -p "$OUT" || exit 2

for t in test_gpu_sieve test_phase1; do
    [ -x "$BUILD/bin/$t" ] || { echo "ERROR: $BUILD/bin/$t missing; build first"; exit 2; }
done

# --- Reduced driver ----------------------------------------------------------
cat > "$OUT/sieve_driver.cu" <<'EOF'
// Reduced sanitizer driver: the production segment sieve (sieve_kernel.cuh,
// both kernels) on two fixed ranges. Results are not compared; the sanitizers
// only need the kernels to execute. Exit 0 = launches completed.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "sieve_kernel.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static std::vector<uint64_t> primes_to(uint64_t n) {
    std::vector<char> c(n + 1, 1);
    std::vector<uint64_t> p;
    for (uint64_t i = 2; i <= n; i++) {
        if (!c[i]) continue;
        p.push_back(i);
        for (uint64_t j = i * i; j <= n; j += i) c[j] = 0;
    }
    return p;
}

int main() {
    struct { uint64_t lo, hi; } r[] = {{3, 1000000},
                                       {100000000001ULL, 100000400001ULL}};
    for (auto& t : r) {
        uint64_t root = 1; while ((root + 1) * (root + 1) <= t.hi) root++;
        auto pr = primes_to(root + 1);
        uint64_t words = ((t.hi - t.lo) / 2 + 1 + 63) / 64;
        uint64_t *db = nullptr, *dp = nullptr;
        CK(cudaMalloc(&db, words * sizeof(uint64_t)));
        CK(cudaMalloc(&dp, pr.size() * sizeof(uint64_t)));
        CK(cudaMemcpy(dp, pr.data(), pr.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        uint64_t sc = sieve_split_prime_count(pr.data(), pr.size());
        launch_segment_sieve(t.lo, t.hi, dp, sc, pr.size() - sc, db, 256, 0);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        printf("[%llu, %llu] tiled primes=%llu large primes=%llu\n",
               (unsigned long long)t.lo, (unsigned long long)t.hi,
               (unsigned long long)sc, (unsigned long long)(pr.size() - sc));
        CK(cudaFree(db));
        CK(cudaFree(dp));
    }
    return 0;
}
EOF

ARCH=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=\([0-9]*\).*/\1/p' "$BUILD/CMakeCache.txt" 2>/dev/null)
ARCH_FLAG="-arch=${ARCH:+sm_$ARCH}"; [ -n "$ARCH" ] || ARCH_FLAG="-arch=native"
if ! nvcc -std=c++17 -O3 -lineinfo "$ARCH_FLAG" -I"$SRC_ROOT/include" \
          "$OUT/sieve_driver.cu" -o "$OUT/sieve_driver" > "$OUT/sieve_driver.build.log" 2>&1; then
    echo "ERROR: could not build the reduced driver; see $OUT/sieve_driver.build.log"; exit 2
fi

target_path() {
    case "$1" in
        sieve_driver) echo "$OUT/sieve_driver" ;;
        *)            echo "$BUILD/bin/$1" ;;
    esac
}

# A log is a tool failure, not a result, if the sanitizer never attached.
tool_failed() { grep -qE "Failed to initialize WDDM|Device not supported|Unable to (attach|initialize)" "$1"; }

# --- Preflight ---------------------------------------------------------------
"$SAN" --tool memcheck "$OUT/sieve_driver" > "$OUT/preflight.log" 2>&1
if tool_failed "$OUT/preflight.log"; then
    echo "ERROR: compute-sanitizer cannot attach to the GPU on this machine:"
    grep -E "Error:" "$OUT/preflight.log" | sed 's/^/    /'
    echo "On WSL2, run EnableDebuggerInterface.bat as administrator on Windows."
    echo "No sanitizer results were produced. Preflight log: $OUT/preflight.log"
    exit 2
fi

# --- Runs --------------------------------------------------------------------
# Distinct from any exit code the targets use, so findings are not confused
# with a failing test.
FINDINGS_RC=86
SUMMARY="$OUT/summary.txt"
{
    echo "compute-sanitizer: $("$SAN" --version | tail -1)"
    echo "source: $(git -C "$SRC_ROOT" rev-parse --short HEAD 2>/dev/null)$(git -C "$SRC_ROOT" diff --quiet 2>/dev/null || echo ' +uncommitted changes')"
    echo "arch: $ARCH_FLAG"
    echo
} > "$SUMMARY"

bad=0
for target in $TARGETS; do
    exe=$(target_path "$target")
    [ -x "$exe" ] || { echo "unknown or missing target: $target" | tee -a "$SUMMARY"; bad=1; continue; }
    for tool in memcheck initcheck synccheck racecheck; do
        extra=()
        [ "$tool" = racecheck ] && extra=(--racecheck-report all)
        log="$OUT/${tool}__${target}.log"
        start=$(date +%s)
        timeout "$TIMEOUT" "$SAN" --tool "$tool" "${extra[@]}" --print-limit 0 \
            --error-exitcode "$FINDINGS_RC" --log-file "$log" \
            "$exe" > "$OUT/${tool}__${target}.stdout" 2>&1
        rc=$?
        secs=$(( $(date +%s) - start ))
        # compute-sanitizer writes its own summary as the last lines of the log.
        verdict=$(grep -E "(ERROR|RACECHECK) SUMMARY" "$log" 2>/dev/null | tail -1 | sed 's/^=* *//')
        # Racecheck warnings do not necessarily set the error exit code, so the
        # count in the summary line is checked as well as the exit code.
        reported=$(grep -oE '[0-9]+ (hazards?|errors?)' <<<"$verdict" | head -1 | cut -d' ' -f1)
        if tool_failed "$log"; then
            status="TOOL-FAILED"
        elif [ "$rc" -eq 124 ]; then
            status="TIMEOUT"
        elif [ "$rc" -eq "$FINDINGS_RC" ] || [ "${reported:-0}" -gt 0 ]; then
            status="FINDINGS"
        elif [ -z "$verdict" ]; then
            status="NO-SUMMARY"
        elif [ "$rc" -ne 0 ]; then
            status="TARGET-FAILED(rc=$rc)"
        else
            status="clean"
        fi
        [ "$status" = clean ] || bad=1
        printf '%-10s %-15s %-22s %6ss  %s\n' "$tool" "$target" "$status" "$secs" "$verdict" \
            | tee -a "$SUMMARY"
    done
done

echo
echo "Logs: $OUT"
exit "$bad"
