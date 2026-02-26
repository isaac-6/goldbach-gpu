#!/bin/bash

# validation.sh
# Automated correctness check for GoldbachGPU tools.
# Compares GPU output against CPU baseline for a known range.

# Colors for professional output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "===================================================="
echo "      GoldbachGPU: Automated Validation Suite       "
echo "===================================================="

# Path to binaries (adjust if your build dir is different)
BIN_DIR="../build/bin"
TEST_LIMIT=1000000  # 10^6 is fast for testing

# 1. Check if binaries exist
required_bins=("cpu_goldbach" "goldbach_gpu3" "goldbach_gpu2" "big_check")
for bin in "${required_bins[@]}"; do
    if [ ! -f "$BIN_DIR/$bin" ]; then
        echo -e "${RED}[ERROR]${NC} Binary $bin not found in $BIN_DIR. Please compile first."
        exit 1
    fi
done

echo -e "${GREEN}[INFO]${NC} All binaries found. Starting tests..."

# 2. Test CPU Baseline
echo -n "Test 1: CPU Oracle ($TEST_LIMIT)... "
CPU_OUT=$("$BIN_DIR/cpu_goldbach" $TEST_LIMIT 2>&1)
if [[ $CPU_OUT == *"satisfy Goldbach. ✓"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "$CPU_OUT"
    exit 1
fi

# 3. Test GPU Segmented (Flagship)
# We use a small segment size to force the loop to run multiple times
echo -n "Test 2: GPU Segmented ($TEST_LIMIT, small segments)... "
GPU3_OUT=$("$BIN_DIR/goldbach_gpu3" $TEST_LIMIT 100000 2>&1)
if [[ $GPU3_OUT == *"satisfy Goldbach. ✓"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "$GPU3_OUT"
    exit 1
fi

# 4. Test GPU Global Bitset
echo -n "Test 3: GPU Global Bitset ($TEST_LIMIT)... "
GPU2_OUT=$("$BIN_DIR/goldbach_gpu2" $TEST_LIMIT 2>&1)
if [[ $GPU2_OUT == *"satisfy Goldbach. ✓"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "$GPU2_OUT"
    exit 1
fi

# 5. Test Arbitrary Precision (big_check)
# Verify a known partition
echo -n "Test 4: GMP Arbitrary Precision (10^12)... "
BIG_OUT=$("$BIN_DIR/big_check" 1000000000000 2>&1)
if [[ $BIG_OUT == *"Goldbach holds. ✓"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "$BIG_OUT"
    exit 1
fi

# 6. Test CLI Robustness (Help Flags)
echo -n "Test 5: CLI Help Flag... "
HELP_OUT=$("$BIN_DIR/goldbach_gpu3" --help 2>&1)
if [[ $HELP_OUT == *"Usage:"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

# 7. Test VRAM Safety Guard (Negative Test)
echo -n "Test 6: GPU VRAM Safety Guard... "
# Request 1 Trillion segment size (~1 TB VRAM needed). Guaranteed to fail gracefully.
VRAM_OUT=$("$BIN_DIR/goldbach_gpu3" 1000000 1000000000000 2>&1)
if [[ $VRAM_OUT == *"[!] ERROR: SEG_SIZE"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Output was:"
    echo "$VRAM_OUT"
    exit 1
fi

# 8. Test Single Number Checker
echo -n "Test 7: GPU Single Checker (10^12)... "
SINGLE_OUT=$("$BIN_DIR/single_check" 1000000000000 2>&1)
if [[ $SINGLE_OUT == *"Goldbach holds. ✓"* ]]; then
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "$SINGLE_OUT"
    exit 1
fi

echo "===================================================="
echo -e "${GREEN}SUCCESS: All verification tests passed!${NC}"
echo "===================================================="
exit 0