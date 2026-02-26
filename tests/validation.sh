#!/bin/bash

# validation.sh
# Automated correctness and robustness check for GoldbachGPU tools.

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "===================================================="
echo "      GoldbachGPU: Automated Validation Suite       "
echo "===================================================="

BIN_DIR="../build/bin"
TEST_LIMIT=1000000

# ---------------------------------------------------------
# PRE-FLIGHT CHECK
# ---------------------------------------------------------
required_bins=("cpu_goldbach" "goldbach_gpu3" "goldbach_gpu2" "big_check" "single_check")
for bin in "${required_bins[@]}"; do
    if [ ! -f "$BIN_DIR/$bin" ]; then
        echo -e "${RED}[ERROR]${NC} Binary $bin not found. Please compile first."
        exit 1
    fi
done
echo -e "${GREEN}[INFO] All binaries found.${NC}"

# ---------------------------------------------------------
# PART 1: CORE FUNCTIONALITY (Positive Tests)
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 1: Core Functionality ---${NC}"

echo -n "Test 1.1: CPU Oracle ($TEST_LIMIT)... "
CPU_OUT=$("$BIN_DIR/cpu_goldbach" $TEST_LIMIT 2>&1)
if [[ $CPU_OUT == *"satisfy Goldbach. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$CPU_OUT"; exit 1; fi

echo -n "Test 1.2: GPU Segmented ($TEST_LIMIT, 100k segments)... "
GPU3_OUT=$("$BIN_DIR/goldbach_gpu3" $TEST_LIMIT 100000 2>&1)
if [[ $GPU3_OUT == *"satisfy Goldbach. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$GPU3_OUT"; exit 1; fi

echo -n "Test 1.3: GPU Global Bitset ($TEST_LIMIT)... "
GPU2_OUT=$("$BIN_DIR/goldbach_gpu2" $TEST_LIMIT 2>&1)
if [[ $GPU2_OUT == *"satisfy Goldbach. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$GPU2_OUT"; exit 1; fi

echo -n "Test 1.4: GPU Single Checker (10^12)... "
SINGLE_OUT=$("$BIN_DIR/single_check" 1000000000000 2>&1)
if [[ $SINGLE_OUT == *"Checking Goldbach"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$SINGLE_OUT"; exit 1; fi

echo -n "Test 1.5: GMP Arbitrary Precision (10^12)... "
BIG_OUT=$("$BIN_DIR/big_check" 1000000000000 2>&1)
if [[ $BIG_OUT == *"Goldbach holds. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_OUT"; exit 1; fi

# ---------------------------------------------------------
# PART 2: CLI ROBUSTNESS (--help tests)
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 2: CLI Robustness ---${NC}"

for bin in "${required_bins[@]}"; do
    echo -n "Test 2.$bin: Help flag (--help)... "
    HELP_OUT=$("$BIN_DIR/$bin" --help 2>&1)
    if [[ $HELP_OUT == *"Usage:"* ]]; then 
        echo -e "${GREEN}PASSED${NC}"
    else 
        echo -e "${RED}FAILED${NC}"
        echo "$HELP_OUT"
        exit 1
    fi
done

# ---------------------------------------------------------
# PART 3: HARDWARE GUARDS & NEGATIVE TESTS
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 3: Hardware Guards & Negative Tests ---${NC}"

# 3.1: GPU3 VRAM Guard (Requesting a 1 Trillion segment size)
echo -n "Test 3.1: goldbach_gpu3 VRAM Guard... "
VRAM3_OUT=$("$BIN_DIR/goldbach_gpu3" 1000000 1000000000000 2>&1)
if [[ $VRAM3_OUT == *"[!] ERROR:"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$VRAM3_OUT"; exit 1; fi

# 3.2: GPU2 VRAM Guard (Requesting 10^14 limit = ~6 TB Bitset)
echo -n "Test 3.2: goldbach_gpu2 VRAM Guard... "
VRAM2_OUT=$("$BIN_DIR/goldbach_gpu2" 100000000000000 2>&1)
if [[ $VRAM2_OUT == *"[!] ERROR:"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$VRAM2_OUT"; exit 1; fi

# 3.3: CPU RAM Guard (Requesting 10^14 limit = 100 TB byte array)
# This will immediately trigger std::bad_alloc without freezing the OS
echo -n "Test 3.3: cpu_goldbach RAM Guard (bad_alloc)... "
RAM_OUT=$("$BIN_DIR/cpu_goldbach" 100000000000000 2>&1)
if [[ $RAM_OUT == *"[!] ERROR: System RAM exhausted"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$RAM_OUT"; exit 1; fi

# 3.4: single_check 64-bit Overflow (Passing a 20-digit number > uint64 max)
echo -n "Test 3.4: single_check 64-bit Overflow Guard... "
OVERFLOW_OUT=$("$BIN_DIR/single_check" 100000000000000000000 2>&1)
if [[ $OVERFLOW_OUT == *"Error: Number is too large"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$OVERFLOW_OUT"; exit 1; fi

# 3.5: big_check Odd Number Rejection
echo -n "Test 3.5: big_check Odd Number Guard... "
ODD_OUT=$("$BIN_DIR/big_check" 1000000000000000000000000000001 2>&1)
if [[ $ODD_OUT == *"Error: Goldbach's conjecture applies to even"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$ODD_OUT"; exit 1; fi

echo -e "\n===================================================="
echo -e "${GREEN}SUCCESS: All 15 verification tests passed!${NC}"
echo "===================================================="
exit 0