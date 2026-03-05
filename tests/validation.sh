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
TEST_LIMIT=1000000  # Small for quick tests

# ---------------------------------------------------------
# PRE-FLIGHT CHECK
# ---------------------------------------------------------
required_bins=("cpu_goldbach" "big_check" "single_check")
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

echo -n "Test 1.2: GPU Single Checker (10^12)... "
SINGLE_OUT=$("$BIN_DIR/single_check" 1000000000000 2>&1)
if [[ $SINGLE_OUT == *"Goldbach holds. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$SINGLE_OUT"; exit 1; fi

echo -n "Test 1.3: GMP Arbitrary Precision (10^12)... "
BIG_OUT=$("$BIN_DIR/big_check" 1000000000000 2>&1)
if [[ $BIG_OUT == *"Goldbach holds. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_OUT"; exit 1; fi

echo -n "Test 1.4: GMP Arbitrary Precision (Large n, 10^50)... "
BIG_LARGE_OUT=$("$BIN_DIR/big_check" 10000000000000000000000000000000000000000000000000 2>&1)
if [[ $BIG_LARGE_OUT == *"Goldbach holds. ✓"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_LARGE_OUT"; exit 1; fi

# ---------------------------------------------------------
# PART 2: CLI ROBUSTNESS (--help tests)
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 2: CLI Robustness ---${NC}"

for bin in "${required_bins[@]}"; do
    echo -n "Test 2.1 ($bin): Help flag (--help)... "
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

echo -n "Test 3.1: cpu_goldbach RAM Guard (bad_alloc, 10^14)... "
RAM_OUT=$("$BIN_DIR/cpu_goldbach" 100000000000000 2>&1)
if [[ $RAM_OUT == *"[!] ERROR: System RAM exhausted"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$RAM_OUT"; exit 1; fi

echo -n "Test 3.2: cpu_goldbach Invalid Numeric (non-number)... "
CPU_INVALID_OUT=$("$BIN_DIR/cpu_goldbach" abc 2>&1)
if [[ $CPU_INVALID_OUT == *"Error: Invalid numeric argument"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$CPU_INVALID_OUT"; exit 1; fi

echo -n "Test 3.3: cpu_goldbach LIMIT < 4... "
CPU_LOW_OUT=$("$BIN_DIR/cpu_goldbach" 2 2>&1)
if [[ $CPU_LOW_OUT == *"Error: LIMIT must be >= 4"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$CPU_LOW_OUT"; exit 1; fi

echo -n "Test 3.4: single_check 64-bit Overflow (20 digits > uint64 max)... "
OVERFLOW_OUT=$("$BIN_DIR/single_check" 100000000000000000000 2>&1)
if [[ $OVERFLOW_OUT == *"Error: Number is too large"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$OVERFLOW_OUT"; exit 1; fi

echo -n "Test 3.5: single_check Odd Number... "
SINGLE_ODD_OUT=$("$BIN_DIR/single_check" 1000000000001 2>&1)
if [[ $SINGLE_ODD_OUT == *"Error: n must be an even integer"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$SINGLE_ODD_OUT"; exit 1; fi

echo -n "Test 3.6: single_check n < 4... "
SINGLE_LOW_OUT=$("$BIN_DIR/single_check" 2 2>&1)
if [[ $SINGLE_LOW_OUT == *"Error: n must be >= 4"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$SINGLE_LOW_OUT"; exit 1; fi

echo -n "Test 3.7: big_check Odd Number... "
BIG_ODD_OUT=$("$BIN_DIR/big_check" 1000000000000000000000000000001 2>&1)
if [[ $BIG_ODD_OUT == *"Error: Goldbach's conjecture applies to even"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_ODD_OUT"; exit 1; fi

echo -n "Test 3.8: big_check Non-Numeric Input... "
BIG_INVALID_OUT=$("$BIN_DIR/big_check" abc 2>&1)
if [[ $BIG_INVALID_OUT == *"Error: Input must be a positive integer"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_INVALID_OUT"; exit 1; fi

echo -n "Test 3.9: big_check n < 4... "
BIG_LOW_OUT=$("$BIN_DIR/big_check" 2 2>&1)
if [[ $BIG_LOW_OUT == *"Error: invalid number or not even / >= 4"* ]]; then echo -e "${GREEN}PASSED${NC}"; else echo -e "${RED}FAILED${NC}"; echo "$BIG_LOW_OUT"; exit 1; fi

echo -e "\n===================================================="
echo -e "${GREEN}SUCCESS: All 12 verification tests passed!${NC}"
echo "===================================================="
exit 0