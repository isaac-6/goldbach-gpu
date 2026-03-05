#!/bin/bash

# goldbach_validation.sh
# JOSS-worthy validation suite for goldbach.cu (flagship multi-GPU Goldbach verifier).
# This script provides comprehensive testing for correctness, CLI robustness, and error handling.
# Designed for reproducibility in CI/CD pipelines or local environments.
# Assumes NVIDIA GPUs are available; skips multi-GPU tests if none detected.

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "===================================================="
echo " GoldbachGPU: Validation Suite for goldbach.cu     "
echo "===================================================="

BIN_DIR="../build/bin"
BIN_NAME="goldbach"
TEST_LIMIT=1000000000  # Small limit for quick tests
PROGRESS_LIMIT=20000000000  # Larger for progress bar to appear
LARGE_LIMIT=100000000000  # For VRAM stress

# ---------------------------------------------------------
# PRE-FLIGHT CHECK
# ---------------------------------------------------------
if [ ! -f "$BIN_DIR/$BIN_NAME" ]; then
    echo -e "${RED}[ERROR]${NC} Binary $BIN_NAME not found. Please compile first."
    exit 1
fi
echo -e "${GREEN}[INFO] Binary found.${NC}"

# Detect number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}[INFO] Detected $NUM_GPUS GPU(s).${NC}"
else
    NUM_GPUS=0
    echo -e "${YELLOW}[WARN] nvidia-smi not found. Skipping multi-GPU tests.${NC}"
fi

# ---------------------------------------------------------
# PART 1: CORE FUNCTIONALITY (Positive Tests)
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 1: Core Functionality ---${NC}"

echo -n "Test 1.1: Basic Run ($TEST_LIMIT)... "
BASIC_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --seg-size=1000000 --p-small=100000 --batch-size=100000 2>&1)
if [[ $BASIC_OUT == *"satisfy Goldbach. ✓"* && $BASIC_OUT == *"Phase 2 fallbacks      : 0"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"; 
else 
    echo -e "${RED}FAILED${NC}"; echo "$BASIC_OUT"; exit 1; 
fi

echo -n "Test 1.2: With Progress Flag ($PROGRESS_LIMIT)... "
PROGRESS_OUT=$("$BIN_DIR/$BIN_NAME" $PROGRESS_LIMIT --seg-size=10000000 --p-small=1000000 --batch-size=1000000 --progress 2>&1)
if [[ $PROGRESS_OUT == *"[Progress]"* && $PROGRESS_OUT == *"satisfy Goldbach. ✓"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"; 
else 
    echo -e "${RED}FAILED${NC}"; echo "$PROGRESS_OUT"; exit 1; 
fi

echo -n "Test 1.3: Custom Start ($TEST_LIMIT, start=1000000)... "
START_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --seg-size=1000000 --p-small=100000 --batch-size=100000 --start=1000000 2>&1)
if [[ $START_OUT == *"Checking range : [1000000"* && $START_OUT == *"satisfy Goldbach. ✓"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"; 
else 
    echo -e "${RED}FAILED${NC}"; echo "$START_OUT"; exit 1; 
fi

if [ $NUM_GPUS -gt 1 ]; then
    echo -n "Test 1.4: Multi-GPU ($TEST_LIMIT, --gpus=2)... "
    MULTI_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --seg-size=1000000 --p-small=100000 --batch-size=100000 --gpus=2 2>&1)
    if [[ $MULTI_OUT == *"satisfy Goldbach. ✓"* ]]; then 
        echo -e "${GREEN}PASSED${NC}"; 
    else 
        echo -e "${RED}FAILED${NC}"; echo "$MULTI_OUT"; exit 1; 
    fi
else
    echo -e "${YELLOW}[SKIP] Test 1.4: Multi-GPU (requires >1 GPU).${NC}"
fi

# ---------------------------------------------------------
# PART 2: CLI ROBUSTNESS
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 2: CLI Robustness ---${NC}"

echo -n "Test 2.1: Help Flag (--help)... "
HELP_OUT=$("$BIN_DIR/$BIN_NAME" --help 2>&1)
if [[ $HELP_OUT == *"Usage:"* && $HELP_OUT == *"--seg-size=N"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$HELP_OUT"; exit 1
fi

echo -n "Test 2.2: Invalid Option (--invalid)... "
INVALID_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --invalid 2>&1)
if [[ $INVALID_OUT == *"Error: Invalid"* || $INVALID_OUT == *"unrecognized option"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$INVALID_OUT"; exit 1
fi

echo -n "Test 2.3: Missing Required Argument (No LIMIT)... "
NO_LIMIT_OUT=$("$BIN_DIR/$BIN_NAME" --seg-size=1000000 2>&1)
if [[ $NO_LIMIT_OUT == *"Usage:"* || $NO_LIMIT_OUT == *"Error: "* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$NO_LIMIT_OUT"; exit 1
fi

# ---------------------------------------------------------
# PART 3: ERROR HANDLING & NEGATIVE TESTS
# ---------------------------------------------------------
echo -e "\n${CYAN}--- Part 3: Error Handling & Negative Tests ---${NC}"

echo -n "Test 3.1: LIMIT < 4... "
LOW_LIMIT_OUT=$("$BIN_DIR/$BIN_NAME" 2 --seg-size=1000000 2>&1)
if [[ $LOW_LIMIT_OUT == *"Error: LIMIT must be >= 4"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$LOW_LIMIT_OUT"; exit 1
fi

echo -n "Test 3.2: Odd SEG_SIZE... "
ODD_SEG_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --seg-size=1000001 2>&1)
if [[ $ODD_SEG_OUT == *"Error: SEG_SIZE must be even"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$ODD_SEG_OUT"; exit 1
fi

echo -n "Test 3.3: P_SMALL > MAX (5e9 > 4e9)... "
HIGH_P_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --p-small=5000000000 2>&1)
if [[ $HIGH_P_OUT == *"Error: P_SMALL must be <= 4000000000"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$HIGH_P_OUT"; exit 1
fi

echo -n "Test 3.4: START > LIMIT... "
HIGH_START_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --start=$((TEST_LIMIT + 2)) 2>&1)
if [[ $HIGH_START_OUT == *"Error: START must be less than LIMIT"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$HIGH_START_OUT"; exit 1
fi

echo -n "Test 3.5: VRAM Guard (Excessive SEG_SIZE on $LARGE_LIMIT)... "
VRAM_OUT=$("$BIN_DIR/$BIN_NAME" $LARGE_LIMIT --seg-size=100000000000 2>&1)  # Intentionally huge
if [[ $VRAM_OUT == *"[!] ERROR: GPU"* || $VRAM_OUT == *"insufficient VRAM"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$VRAM_OUT"; exit 1
fi

if [ $NUM_GPUS -gt 0 ]; then
    echo -n "Test 3.6: More GPUs than Available (--gpus=$((NUM_GPUS + 1)))... "
    EXTRA_GPU_OUT=$("$BIN_DIR/$BIN_NAME" $TEST_LIMIT --gpus=$((NUM_GPUS + 1)) 2>&1)

    if [[ $EXTRA_GPU_OUT == *"Requested"* && $EXTRA_GPU_OUT == *"available"* && $EXTRA_GPU_OUT == *"Using"* ]]; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "$EXTRA_GPU_OUT"
        exit 1
    fi
else
    echo -e "${YELLOW}[SKIP] Test 3.6: GPU Count Guard (no GPUs detected).${NC}"
fi

echo -n "Test 3.7: Invalid Numeric Argument (Non-integer LIMIT)... "
INVALID_NUM_OUT=$("$BIN_DIR/$BIN_NAME" abc --seg-size=1000000 2>&1)
if [[ $INVALID_NUM_OUT == *"Error: Invalid numeric argument"* ]]; then 
    echo -e "${GREEN}PASSED${NC}"
else 
    echo -e "${RED}FAILED${NC}"; echo "$INVALID_NUM_OUT"; exit 1
fi

echo -e "\n===================================================="
echo -e "${GREEN}SUCCESS: All validation tests passed!${NC}"
echo "===================================================="
exit 0