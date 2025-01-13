#!/bin/bash

# Colors for pretty printing
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Base directories
PROJ_ROOT="/data/convert"
TOOL_ROOT="${PROJ_ROOT}/perf_tool"
PROJECT_DIR="${PROJ_ROOT}/AirConditioner_msft"

# Add tool to Python path
source set_python_path.sh

# Change to project directory
cd ${PROJECT_DIR}

# Script and config paths (using relative paths)
TARGET_SCRIPT="item_inference.py"
OPS_CONFIG="../perf_tool/configs/air_conditioner_ops.yaml"
PRECISION_CONFIG="../perf_tool/configs/precision_config.json"

# Default settings
LOG_LEVEL="debug"
NUM_RUNS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Results directory
RESULT_ROOT="../results/air_conditioner"
ERROR_LOG="${RESULT_ROOT}/error.log"

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}Error: File not found: $1${NC}"
        exit 1
    fi
}

# Function to create directory if not exists
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# Function to log error
log_error() {
    local error_msg="$1"
    local error_dir=$(dirname "${ERROR_LOG}")
    ensure_dir "${error_dir}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $error_msg" >> "${ERROR_LOG}"
}

# Validate paths
echo -e "${BLUE}Validating paths...${NC}"
check_file "${TARGET_SCRIPT}"
check_file "${OPS_CONFIG}"
check_file "${PRECISION_CONFIG}"

# Create results directory
ensure_dir "${RESULT_ROOT}"

# Print configuration
echo -e "\n${GREEN}Air Conditioner Performance Monitoring${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "- Project Directory: ${PROJECT_DIR}"
echo -e "- Target Script: ${TARGET_SCRIPT}"
echo -e "- Operations Config: ${OPS_CONFIG}"
echo -e "- Precision Config: ${PRECISION_CONFIG}"
echo -e "- Log Level: ${LOG_LEVEL}"
echo -e "- Number of Runs: ${NUM_RUNS}"
echo -e "- Results Directory: ${RESULT_ROOT}"
echo -e "- Error Log: ${ERROR_LOG}\n"

# Run monitoring using perf_tool command
echo -e "${GREEN}Starting monitoring...${NC}"
perf_tool \
    --script ${TARGET_SCRIPT} \
    --config ${OPS_CONFIG} \
    --precision ${PRECISION_CONFIG} \
    --output ${RESULT_ROOT} \
    --log-level ${LOG_LEVEL} \
    --runs ${NUM_RUNS} 2>> "${ERROR_LOG}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "\n${GREEN}Monitoring completed successfully!${NC}"
    echo -e "Results are saved in: ${BLUE}${RESULT_ROOT}${NC}"
    
    # Show the results structure
    echo -e "\n${BLUE}Results directory structure:${NC}"
    tree ${RESULT_ROOT}
    
    # Show time statistics if available
    TIME_STATS="${RESULT_ROOT}/time/stats.csv"
    if [ -f "${TIME_STATS}" ]; then
        echo -e "\n${BLUE}Time Statistics Summary:${NC}"
        echo "----------------------------------------"
        head -n 5 "${TIME_STATS}"
        echo "..."
    fi
    
    # Show precision statistics if available
    PRECISION_STATS="${RESULT_ROOT}/precision/summary.json"
    if [ -f "${PRECISION_STATS}" ]; then
        echo -e "\n${BLUE}Precision Monitoring Summary:${NC}"
        echo "----------------------------------------"
        cat "${PRECISION_STATS}" | head -n 10
        echo "..."
    fi
else
    echo -e "\n${RED}Monitoring failed with errors.${NC}"
    # Find the most recent log file
    LATEST_LOG=$(find ${RESULT_ROOT} -name "perf_monitor.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
    if [ ! -z "${LATEST_LOG}" ]; then
        echo -e "Check log file at: ${BLUE}${LATEST_LOG}${NC}"
        echo -e "\n${RED}Last few lines of log:${NC}"
        tail -n 10 "${LATEST_LOG}" | tee -a "${ERROR_LOG}"
        
        # Add error summary to error log
        log_error "Monitoring failed with exit code ${EXIT_CODE}"
        log_error "Last few lines of perf_monitor.log:"
        tail -n 10 "${LATEST_LOG}" >> "${ERROR_LOG}"
        
        echo -e "\n${YELLOW}Full error log available at: ${ERROR_LOG}${NC}"
    fi
    exit 1
fi 