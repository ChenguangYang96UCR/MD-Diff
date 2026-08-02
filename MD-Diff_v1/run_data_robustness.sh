#!/usr/bin/env bash
set -euo pipefail

# Run from the MG-Diff_v1 directory, with test_data_robustness.py beside test.py.
# If Conda is not initialized in non-interactive shells, activate the
# pytorch-py39 environment before launching this script.

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-2}"
DATASET="${DATASET:-GZAir}"
CONFIG="${CONFIG:-config_GZAir}"
PRED_ATTR="${PRED_ATTR:-PM25}"
FILE_TIME="${FILE_TIME:-20260726T104228}"
EPOCH="${EPOCH:-best}"
T_LEN="${T_LEN:-24}"
TEST_SEED="${TEST_SEED:-2030}"
CORRUPTION_SEED="${CORRUPTION_SEED:-42}"
MORSE_SEED="${MORSE_SEED:-42}"
MORSE_SCORE="${MORSE_SCORE:-degree}"
MORSE_CONTROL="${MORSE_CONTROL:-morse}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/data_robustness_${DATASET}_${FILE_TIME}}"
RESULT_CSV="${RESULT_CSV:-${OUTPUT_ROOT}/metrics.csv}"

mkdir -p "${OUTPUT_ROOT}"

COMMON_ARGS=(
  --model morsediffusionfore
  --dataset_mode "${DATASET}"
  --pred_attr "${PRED_ATTR}"
  --gpu_ids "${GPU_ID}"
  --config "${CONFIG}"
  --t_len "${T_LEN}"
  --file_time "${FILE_TIME}"
  --epoch "${EPOCH}"
  --seed "${TEST_SEED}"
  --morse_control "${MORSE_CONTROL}"
  --morse_score "${MORSE_SCORE}"
  --morse_seed "${MORSE_SEED}"
  --corruption_seed "${CORRUPTION_SEED}"
  --robustness_csv "${RESULT_CSV}"
)

run_test() {
  local run_name="$1"
  shift
  local log_file="${OUTPUT_ROOT}/${run_name}.log"

  echo
  echo "================================================================"
  echo "Running: ${run_name}"
  echo "Log:     ${log_file}"
  echo "================================================================"

  "${PYTHON_BIN}" test_data_robustness.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_file}"
}

# 1. Clean test set.
run_test "clean" \
  --corruption none

# 2. Randomly remove observed historical events.
for rate in 0.05 0.10 0.20; do
  rate_label="${rate/./p}"
  run_test "missing_events_${rate_label}" \
    --corruption missing_events \
    --corruption_rate "${rate}"
done

# 3. Round/aggregate regularly sampled timestamps into coarser bins.
# For hourly GZAir/BJAir, these correspond to 2-, 3-, and 6-hour bins.
for steps in 2 3 6; do
  run_test "timestamp_rounding_${steps}_steps" \
    --corruption timestamp_rounding \
    --round_steps "${steps}"
done

# 4. Assign a fraction of stations incorrect spatial identities.
for rate in 0.05 0.10 0.20; do
  rate_label="${rate/./p}"
  run_test "geocoding_error_${rate_label}" \
    --corruption geocoding_error \
    --corruption_rate "${rate}"
done

echo
echo "All robustness tests completed."
echo "Metrics: ${RESULT_CSV}"
echo "Logs:    ${OUTPUT_ROOT}"
