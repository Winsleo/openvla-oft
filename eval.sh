#!/bin/bash
set -e
export HF_HUB_OFFLINE=1

CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
    echo "Error: CHECKPOINT environment variable is required." >&2
    echo "Example: CHECKPOINT=/path/to/checkpoint bash eval.sh" >&2
    exit 1
fi

LOG_ROOT=./logs/libero_eval/
mkdir -p "${LOG_ROOT}"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_spatial \
  --local_log_dir "${LOG_ROOT}/spatial" \
  --run_id_note spatial &
PID_SPATIAL=$!

CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_object \
  --local_log_dir "${LOG_ROOT}/object" \
  --run_id_note object &
PID_OBJECT=$!

wait ${PID_SPATIAL}
wait ${PID_OBJECT}