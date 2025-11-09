#!/bin/bash
set -e
export HF_HUB_OFFLINE=1
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
    echo "Error: CHECKPOINT environment variable is required." >&2
    echo "Example: CHECKPOINT=/path/to/checkpoint bash eval.sh" >&2
    exit 1
fi
DATA_ROOT_DIR="${DATA_ROOT_DIR:-}"
if [ -z "${DATA_ROOT_DIR}" ]; then
    echo "Error: DATA_ROOT_DIR environment variable is required." >&2
    echo "Example: DATA_ROOT_DIR=/path/to/data bash eval.sh" >&2
    exit 1
fi
# Detect GPU count: prefer CUDA_VISIBLE_DEVICES, then nvidia-smi, then Python torch
GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _CVD_CLEAN=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | sed 's/,,*/,/g;s/^,//;s/,$//')
    if [ -n "$_CVD_CLEAN" ]; then
        GPU_COUNT=$(echo "$_CVD_CLEAN" | awk -F, '{print NF}')
    else
        GPU_COUNT=0
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
fi

if [ -z "$GPU_COUNT" ] || ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    if command -v python3 >/dev/null 2>&1; then
        GPU_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
)
        GPU_COUNT=$(echo "$GPU_COUNT" | tr -d '[:space:]')
    fi
fi

if ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    echo "Error: Unable to determine GPU count. Set CUDA_VISIBLE_DEVICES or install NVIDIA drivers."
    exit 1
fi

if [ "$GPU_COUNT" -le 0 ]; then
    echo "Error: No GPUs visible to the process."
    exit 1
fi

echo "Detected GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]; then
    echo "Warning: NPROC_PER_NODE ($NPROC_PER_NODE) > available GPUs ($GPU_COUNT)"
    echo "Auto-adjust NPROC_PER_NODE to available GPUs"
    NPROC_PER_NODE=$GPU_COUNT
fi
echo "========================"

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path $CHECKPOINT \
  --data_root_dir $DATA_ROOT_DIR/ \
  --dataset_name libero_2_task_suites_no_noops \
  --run_root_dir ./checkpoints/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --lr_warmup_steps 500 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "wangshilong" \
  --wandb_project "libero-spatial-training" \
  --use_wandb False \
  --use_tensorboard True \
  --log_freq 10 \
  --run_id_note libero_2_task_suites_no_noops \
#   --use_val_set True \
#   --val_freq 5000 \