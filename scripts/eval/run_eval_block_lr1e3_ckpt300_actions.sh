#!/bin/bash
# Evaluate LoRA block_right11_lr1e3 checkpoint-300 on block_on_ramp with left-11, down-11, up-11

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate worldplay

MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors

LORA_CKPT=/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_block_right11_lr1e3/checkpoint-300/transformer/diffusion_pytorch_model.safetensors
IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/block_on_ramp.png

OUTPUT_BASE=/data/ziqi/Repos/HY-WorldPlay-New/outputs/eval_block_lr1e3_ckpt300_actions
export CUDA_VISIBLE_DEVICES=4,5,6,7
N_INFERENCE_GPU=4

ACTIONS=("left-11" "down-11" "up-11")
TAGS=("left11" "down11" "up11")

for i in "${!ACTIONS[@]}"; do
  POSE="${ACTIONS[$i]}"
  TAG="${TAGS[$i]}"
  OUT_DIR="${OUTPUT_BASE}/${TAG}"

  if [ -f "${OUT_DIR}/eval_gen.mp4" ]; then
    echo "=== Skipping ${POSE} (already exists at ${OUT_DIR}/eval_gen.mp4) ==="
    continue
  fi

  echo "==========================================="
  echo "  Evaluating: ${POSE}"
  echo "  Checkpoint: lora_block_right11_lr1e3/checkpoint-300"
  echo "  Image: block_on_ramp.png (16:9)"
  echo "  Output: ${OUT_DIR}"
  echo "==========================================="

  torchrun --master_port=29511 --nproc_per_node=$N_INFERENCE_GPU evaluate_lora.py \
    --lora_path $LORA_CKPT \
    --image_path $IMAGE_PATH \
    --pose_string "$POSE" \
    --model_path $MODEL_PATH \
    --action_ckpt $AR_ACTION_MODEL_PATH \
    --output_path "$OUT_DIR" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --height 432 \
    --width 768

  echo "=== Done: ${POSE} ==="
  echo ""
done

echo "All evaluations complete. Results in ${OUTPUT_BASE}/"
