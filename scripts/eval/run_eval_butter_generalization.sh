#!/bin/bash
# Evaluate LoRA butter checkpoint with generalization actions: left-11, up-11, down-11

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate worldplay

MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors

LORA_CKPT=/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_butter_right11/checkpoint-500/transformer/diffusion_pytorch_model.safetensors
IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/butter2_ini.jpeg

OUTPUT_BASE=/data/ziqi/Repos/HY-WorldPlay-New/outputs/eval_butter_generalization/ckpt500
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_INFERENCE_GPU=4

MOTIONS=("left-11" "up-11" "down-11")

for POSE in "${MOTIONS[@]}"; do
  TAG="${POSE//-/_}"
  OUT_DIR="${OUTPUT_BASE}/${TAG}"

  if [ -f "${OUT_DIR}/eval_gen.mp4" ]; then
    echo "=== Skipping ${POSE} (already exists at ${OUT_DIR}/eval_gen.mp4) ==="
    continue
  fi

  echo "==========================================="
  echo "  Evaluating: ${POSE}  ->  ${OUT_DIR}"
  echo "==========================================="

  torchrun --nproc_per_node=$N_INFERENCE_GPU evaluate_lora.py \
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

echo "All generalization evaluations complete. Results in ${OUTPUT_BASE}/"
