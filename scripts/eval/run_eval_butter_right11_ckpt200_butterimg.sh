#!/bin/bash
# Evaluate LoRA butter_right11 checkpoint-200 on cropped butter image (16:9) with right-11 motion

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate worldplay

MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors

LORA_CKPT=/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_butter_right11/checkpoint-200/transformer/diffusion_pytorch_model.safetensors
IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/banana-pro-1769710006208_16x9.png

OUTPUT_DIR=/data/ziqi/Repos/HY-WorldPlay-New/outputs/eval_butter_right11_ckpt200_butterimg
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_INFERENCE_GPU=4

echo "==========================================="
echo "  Evaluating: right-11 on butter image (16:9)"
echo "  Checkpoint: lora_butter_right11/checkpoint-200"
echo "  Resolution: 432x768 (16:9 landscape)"
echo "  Output: ${OUTPUT_DIR}"
echo "==========================================="

torchrun --nproc_per_node=$N_INFERENCE_GPU evaluate_lora.py \
  --lora_path $LORA_CKPT \
  --image_path $IMAGE_PATH \
  --pose_string "right-11" \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_ACTION_MODEL_PATH \
  --output_path "$OUTPUT_DIR" \
  --lora_rank 16 \
  --lora_alpha 32 \
  --height 432 \
  --width 768

echo "=== Done ==="
