#!/bin/bash
# Evaluate LoRA-finetuned model against training GT frames

MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors

TRAINING_DATA_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/waterfill_right11
LORA_CKPT=/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_waterfill_right11/checkpoint-500/transformer/diffusion_pytorch_model.safetensors

N_INFERENCE_GPU=8

# Eval with LoRA
torchrun --nproc_per_node=$N_INFERENCE_GPU evaluate_lora.py \
    --lora_path $LORA_CKPT \
    --gt_frames_dir ${TRAINING_DATA_DIR}/frames/ \
    --image_path ${TRAINING_DATA_DIR}/frames/0000.png \
    --pose_string "right-11" \
    --model_path $MODEL_PATH \
    --action_ckpt $AR_ACTION_MODEL_PATH \
    --output_path outputs/eval_lora_waterfill/ \
    --lora_rank 16 \
    --lora_alpha 32

# Eval without LoRA (baseline)
# torchrun --nproc_per_node=$N_INFERENCE_GPU evaluate_lora.py \
#     --no_lora \
#     --lora_path dummy \
#     --gt_frames_dir ${TRAINING_DATA_DIR}/frames/ \
#     --image_path ${TRAINING_DATA_DIR}/frames/0000.png \
#     --pose_string "right-11" \
#     --model_path $MODEL_PATH \
#     --action_ckpt $AR_ACTION_MODEL_PATH \
#     --output_path outputs/eval_baseline_waterfill/
