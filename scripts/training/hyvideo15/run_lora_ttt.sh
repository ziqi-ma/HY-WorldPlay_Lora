#!/bin/bash
# LoRA finetuning of HunyuanVideo 1.5 normal AR model
#
# Supports training on multiple videos. Just list mp4 paths and pose strings
# in the arrays below — preprocessing is handled automatically (and cached).

set -e

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false

# Model paths
MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_FILE=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors
BASE_TRANSFORMER_DIR=${MODEL_PATH}/transformer/480p_i2v

# ─── Video list ──────────────────────────────────────────────────────────────
# Add one entry per video. Arrays must be the same length.
# PROMPTS are optional — leave "" for unconditional.
VIDEO_PATHS=(
  #/data/ziqi/Repos/HY-WorldPlay/outputs/butter2_a2d2_ar/veo_output_to0047_save.mp4
  #/data/ziqi/Repos/HY-WorldPlay-New/outputs/butter_all_motions/right11/veo_butter_noisy1.mp4
  /data/ziqi/Repos/HY-WorldPlay-New/outputs/butter_all_motions/right11/veo_butter_noisy2.mp4
  #/data/ziqi/Repos/HY-WorldPlay-New/outputs/butter_all_motions/right11/veo_butter_noisy3.mp4
  #/data/ziqi/Repos/HY-WorldPlay-New/outputs/butter_all_motions/right11/veo_butter_noisy4.mp4
)
POSE_STRINGS=(
  "right-11"
  #"right-11"
  #"right-11"
  #"right-11"
  #"right-11"
)
PROMPTS=(
  ""
  #""
  #""
  #""
  #""
)

# Eval settings (uses the first video by default)
EVAL_IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/butter2_ini.jpeg
EVAL_ACTION=right-11

OUTPUT_DIR=/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_butter_noisy2_right11_lr1e3/
PREPROCESS_DIR=${OUTPUT_DIR}/preprocessed

NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ─── Preprocessing ──────────────────────────────────────────────────────────
NUM_VIDEOS=${#VIDEO_PATHS[@]}
echo "=== Preprocessing ${NUM_VIDEOS} video(s) ==="

PREPROCESS_DIRS=()
for i in $(seq 0 $((NUM_VIDEOS - 1))); do
  VIDEO=${VIDEO_PATHS[$i]}
  POSE=${POSE_STRINGS[$i]}
  PROMPT=${PROMPTS[$i]:-""}

  # Derive a unique subdir name from the video filename (without extension)
  VIDEO_NAME=$(basename "${VIDEO}" .mp4)
  DATA_DIR=${PREPROCESS_DIR}/${VIDEO_NAME}
  PREPROCESS_DIRS+=("${DATA_DIR}")

  if [ -f "${DATA_DIR}/latents.pt" ]; then
    echo "  [${i}] Cached: ${DATA_DIR} (skipping)"
  else
    echo "  [${i}] Preprocessing: ${VIDEO}"
    python preprocess_video.py \
      --video_path "${VIDEO}" \
      --pose_string "${POSE}" \
      --prompt "${PROMPT}" \
      --output_dir "${DATA_DIR}" \
      --model_path "${MODEL_PATH}"
  fi
done

# ─── Merge metadata.json from all preprocessed dirs ─────────────────────────
MERGED_METADATA=${OUTPUT_DIR}/merged_metadata.json
python3 -c "
import json, sys, os
dirs = sys.argv[1:]
merged = []
for d in dirs:
    meta_path = os.path.join(d, 'metadata.json')
    if not os.path.exists(meta_path):
        print(f'WARNING: {meta_path} not found, skipping', file=sys.stderr)
        continue
    with open(meta_path) as f:
        entries = json.load(f)
    merged.extend(entries)
    print(f'  Added {len(entries)} entry from {d}')
print(f'Total training examples: {len(merged)}')
with open('${MERGED_METADATA}', 'w') as f:
    json.dump(merged, f, indent=2)
" "${PREPROCESS_DIRS[@]}"

FIRST_DATA_DIR=${PREPROCESS_DIRS[0]}
echo "=== Starting training with ${NUM_VIDEOS} video(s) ==="

# Training arguments
training_args=(
  --data_path ${FIRST_DATA_DIR}
  --json_path ${MERGED_METADATA}
  --causal
  --action
  --i2v_rate 0.0
  --train_time_shift 3.0
  --window_frames 12
  --output_dir ${OUTPUT_DIR}
  --max_train_steps 350
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 12
  --num_height 432
  --num_width 768
  --num_frames 45
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --training_cfg_rate 0.0
)

# Parallel arguments
parallel_args=(
  --num_gpus $((NUM_GPUS * 1))
  --sp_size 2
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir ${BASE_TRANSFORMER_DIR}
  --ar_action_load_from_dir ${AR_ACTION_MODEL_FILE}
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --dataloader_num_workers 0
)

validation_args=(
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
  --eval_steps 50
  --eval_num_inference_steps 50
  --gt_frames_dir ${FIRST_DATA_DIR}/frames/
  --eval_pose_string ${EVAL_ACTION}
  --eval_prompt ""
  --eval_image_path ${EVAL_IMAGE_PATH}
)

# Optimizer arguments (higher LR for LoRA)
optimizer_args=(
  --learning_rate 3e-4
  --mixed_precision "bf16"
  --checkpointing_steps 50
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# LoRA arguments
lora_args=(
  --lora_training
  --lora_rank 16
  --lora_alpha 32
  --lora_target_modules img_attn_q img_attn_k img_attn_v img_attn_proj img_attn_prope_proj txt_attn_q txt_attn_k txt_attn_v txt_attn_proj linear1_q linear1_k linear1_v
)

# Miscellaneous arguments
miscellaneous_args=(
  --mode finetuning
  --inference_mode False
  --checkpoints_total_limit 5
  --not_apply_cfg_solver
  --num_euler_timesteps 50
  --ema_start_step 0
  --tracker_project_name "lora_block"
  --wandb_run_name "block_lr1e3_rank16"
)

export MASTER_PORT=29611

torchrun \
        --master_port=$MASTER_PORT \
        --nproc_per_node=$NUM_GPUS \
        --nnodes 1 \
        trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${validation_args[@]}" \
        "${lora_args[@]}" \
        "${miscellaneous_args[@]}"
