#!/bin/bash
# Generate videos for all single-motion types at 11 latent steps.
# Skips right-11 and up-11 (already generated).

PROMPT=''
IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/banana-pro-1769710006208_16x9.png
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p
MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors
WIDTH=768 #432
HEIGHT=432 #768
N_INFERENCE_GPU=4
NUM_FRAMES=45  # 4*11+1

OUTPUT_BASE=./outputs/butter1_all_motions

# All single motions minus the two already done
MOTIONS=(
  "right-11"
  #"up-11"
  #"left-11"
  #"down-11"
  #"rightup-11"
  #"rightdown-11"
  #"leftup-11"
  #"leftdown-11"
)

for POSE in "${MOTIONS[@]}"; do
  # e.g. "left-11" -> "left11"
  TAG="${POSE//-/}"
  OUT_DIR="${OUTPUT_BASE}/${TAG}"

  if [ -f "${OUT_DIR}/gen.mp4" ]; then
    echo "=== Skipping ${POSE} (already exists at ${OUT_DIR}/gen.mp4) ==="
    continue
  fi

  echo "==========================================="
  echo "  Generating: ${POSE}  ->  ${OUT_DIR}"
  echo "==========================================="

  torchrun --master_port=25901 --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
    --prompt "$PROMPT" \
    --image_path $IMAGE_PATH \
    --resolution $RESOLUTION \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $NUM_FRAMES \
    --seed $SEED \
    --rewrite false \
    --sr false --save_pre_sr_video \
    --pose "$POSE" \
    --output_path "$OUT_DIR" \
    --model_path $MODEL_PATH \
    --action_ckpt $AR_ACTION_MODEL_PATH \
    --few_step false \
    --width $WIDTH \
    --height $HEIGHT \
    --model_type 'ar'

  echo "=== Done: ${POSE} ==="
  echo ""
done

echo "All motions complete. Results in ${OUTPUT_BASE}/"
