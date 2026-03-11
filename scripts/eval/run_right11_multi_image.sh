#!/bin/bash
# Generate right-11 videos for a list of input images.

PROMPT=''
SEED=1
RESOLUTION=480p
MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors
N_INFERENCE_GPU=4
NUM_FRAMES=45  # 4*11+1
POSE="down-11"

OUTPUT_BASE=./outputs/down11_multi_image

# --- Input: directory of images or list paths below ---
IMAGE_DIR=/data/ziqi/data/worldstate/predynamic/initial_frames/cropped
IMAGES=("$IMAGE_DIR"/*.png)

for IMAGE_PATH in "${IMAGES[@]}"; do
  # Derive a tag from the filename (without extension)
  BASENAME="$(basename "$IMAGE_PATH")"
  TAG="${BASENAME%.*}"
  OUT_DIR="${OUTPUT_BASE}/${TAG}"

  if [ -f "${OUT_DIR}/gen.mp4" ]; then
    echo "=== Skipping ${TAG} (already exists at ${OUT_DIR}/gen.mp4) ==="
    continue
  fi

  # Auto-detect aspect ratio from image dimensions
  read IMG_W IMG_H < <(python3 -c "from PIL import Image; im=Image.open('$IMAGE_PATH'); print(im.size[0], im.size[1])")
  if [ "$IMG_W" -ge "$IMG_H" ]; then
    ASPECT_RATIO="16:9"; WIDTH=768; HEIGHT=432
  else
    ASPECT_RATIO="9:16"; WIDTH=432; HEIGHT=768
  fi

  echo "==========================================="
  echo "  Image: ${TAG}  |  Pose: ${POSE}  |  ${ASPECT_RATIO} (${WIDTH}x${HEIGHT})  ->  ${OUT_DIR}"
  echo "==========================================="

  torchrun --master_port=25901 --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
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

  echo "=== Done: ${TAG} ==="
  echo ""
done

echo "All images complete. Results in ${OUTPUT_BASE}/"
