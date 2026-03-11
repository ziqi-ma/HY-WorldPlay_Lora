export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"


PROMPT=''
IMAGE_PATH=/data/ziqi/data/worldstate/predynamic/tank_filling_up.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=9:16
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./outputs/
MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/bidirectional_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_distilled_action_model/diffusion_pytorch_model.safetensors
POSE='up-11'                # Camera trajectory: pose string (e.g., 'w-3, right-0.5') or JSON file path
NUM_FRAMES=45 # this should be 4*pose+1, %16=13
WIDTH=432
HEIGHT=768

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=4 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_ACTION_MODEL_PATH \
  --few_step false \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar'

# inference with autoregressive distilled model
#torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  # --prompt "$PROMPT" \
  # --image_path $IMAGE_PATH \
  # --resolution $RESOLUTION \
  # --aspect_ratio $ASPECT_RATIO \
  # --video_length $NUM_FRAMES \
  # --seed $SEED \
  # --rewrite $REWRITE \
  # --sr $ENABLE_SR --save_pre_sr_video \
  # --pose "$POSE" \
  # --output_path $OUTPUT_PATH \
  # --model_path $MODEL_PATH \
  # --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  # --few_step true \
  # --num_inference_steps 4 \
  # --model_type 'ar' \
  # --use_vae_parallel false \
  # --use_sageattn false \
  # --use_fp8_gemm false \
