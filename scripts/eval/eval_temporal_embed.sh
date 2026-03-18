#!/bin/bash
# Eval script: load a temporal embedding checkpoint and generate on the w_d_s_a training example.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_rl_model/diffusion_pytorch_model.safetensors
DATA_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4

# Set to the checkpoint you want to eval (e.g. checkpoint-100, checkpoint-200)
CHECKPOINT=${1:-outputs/butter4_temporal_embed/checkpoint-200}
TEMPORAL_EMBED_CKPT=$CHECKPOINT/transformer/diffusion_pytorch_model.safetensors

OUTPUT_DIR=outputs/eval_temporal_embed/$(basename $CHECKPOINT)

# First frame of the w_d_s_a GT video as the image condition
IMAGE_PATH=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4_gt/w_d_s_a.mp4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29713 scripts/eval/run_eval_temporal_embed.py \
    --model_path $MODEL_PATH \
    --action_ckpt $AR_ACTION_PATH \
    --temporal_embed_ckpt $TEMPORAL_EMBED_CKPT \
    --pose_json $DATA_DIR/w_d_s_a/pose.json \
    --image_path $IMAGE_PATH \
    --output_dir $OUTPUT_DIR \
    --num_inference_steps 30 \
    --seed 3208
