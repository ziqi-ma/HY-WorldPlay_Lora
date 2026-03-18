
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Allow ranks 1-3 to wait long enough for eval subprocess to finish on rank 0
# 4h timeout: two sequential eval subprocesses (seen + unseen) run at validation
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400

MODEL_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_rl_model/diffusion_pytorch_model.safetensors

NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4
UNSEEN_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4_unseen/21_a_w2_d

# Training arguments
training_args=(
  --json_path $DATA_DIR/train_all.json
  --neg_prompt_path $DATA_DIR/shared/hunyuan_neg_prompt.pt
  --neg_byt5_path $DATA_DIR/shared/hunyuan_neg_byt5_prompt.pt
  --causal
  --action
  --i2v_rate 0.2
  --train_time_shift 3.0
  --window_frames 24
  --tracker_project_name temporal_embed_per_block_all20
  --output_dir outputs/butter4_temporal_embed_per_block_all20
  --resume_from_checkpoint outputs/butter4_temporal_embed_per_block_all20/checkpoint-600
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_height 480
  --num_width 832
  --num_frames 77
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  # Per-block temporal embedding training flags
  --temporal_embed_per_block_training
  --temporal_embed_max_frames 30
  # Video generation eval every 500 steps: one seen (w_right_w_left, traj 1) + one unseen (21_a_w2_d, traj 21)
  --log_validation
  --validation_steps 600
  --eval_pose_json $DATA_DIR/w_right_w_left/pose.json
  --eval_image_path /data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4_gt/w_right_w_left.mp4
  --eval_pose_json_unseen $UNSEEN_DIR/pose.json
  --eval_image_path_unseen /data/ziqi/data/worldstate/predynamic/butter4.jpg
  --eval_action_ckpt $AR_ACTION_PATH
  --eval_gpus 0,1,2,3
)

# Parallel arguments
parallel_args=(
  --num_gpus $((NUM_GPUS * 1))
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

# Model arguments
model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir $MODEL_PATH/transformer/480p_i2v
  --ar_action_load_from_dir $AR_ACTION_PATH
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --dataloader_num_workers 0
)

# Optimizer arguments — higher LR since only embed params are trained
optimizer_args=(
  --learning_rate 1e-3
  --mixed_precision "bf16"
  --checkpointing_steps 300
  --weight_decay 0.0
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "bf16"
  --enable_gradient_checkpointing_type "full"
  --num_euler_timesteps 50
  --ema_start_step 0
)

export MASTER_PORT=29613

source /data/ziqi/miniconda3/etc/profile.d/conda.sh && conda activate worldplay

PYTHONPATH=/data/ziqi/Repos/HY-WorldPlay-TimeEmb torchrun \
        --master_port=$MASTER_PORT \
        --nproc_per_node=$NUM_GPUS \
        --nnodes 1 \
        trainer/training/ar_hunyuan_w_mem_training_pipeline.py \
        "${parallel_args[@]}" \
        "${model_args[@]}" \
        "${dataset_args[@]}" \
        "${training_args[@]}" \
        "${optimizer_args[@]}" \
        "${miscellaneous_args[@]}"
