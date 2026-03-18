#!/bin/bash
# Evaluate checkpoint-2400 on 3 seen + 3 unseen trajectories using GPUs 4-7.

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29715

MODEL_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_PATH=/data/ziqi/checkpoints/hy-worldplay/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_rl_model/diffusion_pytorch_model.safetensors
DATA_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4
GT_DIR=/data/ziqi/Repos/HY-WorldPlay-New/training_data/butter4_gt
POSES_JSON=/data/ziqi/Repos/HY-WorldPlay-New/outputs/butter4_rlar/poses.json
UNSEEN_IMAGE=/data/ziqi/data/worldstate/predynamic/butter4.jpg

CKPT=outputs/butter4_temporal_embed_per_block_all20/checkpoint-2400
TEMPORAL_EMBED_CKPT=$CKPT/transformer/diffusion_pytorch_model.safetensors
OUT_BASE=outputs/eval_per_block_ckpt2400

source /data/ziqi/miniconda3/etc/profile.d/conda.sh && conda activate worldplay

# ── Generate pose.jsons for unseen trajectories ───────────────────────────────
echo "Generating pose.jsons for unseen trajectories..."
PYTHONPATH=/data/ziqi/Repos/HY-WorldPlay-TimeEmb python3 - <<'PYEOF'
import json, sys, os
sys.path.insert(0, '/data/ziqi/Repos/HY-WorldPlay-New')
from hyvideo.generate import parse_pose_string
from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
import numpy as np

poses_data = json.load(open('/data/ziqi/Repos/HY-WorldPlay-New/outputs/butter4_rlar/poses.json'))
poses = poses_data['poses']

raw_intrinsic = np.array([
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
])
norm_intrinsic = raw_intrinsic.copy()
norm_intrinsic[0, 0] /= raw_intrinsic[0, 2] * 2
norm_intrinsic[1, 1] /= raw_intrinsic[1, 2] * 2
norm_intrinsic[0, 2] = 0.5
norm_intrinsic[1, 2] = 0.5

for key in ['22_w_left_s_right', '25_right_down_w_up', '30_a2_w_d']:
    pose_string = poses[key]
    motions = parse_pose_string(pose_string)
    c2w_list = generate_camera_trajectory_local(motions)
    pose_dict = {}
    for i, c2w in enumerate(c2w_list):
        w2c = np.linalg.inv(c2w)
        pose_dict[str(i)] = {'w2c': w2c.tolist(), 'intrinsic': norm_intrinsic.tolist()}
    out_dir = f'/tmp/unseen_poses/{key}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/pose.json', 'w') as f:
        json.dump(pose_dict, f, indent=2)
    print(f'Generated {key}: {len(pose_dict)} frames')
PYEOF

# ── Helper: run one eval ───────────────────────────────────────────────────────
run_eval() {
    local name=$1
    local pose_json=$2
    local image_path=$3
    local out_dir=$OUT_BASE/$name

    echo ""
    echo "=== Evaluating: $name ==="
    PYTHONPATH=/data/ziqi/Repos/HY-WorldPlay-TimeEmb torchrun \
        --master_port=$MASTER_PORT \
        --nproc_per_node=4 \
        --nnodes=1 \
        scripts/eval/run_eval_temporal_embed.py \
        --model_path $MODEL_PATH \
        --action_ckpt $AR_ACTION_PATH \
        --temporal_embed_ckpt $TEMPORAL_EMBED_CKPT \
        --pose_json $pose_json \
        --image_path $image_path \
        --output_dir $out_dir \
        --num_inference_steps 30 \
        --seed 3208
}

# ── 3 seen trajectories ───────────────────────────────────────────────────────
run_eval "seen_w_right_w_left"    $DATA_DIR/w_right_w_left/pose.json    $GT_DIR/w_right_w_left.mp4
run_eval "seen_a_s_d_w"           $DATA_DIR/a_s_d_w/pose.json           $GT_DIR/a_s_d_w.mp4
run_eval "seen_d2_w_a"            $DATA_DIR/d2_w_a/pose.json            $GT_DIR/d2_w_a.mp4

# ── 3 unseen trajectories ─────────────────────────────────────────────────────
run_eval "unseen_22_w_left_s_right"   /tmp/unseen_poses/22_w_left_s_right/pose.json   $UNSEEN_IMAGE
run_eval "unseen_25_right_down_w_up"  /tmp/unseen_poses/25_right_down_w_up/pose.json  $UNSEEN_IMAGE
run_eval "unseen_30_a2_w_d"           /tmp/unseen_poses/30_a2_w_d/pose.json           $UNSEEN_IMAGE

echo ""
echo "All evals done. Results in $OUT_BASE/"
