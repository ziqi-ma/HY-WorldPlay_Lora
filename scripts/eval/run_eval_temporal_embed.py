"""
Eval script: load a temporal embedding checkpoint and run inference on the w_d_s_a trajectory.
Usage:
    python scripts/eval/run_eval_temporal_embed.py \
        --model_path <model_path> \
        --action_ckpt <ar_action_safetensors> \
        --temporal_embed_ckpt <checkpoint_dir/transformer/diffusion_pytorch_model.safetensors> \
        --pose_json <butter4/w_d_s_a/pose.json> \
        --image_path <butter4_gt/w_d_s_a.mp4> \
        --output_dir outputs/eval_temporal_embed/checkpoint-xxx
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import numpy as np
import torch
import cv2
import imageio
import einops
from PIL import Image
from safetensors.torch import load_file

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state
from scipy.spatial.transform import Rotation as R


# ── helpers ──────────────────────────────────────────────────────────────────

mapping = {
    (0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3, (0, 0, 0, 1): 4, (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6, (0, 1, 1, 0): 7, (0, 1, 0, 1): 8,
}

def one_hot_to_label(one_hot):
    return torch.tensor([mapping[tuple(r.tolist())] for r in one_hot])

def pose_json_to_inputs(pose_json_path):
    """
    Convert training-format pose.json (keys: 'w2c', 'intrinsic') to
    (w2c_tensor, intrinsic_tensor, action_tensor) suitable for the pipeline.
    """
    pose_json = json.load(open(pose_json_path))
    keys = list(pose_json.keys())
    n = len(keys)

    w2c_list, intrinsic_list = [], []
    for k in keys:
        w2c = np.array(pose_json[k]['w2c'])
        intrinsic = np.array(pose_json[k]['intrinsic'])
        # normalize intrinsic (same as dataset code)
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        w2c_list.append(w2c)
        intrinsic_list.append(intrinsic)

    # camera-center normalization (align first camera to origin)
    w2c_arr = np.array(w2c_list)
    c2w_arr = np.linalg.inv(w2c_arr)
    C0_inv = np.linalg.inv(c2w_arr[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w_arr])
    w2c_arr = np.linalg.inv(c2w_aligned)

    # compute action labels from relative c2w
    c2w_arr = np.linalg.inv(w2c_arr)
    C_inv = np.linalg.inv(c2w_arr[:-1])
    rel_c2w = np.zeros_like(c2w_arr)
    rel_c2w[0] = c2w_arr[0]
    rel_c2w[1:] = C_inv @ c2w_arr[1:]

    trans_one_hot = np.zeros((n, 4), dtype=np.int32)
    rot_one_hot   = np.zeros((n, 4), dtype=np.int32)
    move_thresh = 0.0001
    for i in range(1, n):
        move_dirs  = rel_c2w[i, :3, 3]
        move_norm  = np.linalg.norm(move_dirs)
        rot_angles = R.from_matrix(rel_c2w[i, :3, :3]).as_euler('xyz', degrees=True)
        if move_norm > move_thresh:
            nd = move_dirs / move_norm
            ang = np.arccos(nd.clip(-1, 1)) * 180 / np.pi
            if ang[2] <  60: trans_one_hot[i, 0] = 1
            if ang[2] > 120: trans_one_hot[i, 1] = 1
            if ang[0] <  60: trans_one_hot[i, 2] = 1
            if ang[0] > 120: trans_one_hot[i, 3] = 1
        if rot_angles[1] >  5e-2: rot_one_hot[i, 0] = 1
        if rot_angles[1] < -5e-2: rot_one_hot[i, 1] = 1
        if rot_angles[0] >  5e-2: rot_one_hot[i, 2] = 1
        if rot_angles[0] < -5e-2: rot_one_hot[i, 3] = 1

    trans_label = one_hot_to_label(torch.tensor(trans_one_hot))
    rot_label   = one_hot_to_label(torch.tensor(rot_one_hot))
    action      = trans_label * 9 + rot_label

    return (
        torch.as_tensor(w2c_arr, dtype=torch.float32),
        torch.as_tensor(np.array(intrinsic_list), dtype=torch.float32),
        action,
    )


def extract_first_frame(video_path, out_path):
    """Extract first frame of a video and save as PNG."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read {video_path}"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame_rgb).save(out_path)
    return out_path


def save_video(video, path, fps=16):
    if video.ndim == 5:
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c").cpu().numpy()
    imageio.mimwrite(path, vid, fps=fps)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",          type=str, required=True)
    parser.add_argument("--action_ckpt",         type=str, required=True)
    parser.add_argument("--temporal_embed_ckpt", type=str, required=True,
                        help="Path to diffusion_pytorch_model.safetensors from training checkpoint")
    parser.add_argument("--pose_json",           type=str, required=True,
                        help="Training-format pose.json (keys: w2c, intrinsic)")
    parser.add_argument("--image_path",          type=str, required=True,
                        help="Path to GT video (first frame used as image condition)")
    parser.add_argument("--output_dir",          type=str, default="outputs/eval_temporal_embed")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── extract first frame ──
    first_frame_path = os.path.join(args.output_dir, "first_frame.png")
    if args.image_path.endswith(".mp4"):
        extract_first_frame(args.image_path, first_frame_path)
    else:
        first_frame_path = args.image_path

    # ── load pose + actions ──
    w2c, intrinsics, action = pose_json_to_inputs(args.pose_json)
    n_latents = w2c.shape[0]
    video_length = (n_latents - 1) * 4 + 1  # latent → video frames
    print(f"Pose: {n_latents} latent frames → {video_length} video frames")

    # ── build pipeline ──
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    parallel_dims = initialize_parallel_state(sp=world_size)
    torch.cuda.set_device(local_rank)

    infer_args = argparse.Namespace(
        chunk_latent_frames=4,
        model_type="ar",
        attn_type="torch_causal",
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        use_fp8_gemm=False,
        quant_type="fp8-per-block",
        use_vae_parallel=False,
    )
    initialize_infer_state(infer_args)

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=True,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=args.action_ckpt,
    )

    # ── inject temporal embed/token weights ──
    transformer = pipe.transformer
    sd = load_file(args.temporal_embed_ckpt)
    import torch.nn as nn
    if "temporal_frame_embed.weight" in sd:
        weight = sd["temporal_frame_embed.weight"]
        max_frames, hidden_size = weight.shape
        transformer.temporal_frame_embed = nn.Embedding(max_frames, hidden_size)
        transformer.temporal_frame_embed.weight = nn.Parameter(weight.to(next(transformer.parameters()).device))
        print(f"Loaded temporal_frame_embed: {max_frames} frames, hidden_size={hidden_size}")
    if "temporal_token_embed.weight" in sd:
        weight = sd["temporal_token_embed.weight"]
        max_frames, hidden_size = weight.shape
        transformer.temporal_token_embed = nn.Embedding(max_frames, hidden_size)
        transformer.temporal_token_embed.weight = nn.Parameter(weight.to(next(transformer.parameters()).device))
        print(f"Loaded temporal_token_embed: {max_frames} frames, hidden_size={hidden_size}")
    per_block_keys = sorted(
        [k for k in sd if k.startswith("temporal_frame_embed_blocks.")],
        key=lambda k: int(k.split(".")[1])
    )
    if per_block_keys:
        n_blocks = int(per_block_keys[-1].split(".")[1]) + 1
        w0 = sd[per_block_keys[0]]
        max_frames, hidden_size = w0.shape
        transformer.temporal_frame_embed_blocks = nn.ModuleList([
            nn.Embedding(max_frames, hidden_size) for _ in range(n_blocks)
        ])
        for k in per_block_keys:
            i = int(k.split(".")[1])
            transformer.temporal_frame_embed_blocks[i].weight = nn.Parameter(sd[k].cpu())
        device = next(transformer.parameters()).device
        transformer.temporal_frame_embed_blocks.to(device)
        print(f"Loaded temporal_frame_embed_blocks: {n_blocks} blocks, max_frames={max_frames}")

    # ── run inference ──
    out = pipe(
        enable_sr=False,
        prompt="The butter starts to melt.",
        aspect_ratio="16:9",
        num_inference_steps=args.num_inference_steps,
        video_length=video_length,
        negative_prompt="",
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        reference_image=first_frame_path,
        viewmats=w2c.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=False,
        chunk_latent_frames=4,
        model_type="ar",
    )

    # ── save and eval (rank 0 only) ──
    if local_rank == 0:
        out_path = os.path.join(args.output_dir, "gen.mp4")
        save_video(out.videos, out_path)
        print(f"Saved to {out_path}")

        # ── compute frame MSE vs GT ──
        gen = out.videos  # [1, C, F, H, W] or [C, F, H, W], values in [0,1]
        if gen.ndim == 5:
            gen = gen[0]  # [C, F, H, W]

        # Load GT video frames (only possible when image_path is a .mp4)
        gt_frames = []
        if args.image_path.endswith(".mp4"):
            cap = cv2.VideoCapture(args.image_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gt_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

        if gt_frames:
            # Skip the conditioning frame (frame 0 = first_frame.png) before comparing
            gen_for_mse = gen[:, 1:]
            n_gen = gen_for_mse.shape[1]
            n_compare = min(n_gen, len(gt_frames))
            gen_for_mse = gen_for_mse[:, :n_compare]
            gt = torch.from_numpy(np.stack(gt_frames[:n_compare], axis=0)).float() / 255.0  # [F, H, W, C]
            gt = gt.permute(3, 0, 1, 2)  # [C, F, H, W]

            # Resize GT to match generated resolution if needed
            if gt.shape[-2:] != gen_for_mse.shape[-2:]:
                gt = torch.nn.functional.interpolate(
                    gt.permute(1, 0, 2, 3),  # [F, C, H, W]
                    size=gen_for_mse.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).permute(1, 0, 2, 3)  # [C, F, H, W]

            mse = ((gen_for_mse.cpu().float() - gt) ** 2).mean().item()
            print(f"Frame MSE: {mse:.6f}")

            # Write metrics to file so the training process can log them to wandb
            import json as _json
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                _json.dump({"frame_mse": mse}, f)
        else:
            print("No GT video available — skipping MSE computation.")


if __name__ == "__main__":
    main()
