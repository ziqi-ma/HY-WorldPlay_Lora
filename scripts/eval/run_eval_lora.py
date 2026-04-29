#!/usr/bin/env python3
"""
Eval script: load a LoRA checkpoint, merge into base model, run inference.

Usage:
    torchrun --nproc_per_node=N --master_port=PORT scripts/eval/run_eval_lora.py \
        --model_path <model_path> \
        --action_ckpt <ar_action_safetensors> \
        --lora_ckpt <checkpoint-N/transformer/diffusion_pytorch_model.safetensors> \
        --lora_rank 16 \
        --lora_alpha 32 \
        --lora_target_modules img_attn_q img_attn_k ... \
        --pose_json <pose.json> \
        --image_path <image> \
        --output_dir <out_dir> \
        --num_frames 61 \
        --num_height 480 \
        --num_width 832 \
        --num_inference_steps 30 \
        --seed 42
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import numpy as np
import torch
import torch.distributed as dist
import imageio
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation as R


# ── action helpers (same as training data preprocessing) ─────────────────────

_mapping = {
    (0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3, (0, 0, 0, 1): 4, (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6, (0, 1, 1, 0): 7, (0, 1, 0, 1): 8,
}

def _one_hot_to_label(one_hot):
    return torch.tensor([_mapping[tuple(r.tolist())] for r in one_hot])

def pose_json_to_inputs(pose_json_path):
    pose_json = json.load(open(pose_json_path))
    keys = list(pose_json.keys())
    n = len(keys)

    w2c_list, intrinsic_list = [], []
    for k in keys:
        w2c = np.array(pose_json[k]['w2c'])
        intrinsic = np.array(pose_json[k]['intrinsic'])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        w2c_list.append(w2c)
        intrinsic_list.append(intrinsic)

    w2c_arr = np.array(w2c_list)
    c2w_arr = np.linalg.inv(w2c_arr)
    C0_inv = np.linalg.inv(c2w_arr[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w_arr])
    w2c_arr = np.linalg.inv(c2w_aligned)

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

    trans_label = _one_hot_to_label(torch.tensor(trans_one_hot))
    rot_label   = _one_hot_to_label(torch.tensor(rot_one_hot))
    action      = trans_label * 9 + rot_label

    return (
        torch.as_tensor(w2c_arr, dtype=torch.float32),
        torch.as_tensor(np.array(intrinsic_list), dtype=torch.float32),
        action,
    )


class _LinearWithBase(torch.nn.Module):
    """Thin wrapper that keeps the original base weight accessible via .base_layer."""
    def __init__(self, linear: torch.nn.Linear, base_weight: torch.Tensor):
        super().__init__()
        self.linear = linear
        # Frozen copy of the pre-merge weight for prope_base_qk
        self.base_weight = base_weight

    class _BaseProxy:
        """Mimics nn.Linear just enough for _base_linear() to call it."""
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias
        def __call__(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    @property
    def base_layer(self):
        return self._BaseProxy(self.base_weight, self.linear.bias)

    def forward(self, x):
        return self.linear(x)

    # Forward attribute lookups to the wrapped linear so the transformer code
    # (weight, bias, etc.) still works unchanged.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)


def merge_lora(pipe, lora_ckpt_path, lora_rank, lora_alpha, lora_target_modules, rank,
               prope_base_qk: bool = False):
    """Load LoRA safetensors and merge into pipe.transformer weights.

    When prope_base_qk=True, img_attn_q/k layers are wrapped to retain base
    weights so PRoPE can use them without the LoRA delta.
    """
    scale = lora_alpha / lora_rank
    state_dict = load_file(lora_ckpt_path)

    merged = 0
    prope_qk_names = {"img_attn_q", "img_attn_k"}

    for name, module in pipe.transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(t in name for t in lora_target_modules):
            continue

        lora_A = lora_B = None
        for a_key in [f"{name}.lora_A", f"{name}.lora_A.weight"]:
            if a_key in state_dict:
                lora_A = state_dict[a_key]
                break
        for b_key in [f"{name}.lora_B", f"{name}.lora_B.weight"]:
            if b_key in state_dict:
                lora_B = state_dict[b_key]
                break

        if lora_A is None or lora_B is None:
            continue

        # Save base weight before merging if this layer feeds PRoPE Q/K
        is_prope_qk = prope_base_qk and any(t in name for t in prope_qk_names)
        if is_prope_qk:
            base_weight = module.weight.data.clone()

        module.weight.data += scale * (
            lora_B.to(module.weight.device, dtype=module.weight.dtype)
            @ lora_A.to(module.weight.device, dtype=module.weight.dtype)
        ).T
        merged += 1

        # Wrap the layer so _base_linear() can retrieve the pre-merge weight
        if is_prope_qk:
            wrapper = _LinearWithBase(module, base_weight)
            parent = pipe.transformer.get_submodule(".".join(name.split(".")[:-1]))
            setattr(parent, name.split(".")[-1], wrapper)

    if prope_base_qk:
        pipe.transformer.set_prope_base_qk(True)

    if rank == 0:
        print(f"[LoRA] merged {merged} layers (scale={scale:.2f})"
              + (" [prope_base_qk=True]" if prope_base_qk else ""))
    return merged


def run_one(pipe, pose_json, image_path, output_dir, num_frames, num_height, num_width,
            num_inference_steps, seed, rank, latents=None, prompt=""):
    viewmats, Ks, action = pose_json_to_inputs(pose_json)

    out = pipe(
        enable_sr=False,
        prompt=prompt,
        aspect_ratio="9:16",
        num_inference_steps=num_inference_steps,
        sr_num_inference_steps=None,
        video_length=num_frames,
        negative_prompt="",
        seed=seed,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=False,
        chunk_latent_frames=4,
        model_type="ar",
        user_height=num_height,
        user_width=num_width,
        reference_image=image_path,
        latents=latents,
    )

    if rank == 0:
        video_tensor = out.videos
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor[0].cpu().numpy()
            if video_np.shape[0] == 3:
                video_np = np.transpose(video_np, (1, 2, 3, 0))
            video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        else:
            video_np = np.array(video_tensor)

        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, "video.mp4")
        imageio.mimsave(video_path, list(video_np), fps=24)
        print(f"  saved -> {video_path}")

    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--action_ckpt", required=True)
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["img_attn_q", "img_attn_k", "img_attn_v", "img_attn_proj",
                                 "img_attn_prope_proj", "txt_attn_q", "txt_attn_k",
                                 "txt_attn_v", "txt_attn_proj",
                                 "linear1_q", "linear1_k", "linear1_v"])
    parser.add_argument("--pose_json", nargs="+", required=True)
    parser.add_argument("--output_dir", nargs="+", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_path", type=str, default=None,
                        help="Path to fixed_noise.pt saved during training")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prope_base_qk", action="store_true",
                        help="Use base Q/K (no LoRA delta) for PRoPE camera attention scores")
    args = parser.parse_args()

    if len(args.pose_json) != len(args.output_dir):
        raise ValueError("--pose_json and --output_dir must have the same number of entries")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    from hyvideo.generate import pose_to_input  # noqa: triggers collective init
    from hyvideo.commons.infer_state import initialize_infer_state

    class _InferArgs:
        sage_blocks_range = "0-53"
        use_sageattn = False
        enable_torch_compile = False
        use_fp8_gemm = False
        quant_type = "fp8-per-block"
        include_patterns = "double_blocks"
        use_vae_parallel = False

    initialize_infer_state(_InferArgs())

    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
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

    if rank == 0:
        print(f"Merging LoRA from {args.lora_ckpt}")
    merge_lora(pipe, args.lora_ckpt, args.lora_rank, args.lora_alpha,
               args.lora_target_modules, rank, prope_base_qk=args.prope_base_qk)
    dist.barrier()

    latents = None
    if args.noise_path is not None:
        latents = torch.load(args.noise_path, map_location=f"cuda:{local_rank}")
        if rank == 0:
            print(f"Loaded fixed noise from {args.noise_path}")

    for pose_json, output_dir in zip(args.pose_json, args.output_dir):
        if args.skip_existing and os.path.exists(os.path.join(output_dir, "video.mp4")):
            if rank == 0:
                print(f"Skipping {output_dir} (already exists)")
            continue
        if rank == 0:
            print(f"Running inference on {pose_json}")
        run_one(pipe, pose_json, args.image_path, output_dir,
                args.num_frames, args.num_height, args.num_width,
                args.num_inference_steps, args.seed, rank, latents=latents,
                prompt=args.prompt)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
