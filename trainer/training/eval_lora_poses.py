#!/usr/bin/env python3
"""Standalone LoRA evaluation: load base model + LoRA checkpoint, run inference
on all pose JSONs in a directory, and save videos."""
import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from trainer.logger import init_logger

logger = init_logger(__name__)

LORA_TARGET_MODULES = [
    "img_attn_q", "img_attn_k", "img_attn_v", "img_attn_proj",
    "img_attn_prope_proj",
    "txt_attn_q", "txt_attn_k", "txt_attn_v", "txt_attn_proj",
    "img_mlp", "txt_mlp",
]


def discover_poses(pose_dir):
    poses = []
    for sub in sorted(os.listdir(pose_dir)):
        pf = os.path.join(pose_dir, sub, "pose.json")
        if os.path.isfile(pf):
            poses.append((sub, pf))
    return poses


def merge_lora(pipe, lora_ckpt_path, lora_rank, lora_alpha):
    """Merge LoRA weights into the pipeline transformer."""
    ckpt_file = os.path.join(lora_ckpt_path, "transformer",
                             "diffusion_pytorch_model.safetensors")
    lora_state_dict = load_file(ckpt_file)
    scale = lora_alpha / lora_rank

    merged = 0
    for name, module in pipe.transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(t in name for t in LORA_TARGET_MODULES):
            continue
        lora_A = lora_state_dict.get(f"{name}.lora_A")
        lora_B = lora_state_dict.get(f"{name}.lora_B")
        if lora_A is None or lora_B is None:
            continue
        module.weight.data += scale * (
            lora_B.to(module.weight.device, dtype=module.weight.dtype)
            @ lora_A.to(module.weight.device, dtype=module.weight.dtype)
        )
        merged += 1
    logger.info("Merged LoRA into %d layers (rank=%d, alpha=%d, scale=%.2f)",
                merged, lora_rank, lora_alpha, scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--eval_pose_dir", required=True)
    parser.add_argument("--eval_image", required=True)
    parser.add_argument("--num_frames", type=int, default=61)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=832)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_prefix", default="")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.lora_ckpt, "eval_videos")

    # Initialize inference state (collective op — all ranks)
    from hyvideo.generate import pose_to_input
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

    # Build pipeline (all ranks)
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
    ar_action_path = os.path.join(
        os.path.dirname(args.model_path.rstrip("/")),
        "hywp_ckpt/ar_rl_model/diffusion_pytorch_model.safetensors"
    ) if not os.path.exists("/tmp/hywp_ckpt") else \
        "/tmp/hywp_ckpt/ar_rl_model/diffusion_pytorch_model.safetensors"

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=True,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=ar_action_path,
    )

    # Merge LoRA (all ranks)
    merge_lora(pipe, args.lora_ckpt, args.lora_rank, args.lora_alpha)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Discover and run poses
    poses = discover_poses(args.eval_pose_dir)
    logger.info("Found %d eval poses: %s", len(poses), [n for n, _ in poses])

    latent_num = (args.num_frames - 1) // 4 + 1

    for pose_name, pose_path in poses:
        logger.info("Evaluating pose '%s'", pose_name)
        viewmats, Ks, action = pose_to_input(pose_path, latent_num)

        out = pipe(
            enable_sr=False,
            prompt="",
            aspect_ratio="9:16",
            num_inference_steps=args.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=args.num_frames,
            negative_prompt="",
            seed=1,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            viewmats=viewmats.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=False,
            chunk_latent_frames=4,
            model_type="ar",
            user_height=args.num_height,
            user_width=args.num_width,
            reference_image=args.eval_image,
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

            import imageio
            os.makedirs(args.output_dir, exist_ok=True)
            video_path = os.path.join(args.output_dir, f"{args.output_prefix}{pose_name}.mp4")
            imageio.mimsave(video_path, list(video_np), fps=24)
            logger.info("Saved: %s", video_path)

        if dist.is_initialized():
            dist.barrier()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
