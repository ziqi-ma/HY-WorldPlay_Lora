#!/usr/bin/env python3
"""
Evaluate a LoRA-finetuned HunyuanVideo AR model by generating a video
with the same setup as training and computing LPIPS + L2 against GT frames.

Usage:
    torchrun --nproc_per_node=8 evaluate_lora.py \
        --lora_path outputs/lora_waterfill_right11/checkpoint-500/transformer/diffusion_pytorch_model.safetensors \
        --gt_frames_dir training_data/waterfill_right11/frames/ \
        --image_path training_data/waterfill_right11/frames/0000.png \
        --pose_string "right-11" \
        --model_path /path/to/HunyuanVideo-1.5 \
        --action_ckpt /path/to/ar_model/diffusion_pytorch_model.safetensors \
        --output_path outputs/eval_lora/
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import json
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from hyvideo.generate import (
    generate_video, pose_to_input, initialize_infer_state, str_to_bool
)
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline


def apply_lora_to_transformer(transformer, lora_path, lora_rank=16, lora_alpha=32, target_modules=None):
    """Load LoRA weights from safetensors and merge into transformer."""
    from trainer.layers.lora.linear import NNLinearWithLoRA, replace_submodule
    import torch.nn as nn

    if target_modules is None:
        target_modules = [
            "img_attn_q", "img_attn_k", "img_attn_v", "img_attn_proj",
            "img_attn_prope_proj",
            "txt_attn_q", "txt_attn_k", "txt_attn_v", "txt_attn_proj",
            "linear1_q", "linear1_k", "linear1_v",
        ]

    # Load LoRA state dict
    lora_state_dict = load_file(lora_path)

    # Find which keys are LoRA A/B
    lora_keys = set()
    for key in lora_state_dict:
        # Keys look like: double_blocks.0.img_attn_q.lora_A.weight
        base = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        base = base.replace(".lora_A", "").replace(".lora_B", "")
        lora_keys.add(base)

    converted = 0
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Check if we have LoRA weights for this layer
        # Try different key formats
        lora_A = None
        lora_B = None
        for prefix in [name, f"diffusion_model.{name}"]:
            for a_key in [f"{prefix}.lora_A.weight", f"{prefix}.lora_A"]:
                if a_key in lora_state_dict:
                    lora_A = lora_state_dict[a_key]
                    break
            for b_key in [f"{prefix}.lora_B.weight", f"{prefix}.lora_B"]:
                if b_key in lora_state_dict:
                    lora_B = lora_state_dict[b_key]
                    break
            if lora_A is not None:
                break

        if lora_A is None or lora_B is None:
            continue

        # Training forward computes x @ (B @ A), but nn.Linear computes
        # x @ W.T.  To match, we need W' = W + scale*(B @ A).T.
        scale = lora_alpha / lora_rank
        module.weight.data += scale * (lora_B.to(module.weight.device, dtype=module.weight.dtype)
                                       @ lora_A.to(module.weight.device, dtype=module.weight.dtype)).T
        converted += 1

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"Merged LoRA weights into {converted} layers (from {lora_path})")


def compute_metrics(gen_frames, gt_frames_dir):
    """Compute frame-average LPIPS and L2 between generated and GT frames."""
    import lpips
    from torchvision import transforms

    # Load GT frames
    gt_files = sorted([f for f in os.listdir(gt_frames_dir) if f.endswith('.png')])
    gt_images = []
    for f in gt_files:
        img = Image.open(os.path.join(gt_frames_dir, f)).convert('RGB')
        gt_images.append(np.array(img))
    gt_images = np.stack(gt_images)  # [T, H, W, 3]

    # gen_frames: [T, H, W, 3] uint8 numpy
    num_frames = min(len(gen_frames), len(gt_images))
    gen_frames = gen_frames[:num_frames]
    gt_images = gt_images[:num_frames]

    # Resize gen to match GT if needed
    if gen_frames.shape[1:3] != gt_images.shape[1:3]:
        import cv2
        h, w = gt_images.shape[1], gt_images.shape[2]
        gen_frames = np.stack([
            cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
            for f in gen_frames
        ])

    # L2 error (normalized to [0,1])
    gen_float = gen_frames.astype(np.float32) / 255.0
    gt_float = gt_images.astype(np.float32) / 255.0
    l2_per_frame = np.mean((gen_float - gt_float) ** 2, axis=(1, 2, 3))
    l2_mean = np.mean(l2_per_frame)

    # LPIPS
    loss_fn = lpips.LPIPS(net='alex').cuda()
    to_tensor = transforms.ToTensor()

    lpips_scores = []
    with torch.no_grad():
        for i in range(num_frames):
            gen_t = to_tensor(gen_frames[i]).unsqueeze(0).cuda() * 2 - 1  # [0,1] -> [-1,1]
            gt_t = to_tensor(gt_images[i]).unsqueeze(0).cuda() * 2 - 1
            score = loss_fn(gen_t, gt_t).item()
            lpips_scores.append(score)

    lpips_mean = np.mean(lpips_scores)

    return {
        "l2_mean": float(l2_mean),
        "l2_per_frame": [float(x) for x in l2_per_frame],
        "lpips_mean": float(lpips_mean),
        "lpips_per_frame": [float(x) for x in lpips_scores],
        "num_frames": num_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-finetuned model")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA safetensors checkpoint")
    parser.add_argument("--gt_frames_dir", type=str, default=None,
                        help="Directory with GT frames (optional, for metrics)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="First frame image for i2v generation")
    parser.add_argument("--pose_string", type=str, required=True,
                        help="Pose string (e.g., 'right-11')")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt (default: empty)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HunyuanVideo-1.5 model directory")
    parser.add_argument("--action_ckpt", type=str, required=True,
                        help="Path to AR action model safetensors")
    parser.add_argument("--output_path", type=str, default="outputs/eval_lora/",
                        help="Output directory")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_lora", action="store_true",
                        help="Run without LoRA (baseline comparison)")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))

    # Compute video length from pose string
    commands = [cmd.strip() for cmd in args.pose_string.split(",")]
    total_latent_duration = sum(int(cmd.split("-")[1]) for cmd in commands if cmd)
    video_length = 4 * total_latent_duration + 1

    # Create inference args (mimicking generate.py)
    class InferArgs:
        pass

    infer_args = InferArgs()
    infer_args.sage_blocks_range = "0-53"
    infer_args.use_sageattn = False
    infer_args.enable_torch_compile = False
    infer_args.use_fp8_gemm = False
    infer_args.quant_type = "fp8-per-block"
    infer_args.include_patterns = "double_blocks"
    infer_args.use_vae_parallel = False

    initialize_infer_state(infer_args)

    # Build pipeline
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=False,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=args.action_ckpt,
    )

    # Apply LoRA
    if not args.no_lora:
        apply_lora_to_transformer(
            pipe.transformer, args.lora_path,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        )

    # Prepare pose
    from hyvideo.generate import pose_string_to_json, pose_to_input
    viewmats, Ks, action = pose_to_input(args.pose_string, (video_length - 1) // 4 + 1)

    # Run inference
    out = pipe(
        enable_sr=False,
        prompt=args.prompt,
        aspect_ratio="9:16",
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=video_length,
        negative_prompt="",
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=False,
        chunk_latent_frames=4,
        model_type="ar",
        user_height=args.height,
        user_width=args.width,
        reference_image=args.image_path,
    )

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

        # Extract video frames
        video_tensor = out.videos  # [B, C, T, H, W], float [0, 1]
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor[0].cpu().numpy()
            # Handle different formats
            if video_np.shape[0] == 3:  # [C, T, H, W]
                video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
            video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        else:
            video_np = np.array(video_tensor)

        # Save generated video
        import cv2
        save_path = os.path.join(args.output_path, "eval_gen.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = video_np.shape[1], video_np.shape[2]
        writer = cv2.VideoWriter(save_path, fourcc, 24, (w, h))
        for frame in video_np:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved generated video to {save_path} ({video_np.shape[0]} frames)")

        # Save individual generated frames
        gen_frames_dir = os.path.join(args.output_path, "gen_frames")
        os.makedirs(gen_frames_dir, exist_ok=True)
        for i, frame in enumerate(video_np):
            Image.fromarray(frame).save(os.path.join(gen_frames_dir, f"{i:04d}.png"))

        # Compute metrics (only if GT frames provided)
        if args.gt_frames_dir:
            print("\nComputing metrics...")
            metrics = compute_metrics(video_np, args.gt_frames_dir)

            print(f"\n=== Evaluation Results ===")
            print(f"  Frames compared: {metrics['num_frames']}")
            print(f"  L2 (MSE, [0-1]):  {metrics['l2_mean']:.6f}")
            print(f"  LPIPS (alex):     {metrics['lpips_mean']:.4f}")

            metrics_path = os.path.join(args.output_path, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved metrics to {metrics_path}")
        else:
            print("\nNo --gt_frames_dir provided, skipping metrics.")


if __name__ == "__main__":
    main()
