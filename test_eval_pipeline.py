#!/usr/bin/env python3
"""
Quick test: run the eval pipeline (matching run.sh exactly) to verify it
generates a video and computes LPIPS + L2 against GT frames.

Usage:
    torchrun --nproc_per_node=8 test_eval_pipeline.py
"""
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

def main():
    # --- Config ---
    MODEL_PATH = "/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
    ACTION_CKPT = "/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors"
    GT_FRAMES_DIR = "/data/ziqi/Repos/HY-WorldPlay-New/training_data/waterfill_right11/frames/"
    IMAGE_PATH = "/data/ziqi/data/worldstate/predynamic/tank_filling_up.png"
    POSE_STRING = "right-11"
    NUM_FRAMES = 45
    HEIGHT = 768
    WIDTH = 432
    NUM_INFERENCE_STEPS = 50
    SEED = 1
    OUTPUT_DIR = "/data/ziqi/Repos/HY-WorldPlay-New/outputs/test_eval/"

    rank = int(os.environ.get("RANK", "0"))

    # Use generate.py exactly like run.sh does — this sets up sp=WORLD_SIZE
    from hyvideo.generate import pose_to_input, initialize_infer_state
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

    class _InferArgs:
        sage_blocks_range = "0-53"
        use_sageattn = False
        enable_torch_compile = False
        use_fp8_gemm = False
        quant_type = "fp8-per-block"
        include_patterns = "double_blocks"
        use_vae_parallel = False

    initialize_infer_state(_InferArgs())

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=MODEL_PATH,
        transformer_version="480p_i2v",
        enable_offloading=True,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt=ACTION_CKPT,
    )

    if rank == 0:
        print("Pipeline created.")

    # Prepare pose (all ranks)
    video_length = NUM_FRAMES
    latent_num = (video_length - 1) // 4 + 1
    viewmats, Ks, action = pose_to_input(POSE_STRING, latent_num)

    gt_files = sorted([f for f in os.listdir(GT_FRAMES_DIR) if f.endswith('.png')])
    eval_image_path = IMAGE_PATH

    if rank == 0:
        print(f"Running inference with sp={int(os.environ.get('WORLD_SIZE', '1'))}...")

    # All ranks participate in inference (sp=WORLD_SIZE)
    out = pipe(
        enable_sr=False,
        prompt="",
        aspect_ratio="9:16",
        num_inference_steps=NUM_INFERENCE_STEPS,
        sr_num_inference_steps=None,
        video_length=video_length,
        negative_prompt="",
        seed=SEED,
        output_type="pt",
        prompt_rewrite=False,
        return_pre_sr_video=False,
        viewmats=viewmats.unsqueeze(0),
        Ks=Ks.unsqueeze(0),
        action=action.unsqueeze(0),
        few_step=False,
        chunk_latent_frames=4,
        model_type="ar",
        user_height=HEIGHT,
        user_width=WIDTH,
        reference_image=eval_image_path,
    )

    if rank == 0:
        print("Inference done.")

        # Extract video
        video_tensor = out.videos
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor[0].cpu().numpy()
            if video_np.shape[0] == 3:
                video_np = np.transpose(video_np, (1, 2, 3, 0))
            video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        else:
            video_np = np.array(video_tensor)
        print(f"Generated video shape: {video_np.shape}")

        # Load GT frames
        from PIL import Image
        gt_frames = np.stack([
            np.array(Image.open(os.path.join(GT_FRAMES_DIR, f)).convert('RGB'))
            for f in gt_files
        ])

        # Compute metrics
        num_frames = min(len(video_np), len(gt_frames))
        gen = video_np[:num_frames]
        gt_eval = gt_frames[:num_frames]

        if gen.shape[1:3] != gt_eval.shape[1:3]:
            import cv2
            h, w = gt_eval.shape[1], gt_eval.shape[2]
            gen = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in gen])

        gen_f = gen.astype(np.float32) / 255.0
        gt_f = gt_eval.astype(np.float32) / 255.0
        l2_mean = float(np.mean((gen_f - gt_f) ** 2))

        import lpips
        from torchvision import transforms
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        lpips_fn.eval()
        to_tensor = transforms.ToTensor()
        lpips_scores = []
        with torch.no_grad():
            for i in range(num_frames):
                gen_t = to_tensor(gen[i]).unsqueeze(0).cuda() * 2 - 1
                gt_t = to_tensor(gt_eval[i]).unsqueeze(0).cuda() * 2 - 1
                score = lpips_fn(gen_t, gt_t).item()
                lpips_scores.append(score)
        lpips_mean = float(np.mean(lpips_scores))

        print(f"\n=== Eval Results ===")
        print(f"  Frames: {num_frames}")
        print(f"  L2 (MSE): {l2_mean:.6f}")
        print(f"  LPIPS:    {lpips_mean:.4f}")

        # Save video
        import imageio
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        video_path = os.path.join(OUTPUT_DIR, "test_eval_original.mp4")
        imageio.mimsave(video_path, list(video_np), fps=24)
        print(f"  Saved video to {video_path}")


if __name__ == "__main__":
    main()
