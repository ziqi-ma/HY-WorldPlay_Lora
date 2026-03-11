#!/usr/bin/env python3
"""
Offline eval: load LoRA checkpoint, merge into base model, run sp=8 inference.
Usage: torchrun --nproc_per_node=8 test_eval_checkpoint.py
"""
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch
import torch.distributed as dist

LORA_CKPT = "/data/ziqi/Repos/HY-WorldPlay-New/outputs/lora_waterfill_right11/checkpoint-100/transformer/diffusion_pytorch_model.safetensors"
LORA_RANK = 16
LORA_ALPHA = 32
POSE_STRING = "right-11"
REFERENCE_IMAGE = "/data/ziqi/data/worldstate/predynamic/tank_filling_up.png"
VIDEO_LENGTH = 45
SEED = 1
OUTPUT_PATH = "/data/ziqi/Repos/HY-WorldPlay-New/outputs/test_eval/test_eval_ckpt100.mp4"

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # All ranks: import hyvideo.generate (collective init sp=WORLD_SIZE)
    from hyvideo.generate import pose_to_input
    # Keep sp=8 (don't override to sp=1)

    from hyvideo.commons.infer_state import initialize_infer_state

    class _EvalInferArgs:
        sage_blocks_range = "0-53"
        use_sageattn = False
        enable_torch_compile = False
        use_fp8_gemm = False
        quant_type = "fp8-per-block"
        include_patterns = "double_blocks"
        use_vae_parallel = False

    initialize_infer_state(_EvalInferArgs())

    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path="/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038",
        transformer_version="480p_i2v",
        enable_offloading=True,
        enable_group_offloading=False,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=torch.bfloat16,
        action_ckpt="/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/diffusion_pytorch_model.safetensors",
    )

    # --- Skip LoRA merge (testing base model only) ---
    if rank == 0:
        print("Skipping LoRA merge — running base model only")

    dist.barrier()

    # --- Prepare pose/action ---
    latent_num = (VIDEO_LENGTH - 1) // 4 + 1
    viewmats, Ks, action = pose_to_input(POSE_STRING, latent_num)

    if rank == 0:
        print("Running inference (sp=8, all ranks)...")

    out = pipe(
        enable_sr=False,
        prompt="",
        aspect_ratio="9:16",
        num_inference_steps=50,
        sr_num_inference_steps=None,
        video_length=VIDEO_LENGTH,
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
        user_height=768,
        user_width=432,
        reference_image=REFERENCE_IMAGE,
    )

    if rank == 0:
        print("Inference done.")
        video_tensor = out.videos
        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor[0].cpu().numpy()
            if video_np.shape[0] == 3:
                video_np = np.transpose(video_np, (1, 2, 3, 0))
            video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        else:
            video_np = np.array(video_tensor)

        print(f"Video shape: {video_np.shape}, min={video_np.min()}, max={video_np.max()}, mean={video_np.mean():.1f}")

        import imageio
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        imageio.mimsave(OUTPUT_PATH, list(video_np), fps=24)
        print(f"Saved to {OUTPUT_PATH}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
