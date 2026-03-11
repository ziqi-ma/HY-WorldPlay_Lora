#!/usr/bin/env python3
"""Decode preprocessed training latents with VAE to verify ground truth."""
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

MODEL_PATH = "/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
LATENT_PATH = "/data/ziqi/Repos/HY-WorldPlay-New/training_data/waterfill_right11/latents.pt"
OUTPUT_PATH = "/data/ziqi/Repos/HY-WorldPlay-New/outputs/test_eval/decoded_gt_latents.mp4"

def main():
    data = torch.load(LATENT_PATH, map_location='cpu')
    latent = data['latent']  # [1, 32, 12, 48, 27]
    print(f"Latent shape: {latent.shape}, dtype: {latent.dtype}")

    # Load VAE
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
    vae_path = os.path.join(MODEL_PATH, "vae")
    vae = AutoencoderKLConv3D.from_pretrained(vae_path, torch_dtype=torch.float16)
    vae = vae.cuda().eval()

    # The latent needs to be unscaled by the VAE scaling factor
    scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.476986
    print(f"VAE scaling factor: {scaling_factor}")

    with torch.no_grad():
        z = latent.to(device='cuda', dtype=torch.float16) / scaling_factor
        # VAE expects [B, C, T, H, W]
        print(f"Decoding latent {z.shape}...")
        decoded = vae.decode(z).sample  # [B, C, T, H, W]
        print(f"Decoded shape: {decoded.shape}")

    # Convert to video frames [T, H, W, C] uint8
    video = decoded[0].cpu().float().numpy()
    if video.shape[0] == 3:  # [C, T, H, W]
        video = np.transpose(video, (1, 2, 3, 0))
    video = np.clip((video + 1) / 2 * 255, 0, 255).astype(np.uint8)
    print(f"Video shape: {video.shape}, min={video.min()}, max={video.max()}, mean={video.mean():.1f}")

    import imageio
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimsave(OUTPUT_PATH, list(video), fps=24)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
