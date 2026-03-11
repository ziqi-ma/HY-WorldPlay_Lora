#!/usr/bin/env python3
"""
Preprocess a video for LoRA finetuning of HunyuanVideo 1.5 AR model.

Encodes video + prompt + action into .pt + pose JSON format expected by the
trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py.

Usage:
    python preprocess_video.py \
        --video_path /data/ziqi/Repos/HY-WorldPlay-New/outputs/block_all_motions/right11/noisy_block_to44.mp4  \
        --pose_string "right-11" \
        --prompt "" \
        --output_dir training_data/block_right11/ \
        --model_path /data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def compute_frames_from_pose(pose_string):
    """Compute target video frame count and t_latent from pose string.

    Each command contributes `duration` latent frames. First command also adds
    1 extra frame (the initial frame). So total latent frames = sum(durations) + 1,
    and total video frames = 4 * sum(durations) + 1.
    """
    commands = [cmd.strip() for cmd in pose_string.split(",")]
    total_latent_duration = 0
    for cmd in commands:
        if not cmd:
            continue
        parts = cmd.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pose command: {cmd}")
        total_latent_duration += int(parts[1].strip())

    t_latent = total_latent_duration + 1
    target_frames = 4 * total_latent_duration + 1
    return target_frames, t_latent


def load_video_frames(video_path, target_height, target_width, target_frames):
    """Load video frames, resize, and subsample to target_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"Loaded {total_frames} frames from {video_path}")

    if total_frames < target_frames:
        raise RuntimeError(
            f"Video too short: {total_frames} frames, need {target_frames}")

    if total_frames > target_frames:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"Subsampled to {target_frames} frames")
    else:
        print(f"Using all {total_frames} frames")

    return np.stack(frames)  # [T, H, W, 3]


def encode_video_with_vae(frames_np, vae, device):
    """Encode video frames with VAE, return latent and image_cond."""
    # Convert to tensor: [1, C, T, H, W], normalized to [-1, 1]
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0 * 2.0 - 1.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]

    print(f"  Video tensor shape: {frames_tensor.shape}")

    # Encode full video
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_dist = vae.encode(frames_tensor.to(device)).latent_dist
            latent = latent_dist.mode()
            latent.mul_(vae.config.scaling_factor)

    print(f"  Latent shape: {latent.shape}")  # [1, C_latent, T_latent, H', W']

    # Encode first frame for image_cond
    first_frame = frames_tensor[:, :, :1, :, :]  # [1, C, 1, H, W]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_cond_dist = vae.encode(first_frame.to(device)).latent_dist
            image_cond = image_cond_dist.mode()
            image_cond.mul_(vae.config.scaling_factor)

    print(f"  image_cond shape: {image_cond.shape}")  # [1, C_latent, 1, H', W']

    return latent.cpu(), image_cond.cpu()


def encode_text_with_llm(prompt, model_path, device):
    """Encode prompt with LLM text encoder (Qwen2.5-VL-7B)."""
    from hyvideo.models.text_encoders import PROMPT_TEMPLATE, TextEncoder

    text_encoder_path = os.path.join(model_path, "text_encoder", "llm")
    text_encoder = TextEncoder(
        text_encoder_type="llm",
        tokenizer_type="llm",
        text_encoder_path=text_encoder_path,
        max_length=1000,
        text_encoder_precision="fp16",
        prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
        prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=None,
        device=device,
    )

    text_inputs = text_encoder.text2tokens(prompt, data_type="video", max_length=1000)
    prompt_outputs = text_encoder.encode(text_inputs, data_type="video", device=device)
    prompt_embeds = prompt_outputs.hidden_state  # [1, seq_len, dim]
    prompt_mask = prompt_outputs.attention_mask  # [1, seq_len]

    print(f"  prompt_embeds shape: {prompt_embeds.shape}")
    print(f"  prompt_mask shape: {prompt_mask.shape}")

    # Also encode empty string for negative prompt
    neg_inputs = text_encoder.text2tokens("", data_type="video", max_length=1000)
    neg_outputs = text_encoder.encode(neg_inputs, data_type="video", device=device)
    neg_prompt_embeds = neg_outputs.hidden_state
    neg_prompt_mask = neg_outputs.attention_mask

    # Clean up
    del text_encoder
    torch.cuda.empty_cache()

    return (prompt_embeds.cpu(), prompt_mask.cpu(),
            neg_prompt_embeds.cpu(), neg_prompt_mask.cpu())


def encode_text_with_byt5(prompt, model_path, device):
    """Encode prompt with ByT5 text encoder."""
    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2
    from hyvideo.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat

    load_from = os.path.join(model_path, "text_encoder")
    glyph_root = os.path.join(load_from, "Glyph-SDXL-v2")
    byT5_google_path = os.path.join(load_from, "byt5-small")

    if not os.path.exists(byT5_google_path):
        byT5_google_path = "google/byt5-small"

    byt5_args = dict(
        byT5_google_path=byT5_google_path,
        byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
        multilingual_prompt_format_color_path=os.path.join(glyph_root, "assets/color_idx.json"),
        multilingual_prompt_format_font_path=os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json"),
        byt5_max_length=256,
    )

    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=device)
    byt5_model = byt5_kwargs["byt5_model"]
    byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
    byt5_max_length = byt5_kwargs["byt5_max_length"]

    prompt_format = MultilingualPromptFormat(
        font_path=byt5_args["multilingual_prompt_format_font_path"],
        color_path=byt5_args["multilingual_prompt_format_color_path"],
    )

    def encode_single_byt5(text):
        from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
        glyph_texts = [text] if isinstance(text, str) else text
        text_styles = [{"color": None, "font-family": None} for _ in range(len(glyph_texts))]
        formatted_text = prompt_format.format_prompt(glyph_texts, text_styles)
        text_ids, text_mask = HunyuanVideo_1_5_Pipeline.get_byt5_text_tokens(
            byt5_tokenizer, byt5_max_length, formatted_text
        )
        text_ids = text_ids.to(device=device)
        text_mask = text_mask.to(device=device)
        with torch.no_grad():
            byt5_outputs = byt5_model(text_ids, attention_mask=text_mask.float())
        return byt5_outputs[0], text_mask

    byt5_text_states, byt5_text_mask = encode_single_byt5(prompt)
    print(f"  byt5_text_states shape: {byt5_text_states.shape}")
    print(f"  byt5_text_mask shape: {byt5_text_mask.shape}")

    # Also encode empty string for negative
    neg_byt5_states, neg_byt5_mask = encode_single_byt5("")

    del byt5_model, byt5_tokenizer
    torch.cuda.empty_cache()

    return (byt5_text_states.cpu(), byt5_text_mask.cpu(),
            neg_byt5_states.cpu(), neg_byt5_mask.cpu())


def encode_image_with_siglip(first_frame_np, model_path, target_height, target_width, device):
    """Encode first frame with SigLIP vision encoder."""
    from hyvideo.models.vision_encoder import VisionEncoder
    from hyvideo.utils.data_utils import resize_and_center_crop

    vision_encoder_path = os.path.join(model_path, "vision_encoder", "siglip")
    vision_encoder = VisionEncoder(
        vision_encoder_type="siglip",
        vision_encoder_precision="fp16",
        vision_encoder_path=vision_encoder_path,
        processor_type=None,
        processor_path=None,
        output_key=None,
        logger=None,
        device=device,
    )

    # Resize and center crop for vision encoder
    input_image_np = resize_and_center_crop(
        first_frame_np, target_width=target_width, target_height=target_height
    )
    vision_output = vision_encoder.encode_images(input_image_np)
    vision_states = vision_output.last_hidden_state.to(dtype=torch.bfloat16)

    print(f"  vision_states shape: {vision_states.shape}")

    del vision_encoder
    torch.cuda.empty_cache()

    return vision_states.cpu()


def generate_training_pose_json(pose_string, t_latent):
    """Generate pose JSON in training format (w2c + intrinsic, per video frame).

    The training dataset (ar_camera_hunyuan_w_mem_dataset.py) expects:
    - Keys: string frame indices "0", "1", "2", ...
    - Values: {"w2c": [[4x4 matrix]], "intrinsic": [[3x3 matrix]]}
    - Indexing: pose_keys[0] for latent 0, pose_keys[4*k] for latent k (k >= 1)
    """
    from hyvideo.generate import pose_string_to_json

    print(f"  Generating pose: '{pose_string}' ({t_latent} latent frames)")

    # Get inference-format pose JSON (per-latent, with "extrinsic" (c2w) and "K")
    inference_pose = pose_string_to_json(pose_string)

    # Convert to training format (per-video-frame, with "w2c" and "intrinsic")
    # Training dataset accesses: pose_keys[0] for i=0, pose_keys[4*k] for i=k (k>=1)
    # So we need entries from frame 0 to frame 4*(t_latent-1), that's 4*t_latent-3 entries
    training_pose = {}
    num_video_frames = 4 * (t_latent - 1) + 1  # = 4*t_latent - 3

    for frame_idx in range(num_video_frames):
        # Map video frame to latent index
        if frame_idx == 0:
            latent_idx = 0
        else:
            latent_idx = (frame_idx - 1) // 4 + 1
            latent_idx = min(latent_idx, t_latent - 1)

        # Get c2w from inference pose and convert to w2c
        c2w = np.array(inference_pose[str(latent_idx)]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        K = np.array(inference_pose[str(latent_idx)]["K"])

        training_pose[str(frame_idx)] = {
            "w2c": w2c.tolist(),
            "intrinsic": K.tolist(),
        }

    return training_pose


def main():
    parser = argparse.ArgumentParser(description="Preprocess video for LoRA finetuning")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--pose_string", type=str, required=True,
                        help="Action pose string, e.g. 'right-11' (duration in latent frames)")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (empty for unconditional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for preprocessed data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HunyuanVideo-1.5 model directory")
    parser.add_argument("--height", type=int, default=432, help="Target video height")
    parser.add_argument("--width", type=int, default=768, help="Target video width")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Step 0: Compute target frame count from pose string ────────────
    target_frames, t_latent = compute_frames_from_pose(args.pose_string)
    print(f"Pose '{args.pose_string}' → {target_frames} video frames, {t_latent} latent frames")

    # ─── Step 1: Load and preprocess video frames ──────────────────────
    print("\n=== Step 1: Loading video frames ===")
    frames_np = load_video_frames(args.video_path, args.height, args.width, target_frames)
    first_frame_np = frames_np[0].copy()  # uint8 [H, W, 3]

    # ─── Step 2: Encode video with VAE ─────────────────────────────────
    print("\n=== Step 2: Encoding video with VAE ===")
    from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache
    vae = hunyuanvideo_15_vae_w_cache.AutoencoderKLConv3D.from_pretrained(
        os.path.join(args.model_path, "vae"),
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    latent, image_cond = encode_video_with_vae(frames_np, vae, device)
    del vae
    torch.cuda.empty_cache()

    # ─── Step 3: Encode prompt with LLM text encoder ───────────────────
    print("\n=== Step 3: Encoding prompt with LLM ===")
    prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask = \
        encode_text_with_llm(args.prompt, args.model_path, device)

    # ─── Step 4: Encode with ByT5 ─────────────────────────────────────
    print("\n=== Step 4: Encoding with ByT5 ===")
    byt5_text_states, byt5_text_mask, neg_byt5_states, neg_byt5_mask = \
        encode_text_with_byt5(args.prompt, args.model_path, device)

    # ─── Step 5: Encode first frame with SigLIP ────────────────────────
    print("\n=== Step 5: Encoding first frame with SigLIP ===")
    vision_states = encode_image_with_siglip(
        first_frame_np, args.model_path, args.height, args.width, device
    )

    # ─── Step 6: Generate pose data ────────────────────────────────────
    print("\n=== Step 6: Generating pose data ===")
    training_pose = generate_training_pose_json(args.pose_string, t_latent)

    # ─── Step 7: Save everything ───────────────────────────────────────
    print("\n=== Step 7: Saving outputs ===")

    # Save main training data .pt
    latent_path = os.path.join(args.output_dir, "latents.pt")
    data = {
        "latent": latent,                      # [1, C, T_latent, H', W']
        "prompt_embeds": prompt_embeds,         # [1, seq_len, dim]
        "prompt_mask": prompt_mask,             # [1, seq_len]
        "image_cond": image_cond,              # [1, C, 1, H', W']
        "vision_states": vision_states,         # [1, num_patches, dim]
        "byt5_text_states": byt5_text_states,   # [1, text_len, dim]
        "byt5_text_mask": byt5_text_mask,       # [1, text_len]
    }
    torch.save(data, latent_path)
    print(f"  Saved latent data to {latent_path}")

    # Save pose JSON
    pose_path = os.path.join(args.output_dir, "pose.json")
    with open(pose_path, "w") as f:
        json.dump(training_pose, f, indent=2)
    print(f"  Saved pose data to {pose_path}")

    # Save negative prompt embeddings (needed by dataset for CFG training)
    neg_prompt_path = os.path.join(args.output_dir, "hunyuan_neg_prompt.pt")
    torch.save({
        "negative_prompt_embeds": neg_prompt_embeds,
        "negative_prompt_mask": neg_prompt_mask,
    }, neg_prompt_path)
    print(f"  Saved negative prompt to {neg_prompt_path}")

    neg_byt5_path = os.path.join(args.output_dir, "hunyuan_neg_byt5_prompt.pt")
    torch.save({
        "byt5_text_states": neg_byt5_states,
        "byt5_text_mask": neg_byt5_mask,
    }, neg_byt5_path)
    print(f"  Saved negative ByT5 to {neg_byt5_path}")

    # Save subsampled frames as images for inspection
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(frames_np.shape[0]):
        Image.fromarray(frames_np[i]).save(os.path.join(frames_dir, f"{i:04d}.png"))
    print(f"  Saved {frames_np.shape[0]} frames to {frames_dir}/")

    # Save metadata JSON (for the dataloader)
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    metadata = [
        {
            "latent_path": os.path.abspath(latent_path),
            "pose_path": os.path.abspath(pose_path),
        }
    ]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    # Print summary
    print("\n=== Preprocessing complete ===")
    print(f"  Video: {args.video_path} ({frames_np.shape[0]} frames → {t_latent} latent frames)")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Latent shape: {list(latent.shape)}")
    print(f"  Pose: {args.pose_string} ({len(training_pose)} video-frame entries)")
    print(f"  Output: {args.output_dir}")
    for key, val in data.items():
        print(f"    {key}: {list(val.shape)} ({val.dtype})")


if __name__ == "__main__":
    main()
