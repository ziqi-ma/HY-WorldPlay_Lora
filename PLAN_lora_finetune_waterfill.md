# Plan: LoRA Finetuning of HunyuanVideo AR Model on Waterfill Video

## Goal
LoRA finetune the **HunyuanVideo 1.5 normal AR model** (`HunyuanTransformer3DARActionModel`) on the video `/data/ziqi/Repos/HY-WorldPlay/outputs/waterfill_d3a3/veo_output_to23.mp4` with action `"right-2"`, using 8 GPUs in the HY-WorldPlay-New training framework.

---

## Why This Is the Simplest Option

The training framework (`trainer/`) is already built for this exact model:
- `HunyuanTransformer3DARActionModel` is **already registered** in `trainer/models/registry.py`
- `_build_input_kwargs()` already builds the correct forward kwargs
- The dataset class already expects the right fields (latent, prompt_embeds, vision_states, byt5, etc.)
- The training launch script `run_ar_hunyuan_action_mem.sh` already exists as a template
- **No training pipeline modifications needed** — only a preprocessing script and launch script

---

## Model: HunyuanVideo 1.5 Normal AR

| Property | Value |
|----------|-------|
| Class | `HunyuanTransformer3DARActionModel` |
| Architecture | Dual-stream: 20 `MMDoubleStreamBlock` + 40 `MMSingleStreamBlock` |
| Hidden dim | 3072 (24 heads × 128 head_dim) |
| Text encoders | LLM (Qwen2.5-VL-7B) + ByT5 |
| Vision encoder | SigLIP (from FLUX.1-Redux-dev) |
| VAE | `AutoencoderKLConv3D` (HunyuanVideo VAE) |
| Checkpoint | `ar_model/diffusion_pytorch_model.safetensors` (~32GB) |
| Total params | ~13B |
| Inference | 50 steps (`--few_step false`) |

---

## Overview

Two deliverables:
1. **Preprocessing script** (`preprocess_video.py`) — encodes video + prompt + action into `.pt` + pose JSON
2. **Training launch script** (`run_lora_waterfill.sh`) — configures and launches LoRA finetuning

---

## Step 1: Create Preprocessing Script

**File to create:** `/data/ziqi/Repos/HY-WorldPlay-New/preprocess_video.py`

This script will:
1. Load the video from MP4, extract frames, resize to 432×768 (portrait 9:16, matching `run_waterfill.sh`)
2. Load the HunyuanVideo VAE and encode all frames into latents
3. Load the text encoders (LLM + ByT5) and encode the prompt (empty string)
4. Load the vision encoder (SigLIP) and encode the first frame
5. Encode the first frame with the VAE for `image_cond`
6. Generate camera pose data from `"right-2"` using existing code
7. Save everything to `.pt` + pose `.json` + metadata `.json`

### Components to load

| Component | Source | Purpose |
|-----------|--------|---------|
| VAE | `HunyuanVideo-1.5/vae/` | Encode video frames → latents |
| LLM text encoder | `HunyuanVideo-1.5/text_encoder/llm/` (Qwen2.5-VL-7B) | Text embeddings |
| ByT5 text encoder | `HunyuanVideo-1.5/text_encoder/byt5-small/` + `Glyph-SDXL-v2/` | Additional text embeddings |
| Vision encoder | `HunyuanVideo-1.5/vision_encoder/siglip/` | First-frame image features |
| Pose utils | `hyvideo/generate.py` | `pose_string_to_json()`, `pose_to_input()` |

Can reuse `HunyuanVideo_1_5_Pipeline.create_pipeline()` from `hyvideo/pipelines/worldplay_video_pipeline.py:1716` to load all encoders at once.

### Output format

Matches `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`:

```python
# .pt file contains:
{
    "latent": tensor,            # [1, T_latent, 16, H', W'] — VAE-encoded video
    "prompt_embeds": tensor,     # [1, seq_len, 4096] — LLM text embeddings
    "prompt_mask": tensor,       # [1, seq_len] — text attention mask
    "image_cond": tensor,        # [1, C, 1, H', W'] — first frame VAE latent
    "vision_states": tensor,     # [1, num_patches, hidden_dim] — SigLIP features
    "byt5_text_states": tensor,  # [1, text_len, dim] — ByT5 text embeddings
    "byt5_text_mask": tensor,    # [1, text_len] — ByT5 attention mask
}
```

```json
// pose.json — per-frame camera extrinsics/intrinsics:
{
    "0": {"extrinsic": [[4x4 matrix]], "K": [[3x3 matrix]]},
    "4": {"extrinsic": [[4x4 matrix]], "K": [[3x3 matrix]]},
    ...
}
```

```json
// metadata.json — dataset index:
[
    {
        "latent_path": ".../training_data/waterfill_right2/latents.pt",
        "pose_path": ".../training_data/waterfill_right2/pose.json"
    }
]
```

### Key preprocessing details

- **Resolution**: 432×768 (portrait 9:16, matching original waterfill video)
- **VAE compression**: 4× temporal, 8× spatial → latent shape `[1, T_latent, 16, 54, 96]`
- **Frame count**: Must be `4*N + 1` for VAE temporal compression (e.g., 45 frames → 12 latent frames)
- **Action** `"right-2"` = yaw right for 2 seconds at 6fps = 12 frames of camera motion
- **Action encoding**: yaw right → `rotate_one_label=1`, `trans_one_label=0` → `action_label = 0*9 + 1 = 1`
- Requires 1 GPU

---

## Step 2: Create LoRA Training Launch Script

**File to create:** `/data/ziqi/Repos/HY-WorldPlay-New/scripts/training/hyvideo15/run_lora_waterfill.sh`

Based on the existing `run_ar_hunyuan_action_mem.sh`, with LoRA-specific modifications.

### Training configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `NUM_GPUS` | 8 | |
| `sp_size` | 4 | 2 sequence parallel groups |
| `--cls_name` | `"HunyuanTransformer3DARActionModel"` | Already registered in trainer |
| `--lora_training` | (flag) | Enable LoRA mode |
| `--lora_rank` | 16 | Low-rank dimension |
| `--lora_alpha` | 32 | Scaling factor (alpha/rank = 2) |
| `--lora_target_modules` | see below | Attention layers in MMDoubleStreamBlock |
| `--learning_rate` | 1e-4 | Higher than full finetune |
| `--max_train_steps` | 500 | Single video, keep short |
| `--checkpointing_steps` | 100 | Save every 100 steps |
| `--train_batch_size` | 1 | |
| `--gradient_accumulation_steps` | 1 | |
| `--window_frames` | 24 | Latent window size for AR training |
| `--num_frames` | 45 | Match the video frame count |
| `--num_height` | 768 | Portrait orientation |
| `--num_width` | 432 | Portrait orientation |

### LoRA target modules

The model has 20 `MMDoubleStreamBlock` blocks with these attention Linear layers:

```
double_blocks.N.img_attn_q      [3072 → 3072]   # Image stream query
double_blocks.N.img_attn_k      [3072 → 3072]   # Image stream key
double_blocks.N.img_attn_v      [3072 → 3072]   # Image stream value
double_blocks.N.img_attn_proj   [3072 → 3072]   # Image stream output
double_blocks.N.txt_attn_q      [3072 → 3072]   # Text stream query
double_blocks.N.txt_attn_kv     [3072 → 6144]   # Text stream key+value (combined)
double_blocks.N.txt_attn_proj   [3072 → 3072]   # Text stream output
```

The `LoRAPipeline.is_target_layer()` uses substring matching. The **default** target modules (`["q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out", "to_qkv"]`) **won't match** these HunyuanVideo layer names. We **must** pass the correct names:

```bash
--lora_target_modules img_attn_q img_attn_k img_attn_v img_attn_proj txt_attn_q txt_attn_kv txt_attn_proj
```

### Model paths

```bash
MODEL_PATH=/data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_DIR=/data/ziqi/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/95036f76df1e446fd046765ddadb868b84b05d8e/ar_model/
```

### LoRA parameter count estimate

With LoRA rank=16 on attention layers in 20 double-stream blocks (7 layers × 20 blocks = 140 LoRA pairs):
- Per LoRA pair (3072 dim): `3072 * 16 * 2 = 98,304` params
- `txt_attn_kv` is wider (6144 out): `(3072 + 6144) * 16 = 147,456` per pair → 20 pairs
- Total: ~**14.6M** trainable params (vs ~13B total → ~0.11%)

---

## Step 3: Verify LoRA CLI Argument Handling

**File to check:** `trainer/trainer_args.py`

```python
lora_rank: int | None = None
lora_alpha: int | None = None
lora_training: bool = False
lora_target_modules: list[str] | None = None
```

Need to verify that `FlexibleArgumentParser` properly parses `--lora_target_modules` as a list when passed as space-separated args. If not, may need repeated flags or comma-separated format.

---

## Step 4: Potential Issues & Mitigations

### 4a. Single-video training
With only one video, the dataloader loops over the same sample. Expected for LoRA finetuning:
- Higher learning rate (1e-4)
- Short training (100-500 steps)
- Monitor loss for convergence

### 4b. Action label encoding
Action `"right-2"` = yaw-right rotation:
- Translation: `(0,0,0,0)` → label 0
- Rotation: `(1,0,0,0)` → label 1 (yaw right)
- Combined: `0 * 9 + 1 = 1` for all frames

### 4c. Video frame count alignment
The video must have `4*N + 1` frames for VAE temporal compression. If the source video has a different count, subsample/pad to nearest valid count (e.g., 45 frames).

### 4d. Memory requirements
The full AR model is ~13B params. With LoRA (only ~15M trainable), the base model is frozen but still loaded in memory. With 8 GPUs + FSDP sharding + bf16, this should fit (~3.2GB/GPU for the model + gradients for LoRA only).

---

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `preprocess_video.py` | **Create** | Video → VAE latents + text embeddings + pose data |
| `training_data/waterfill_right2/` | **Create dir** | Output directory for preprocessed data |
| `scripts/training/hyvideo15/run_lora_waterfill.sh` | **Create** | LoRA training launch script |

No modifications to existing training pipeline files needed.

---

## Execution Order

```
1. python preprocess_video.py \
     --video_path /data/ziqi/Repos/HY-WorldPlay/outputs/waterfill_d3a3/veo_output_to23.mp4 \
     --pose_string "right-2" \
     --prompt "" \
     --output_dir training_data/waterfill_right2/ \
     --model_path /data/ziqi/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038

2. bash scripts/training/hyvideo15/run_lora_waterfill.sh

3. LoRA weights saved to: outputs/lora_waterfill_right2/checkpoint-{step}/
```

---

## Verification

1. **Preprocessing**: Check `.pt` file loads correctly:
   ```python
   data = torch.load("training_data/waterfill_right2/latents.pt")
   print(data["latent"].shape)        # [1, T_latent, 16, 54, 96]
   print(data["prompt_embeds"].shape)  # [1, seq_len, 4096]
   print(data["vision_states"].shape)  # [1, num_patches, hidden_dim]
   ```

2. **Training**: Monitor stdout/wandb for loss convergence

3. **Inference**: Load LoRA weights back into the AR model and run with `run_waterfill.sh` pointing to the finetuned checkpoint

---

## Architecture Reference

### MMDoubleStreamBlock (×20) — LoRA targets

```
img_attn_q      [3072 → 3072]   # Image attention query
img_attn_k      [3072 → 3072]   # Image attention key
img_attn_v      [3072 → 3072]   # Image attention value
img_attn_proj   [3072 → 3072]   # Image attention output
img_mlp.fc1     [3072 → 12288]  # Image MLP (not targeted)
img_mlp.fc2     [12288 → 3072]  # Image MLP (not targeted)
txt_attn_q      [3072 → 3072]   # Text attention query
txt_attn_kv     [3072 → 6144]   # Text attention key+value
txt_attn_proj   [3072 → 3072]   # Text attention output
txt_mlp.fc1     [3072 → 12288]  # Text MLP (not targeted)
txt_mlp.fc2     [12288 → 3072]  # Text MLP (not targeted)
```

### MMSingleStreamBlock (×40) — NOT targeted by LoRA

These use fused `linear1` and `linear2` layers that combine attention + MLP. Could optionally be targeted for more capacity but increases trainable params significantly.
