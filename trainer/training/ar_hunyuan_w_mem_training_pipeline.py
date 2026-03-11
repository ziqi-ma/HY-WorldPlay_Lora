# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.training.ar_hunyuan_mem_training_pipeline import TrainingPipeline
from trainer.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class HunyuanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Hunyuan.
    """
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        if training_args.eval_steps <= 0 or not training_args.gt_frames_dir:
            logger.info("AR eval disabled (eval_steps=%s, gt_frames_dir=%s)",
                        training_args.eval_steps, training_args.gt_frames_dir)
            return

        if not training_args.eval_pose_string:
            logger.info("AR eval disabled: eval_pose_string not set")
            return

        self.eval_enabled = True
        logger.info("AR eval enabled: every %d steps, GT from %s, pose='%s'",
                    training_args.eval_steps, training_args.gt_frames_dir,
                    training_args.eval_pose_string)

        # Import hyvideo.generate on ALL ranks — its module-level code calls
        # initialize_parallel_state(sp=WORLD_SIZE) which is a collective op.
        # Keep sp=WORLD_SIZE so all ranks participate in inference (sp=8).
        from hyvideo.generate import pose_to_input  # noqa: F401 — triggers collective init

        # Cache pose_to_input for later use in _log_ar_validation
        self._pose_to_input = pose_to_input

        # ALL ranks: initialize infer state and build inference pipeline
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
        self.eval_pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=training_args.model_path,
            transformer_version="480p_i2v",
            enable_offloading=True,
            enable_group_offloading=False,
            create_sr_pipeline=False,
            force_sparse_attn=False,
            transformer_dtype=torch.bfloat16,
            action_ckpt=training_args.ar_action_load_from_dir,
        )
        logger.info("Loaded inference pipeline for eval (rank %d)", self.global_rank)

        # Save base weights of LoRA target modules for restoring after merge
        self.eval_target_modules = training_args.lora_target_modules or [
            "img_attn_q", "img_attn_k", "img_attn_v", "img_attn_proj",
            "img_attn_prope_proj",
            "txt_attn_q", "txt_attn_k", "txt_attn_v", "txt_attn_proj",
            "linear1_q", "linear1_k", "linear1_v",
        ]
        self.eval_base_weights = {}
        for name, module in self.eval_pipe.transformer.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(t in name for t in self.eval_target_modules):
                    self.eval_base_weights[name] = module.weight.data.clone()
        logger.info("Saved %d base weight tensors for LoRA merge/restore",
                    len(self.eval_base_weights))

        # ALL ranks need the reference image path (used in pipe() call)
        gt_dir = training_args.gt_frames_dir
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
        self.eval_image_path = training_args.eval_image_path or os.path.join(gt_dir, gt_files[0])

        if self.global_rank == 0:
            # Load GT frames (rank 0 only — for metrics)
            from PIL import Image
            self.gt_frames = np.stack([
                np.array(Image.open(os.path.join(gt_dir, f)).convert('RGB'))
                for f in gt_files
            ])  # [T, H, W, 3] uint8
            logger.info("Loaded %d GT frames from %s", len(self.gt_frames), gt_dir)

            # LPIPS
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').cuda()
            self.lpips_fn.eval()

        self.sp_group.barrier()

    @torch.no_grad()
    def _log_ar_validation(self, step):
        """
        Full AR rollout evaluation using the inference pipeline
        (HunyuanVideo_1_5_Pipeline) with sp=8 (all ranks participate).

        ALL ranks: gather LoRA, broadcast, merge, run inference.
        Only rank 0: compute metrics, save video.
        """
        import wandb
        import torch.distributed as dist
        from trainer.training.training_utils import gather_state_dict_on_cpu_rank0

        training_args = self.training_args
        if not getattr(self, 'eval_enabled', False):
            return

        logger.info("Starting AR eval at step %d (sp=8, all ranks)", step)
        self.transformer.eval()

        # --- ALL ranks: Gather LoRA weights to rank 0 ---
        lora_state_dict = gather_state_dict_on_cpu_rank0(self.transformer, device=None)

        # --- ALL ranks: Restore base weights ---
        for name, module in self.eval_pipe.transformer.named_modules():
            if name in self.eval_base_weights:
                module.weight.data.copy_(self.eval_base_weights[name])

        # --- ALL ranks: Broadcast and merge LoRA weights ---
        lora_rank = training_args.lora_rank
        lora_alpha = training_args.lora_alpha
        scale = lora_alpha / lora_rank

        # Rank 0 has the gathered state dict; broadcast LoRA tensors to all ranks
        if self.global_rank == 0:
            # Collect LoRA key-value pairs that match target modules
            lora_pairs = {}
            for name, module in self.eval_pipe.transformer.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if not any(t in name for t in self.eval_target_modules):
                    continue
                lora_A = lora_B = None
                for a_key in [f"{name}.lora_A", f"{name}.lora_A.weight"]:
                    if a_key in lora_state_dict:
                        lora_A = lora_state_dict[a_key]
                        break
                for b_key in [f"{name}.lora_B", f"{name}.lora_B.weight"]:
                    if b_key in lora_state_dict:
                        lora_B = lora_state_dict[b_key]
                        break
                if lora_A is not None and lora_B is not None:
                    lora_pairs[name] = (lora_A, lora_B)
            logger.info("Gathered %d LoRA weight tensors, %d layer pairs to merge",
                        len(lora_state_dict), len(lora_pairs))
            if len(lora_pairs) == 0:
                logger.warning("No LoRA pairs matched! Check that gathered "
                               "state-dict keys align with eval-pipe module names. "
                               "First 5 state-dict keys: %s",
                               list(lora_state_dict.keys())[:5])
        else:
            lora_pairs = None

        # Broadcast the lora_pairs dict from rank 0 to all ranks
        lora_pairs_list = [lora_pairs]
        dist.broadcast_object_list(lora_pairs_list, src=0)
        lora_pairs = lora_pairs_list[0]

        # Merge on ALL ranks
        merged_count = 0
        for name, module in self.eval_pipe.transformer.named_modules():
            if name not in lora_pairs:
                continue
            lora_A, lora_B = lora_pairs[name]
            # Training forward computes x @ (B @ A), but nn.Linear computes
            # x @ W.T.  To match, we need W_new = W + scale*(B @ A).T so that
            # x @ W_new.T = x @ W.T + scale * x @ (B @ A).
            module.weight.data += scale * (
                lora_B.to(module.weight.device, dtype=module.weight.dtype)
                @ lora_A.to(module.weight.device, dtype=module.weight.dtype)
            ).T
            merged_count += 1
        logger.info("Rank %d: merged LoRA into %d layers (scale=%.2f)",
                    self.global_rank, merged_count, scale)

        # --- ALL ranks: Prepare pose/action data ---
        video_length = training_args.num_frames
        latent_num = (video_length - 1) // 4 + 1
        viewmats, Ks, action = self._pose_to_input(
            training_args.eval_pose_string, latent_num
        )

        # --- ALL ranks: Run full inference (sp=8) ---
        out = self.eval_pipe(
            enable_sr=False,
            prompt=training_args.eval_prompt,
            aspect_ratio="9:16",
            num_inference_steps=training_args.eval_num_inference_steps,
            sr_num_inference_steps=None,
            video_length=video_length,
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
            user_height=training_args.num_height,
            user_width=training_args.num_width,
            reference_image=self.eval_image_path,
        )

        # --- Rank 0 only: extract frames, compute metrics, save ---
        if self.global_rank == 0:
            video_tensor = out.videos  # [B, C, T, H, W], float [0, 1]
            if isinstance(video_tensor, torch.Tensor):
                video_np = video_tensor[0].cpu().numpy()
                if video_np.shape[0] == 3:  # [C, T, H, W]
                    video_np = np.transpose(video_np, (1, 2, 3, 0))  # [T, H, W, C]
                video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
            else:
                video_np = np.array(video_tensor)

            # --- Compute metrics ---
            gt = self.gt_frames
            num_frames = min(len(video_np), len(gt))
            gen = video_np[:num_frames]
            gt_eval = gt[:num_frames]

            if gen.shape[1:3] != gt_eval.shape[1:3]:
                import cv2
                h, w = gt_eval.shape[1], gt_eval.shape[2]
                gen = np.stack([
                    cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    for f in gen
                ])

            # Per-frame MSE and LPIPS
            gen_f = gen.astype(np.float32) / 255.0
            gt_f = gt_eval.astype(np.float32) / 255.0

            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            mse_scores = []
            lpips_scores = []
            for i in range(num_frames):
                mse_scores.append(float(np.mean((gen_f[i] - gt_f[i]) ** 2)))
                gen_t = to_tensor(gen[i]).unsqueeze(0).cuda() * 2 - 1
                gt_t = to_tensor(gt_eval[i]).unsqueeze(0).cuda() * 2 - 1
                lpips_scores.append(self.lpips_fn(gen_t, gt_t).item())

            l2_mean = float(np.mean(mse_scores))
            lpips_mean = float(np.mean(lpips_scores))

            logger.info("Eval step %d: MSE=%.6f, LPIPS=%.4f (%d frames)",
                        step, l2_mean, lpips_mean, num_frames)
            logger.info("  per-frame MSE : %s",
                        [f"{v:.5f}" for v in mse_scores])
            logger.info("  per-frame LPIPS: %s",
                        [f"{v:.4f}" for v in lpips_scores])
            wandb.log({
                "eval/l2_mse": l2_mean,
                "eval/lpips": lpips_mean,
                "eval/mse_first_frame": mse_scores[0],
                "eval/mse_last_frame": mse_scores[-1],
                "eval/lpips_first_frame": lpips_scores[0],
                "eval/lpips_last_frame": lpips_scores[-1],
            }, step=step)

            # Save video
            import imageio
            eval_dir = os.path.join(training_args.output_dir, "eval_videos")
            os.makedirs(eval_dir, exist_ok=True)
            video_path = os.path.join(eval_dir, f"eval_step_{step}.mp4")
            imageio.mimsave(video_path, list(video_np), fps=24)
            wandb.log({
                "eval/video": wandb.Video(video_path, caption=f"step {step}"),
            }, step=step)

        self.transformer.train()
        self.sp_group.barrier()


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = HunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    argv = sys.argv
    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
