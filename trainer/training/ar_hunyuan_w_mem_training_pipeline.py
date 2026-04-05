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

    def _discover_eval_poses(self, training_args: TrainingArgs):
        """Build list of (name, pose_data) for evaluation.

        Sources (in priority order):
        1. eval_pose_dir — directory containing subdirs, each with a pose.json
        2. eval_pose_string — single pose string
        Returns a list of (name, pose_data) tuples where pose_data is either a
        file path (str ending in .json) or a pose string.
        """
        poses = []
        pose_dir = getattr(training_args, 'eval_pose_dir', '')
        if pose_dir and os.path.isdir(pose_dir):
            for sub in sorted(os.listdir(pose_dir)):
                pf = os.path.join(pose_dir, sub, 'pose.json')
                if os.path.isfile(pf):
                    poses.append((sub, pf))
        if not poses and training_args.eval_pose_string:
            poses.append(("pose", training_args.eval_pose_string))
        return poses

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        if training_args.eval_steps <= 0:
            logger.info("AR eval disabled (eval_steps=%s)", training_args.eval_steps)
            return

        self.eval_poses = self._discover_eval_poses(training_args)
        if not self.eval_poses:
            logger.info("AR eval disabled: no eval_pose_dir or eval_pose_string set")
            return

        self.eval_enabled = True
        pose_names = [n for n, _ in self.eval_poses]
        logger.info("AR eval enabled: every %d steps, %d pose(s): %s",
                    training_args.eval_steps, len(self.eval_poses), pose_names)

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
            "img_mlp", "txt_mlp",
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
        self.gt_frames = None
        self.lpips_fn = None
        if gt_dir:
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
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').cuda()
                self.lpips_fn.eval()
        else:
            self.eval_image_path = training_args.eval_image_path
            logger.info("gt_frames_dir not set — LPIPS/L2 metrics will be skipped")

        self.sp_group.barrier()

    @torch.no_grad()
    def _merge_lora_into_eval_pipe(self):
        """Gather LoRA weights, broadcast, and merge into eval pipeline.
        Must be called by ALL ranks. Returns the number of merged layers."""
        import torch.distributed as dist
        from trainer.training.training_utils import gather_state_dict_on_cpu_rank0

        training_args = self.training_args

        # Restore base weights on ALL ranks
        for name, module in self.eval_pipe.transformer.named_modules():
            if name in self.eval_base_weights:
                module.weight.data.copy_(self.eval_base_weights[name])

        # Gather LoRA weights to rank 0
        lora_state_dict = gather_state_dict_on_cpu_rank0(self.transformer, device=None)

        lora_rank = training_args.lora_rank
        lora_alpha = training_args.lora_alpha
        scale = lora_alpha / lora_rank

        if self.global_rank == 0:
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
                logger.warning("No LoRA pairs matched! First 5 state-dict keys: %s",
                               list(lora_state_dict.keys())[:5])
        else:
            lora_pairs = None

        lora_pairs_list = [lora_pairs]
        dist.broadcast_object_list(lora_pairs_list, src=0)
        lora_pairs = lora_pairs_list[0]

        merged_count = 0
        for name, module in self.eval_pipe.transformer.named_modules():
            if name not in lora_pairs:
                continue
            lora_A, lora_B = lora_pairs[name]
            module.weight.data += scale * (
                lora_B.to(module.weight.device, dtype=module.weight.dtype)
                @ lora_A.to(module.weight.device, dtype=module.weight.dtype)
            )
            merged_count += 1
        logger.info("Rank %d: merged LoRA into %d layers (scale=%.2f)",
                    self.global_rank, merged_count, scale)
        return merged_count

    def _run_eval_single_pose(self, pose_name, pose_data, step):
        """Run inference for a single pose on ALL ranks. Rank 0 saves video."""
        import wandb

        training_args = self.training_args
        video_length = training_args.num_frames
        latent_num = (video_length - 1) // 4 + 1

        # pose_to_input handles both w2c/intrinsic and extrinsic/K formats,
        # and both latent-frame (16) and video-frame (61) entry counts.
        viewmats, Ks, action = self._pose_to_input(pose_data, latent_num)

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

        if self.global_rank == 0:
            video_tensor = out.videos
            if isinstance(video_tensor, torch.Tensor):
                video_np = video_tensor[0].cpu().numpy()
                if video_np.shape[0] == 3:
                    video_np = np.transpose(video_np, (1, 2, 3, 0))
                video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
            else:
                video_np = np.array(video_tensor)

            import imageio
            eval_dir = os.path.join(training_args.output_dir, "eval_videos")
            os.makedirs(eval_dir, exist_ok=True)
            video_path = os.path.join(eval_dir, f"eval_step_{step}_{pose_name}.mp4")
            imageio.mimsave(video_path, list(video_np), fps=24)
            logger.info("Saved eval video: %s", video_path)
            wandb.log({
                f"eval/{pose_name}": wandb.Video(video_path,
                                                  caption=f"step {step} {pose_name}"),
            }, step=step)

    def _log_ar_validation(self, step):
        """
        Full AR rollout evaluation on all discovered eval poses.
        ALL ranks participate in inference (sp-parallel).
        """
        if not getattr(self, 'eval_enabled', False):
            return

        pose_names = [n for n, _ in self.eval_poses]
        logger.info("Starting AR eval at step %d on %d pose(s): %s",
                    step, len(self.eval_poses), pose_names)
        self.transformer.eval()

        self._merge_lora_into_eval_pipe()

        for pose_name, pose_data in self.eval_poses:
            logger.info("Evaluating pose '%s' at step %d", pose_name, step)
            self._run_eval_single_pose(pose_name, pose_data, step)
            self.sp_group.barrier()

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
