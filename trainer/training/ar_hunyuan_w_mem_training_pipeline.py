# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.distributed as dist
import wandb

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.training.ar_hunyuan_mem_training_pipeline import TrainingPipeline, TrainingBatch
from trainer.dataset import build_ar_camera_hunyuan_w_mem_dataloader
from trainer.utils import is_vsa_available
from trainer.distributed import get_local_torch_device
from trainer.forward_context import set_forward_context

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
        """Build one-batch DataLoaders for seen and unseen validation examples."""
        self.val_seen_iter = None
        self.val_unseen_iter = None

        if training_args.val_seen_json_path:
            _, seen_loader = build_ar_camera_hunyuan_w_mem_dataloader(
                json_path=training_args.val_seen_json_path,
                causal=training_args.causal,
                window_frames=training_args.window_frames,
                batch_size=training_args.train_batch_size,
                num_data_workers=0,
                drop_last=False,
                drop_first_row=False,
                seed=self.seed,
                cfg_rate=0.0,   # never drop conditioning for eval
                i2v_rate=training_args.i2v_rate,
                neg_prompt_path=training_args.neg_prompt_path,
                neg_byt5_path=training_args.neg_byt5_path,
            )
            self.val_seen_iter = self._cycle(seen_loader)
            logger.info("Validation seen loader ready: %s", training_args.val_seen_json_path)

        if training_args.val_unseen_json_path:
            _, unseen_loader = build_ar_camera_hunyuan_w_mem_dataloader(
                json_path=training_args.val_unseen_json_path,
                causal=training_args.causal,
                window_frames=training_args.window_frames,
                batch_size=training_args.train_batch_size,
                num_data_workers=0,
                drop_last=False,
                drop_first_row=False,
                seed=self.seed,
                cfg_rate=0.0,
                i2v_rate=training_args.i2v_rate,
                neg_prompt_path=training_args.neg_prompt_path,
                neg_byt5_path=training_args.neg_byt5_path,
            )
            self.val_unseen_iter = self._cycle(unseen_loader)
            logger.info("Validation unseen loader ready: %s", training_args.val_unseen_json_path)

    @staticmethod
    def _cycle(loader):
        while True:
            for batch in loader:
                yield batch

    @torch.no_grad()
    def _eval_loss_on_batch(self, raw_batch: dict) -> float:
        """Load one raw dataset batch, run a forward pass, return the loss scalar."""
        device = get_local_torch_device()
        tb = TrainingBatch()
        tb.current_timestep = 0  # unused by loss computation
        tb.current_vsa_sparsity = 0.0

        # Replicate what _get_next_batch does for action-mode batches
        tb.latents        = raw_batch["latent"].to(device, dtype=torch.bfloat16)
        tb.prompt_embed   = raw_batch["prompt_embed"].to(device, dtype=torch.bfloat16)
        tb.w2c            = raw_batch["w2c"].to(device, dtype=torch.bfloat16)
        tb.intrinsic      = raw_batch["intrinsic"].to(device, dtype=torch.bfloat16)
        tb.action         = raw_batch["action"].to(device, dtype=torch.bfloat16)
        tb.image_cond     = raw_batch["image_cond"].to(device, dtype=torch.bfloat16)
        tb.vision_states  = raw_batch["vision_states"].to(device, dtype=torch.bfloat16)
        tb.prompt_mask    = raw_batch["prompt_mask"].to(device, dtype=torch.bfloat16)
        tb.byt5_text_states = raw_batch["byt5_text_states"].to(device, dtype=torch.bfloat16)
        tb.byt5_text_mask   = raw_batch["byt5_text_mask"].to(device, dtype=torch.bfloat16)
        tb.select_window_out_flag = raw_batch["select_window_out_flag"]
        tb.i2v_mask       = raw_batch["i2v_mask"].to(device, dtype=torch.bfloat16)
        tb.video_path     = raw_batch["video_path"][0]

        tb = self._prepare_ar_dit_inputs(tb)
        tb = self._build_input_kwargs(tb)

        input_kwargs = tb.input_kwargs
        with set_forward_context(current_timestep=0, attn_metadata=None):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model_pred = self.transformer(**input_kwargs)[0]

        target = tb.noise - tb.latents
        i2v_mask = tb.i2v_mask
        diff = (model_pred.float() * i2v_mask - target.float() * i2v_mask) ** 2
        loss = diff.sum() / max(i2v_mask.sum(), 1)

        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        return avg_loss.item()

    def _save_temporal_embed_for_eval(self, output_path: str) -> None:
        """Gather temporal embed/token weight and save to safetensors on rank 0."""
        weights = {}
        for attr in ['temporal_frame_embed', 'temporal_token_embed']:
            if not hasattr(self.transformer, attr):
                continue
            weight = getattr(self.transformer, attr).weight
            try:
                from torch.distributed.tensor import DTensor
                if isinstance(weight, DTensor):
                    weight = weight.full_tensor()
            except (ImportError, Exception):
                pass
            weights[f"{attr}.weight"] = weight.detach().cpu()
        if hasattr(self.transformer, 'temporal_frame_embed_blocks'):
            for i, emb in enumerate(self.transformer.temporal_frame_embed_blocks):
                weight = emb.weight
                try:
                    from torch.distributed.tensor import DTensor
                    if isinstance(weight, DTensor):
                        weight = weight.full_tensor()
                except (ImportError, Exception):
                    pass
                weights[f"temporal_frame_embed_blocks.{i}.weight"] = weight.detach().cpu()
        if self.global_rank == 0 and weights:
            from safetensors.torch import save_file
            save_file(weights, output_path)

    def _maybe_eval(self, step: int) -> None:
        is_temporal_training = (
            self.training_args.temporal_embed_training or
            getattr(self.training_args, 'temporal_token_training', False) or
            getattr(self.training_args, 'temporal_embed_per_block_training', False)
        )
        if not is_temporal_training:
            return
        eval_trajectories = []
        if self.training_args.eval_pose_json and self.training_args.eval_image_path:
            eval_trajectories.append(("seen", self.training_args.eval_pose_json, self.training_args.eval_image_path))
        unseen_pose = getattr(self.training_args, 'eval_pose_json_unseen', '')
        unseen_img  = getattr(self.training_args, 'eval_image_path_unseen', '')
        if unseen_pose and unseen_img:
            eval_trajectories.append(("unseen", unseen_pose, unseen_img))
        if not eval_trajectories:
            return

        import subprocess
        import datetime

        # Extend NCCL timeout so ranks 1-3 can wait for eval subprocess(es)
        _eval_timeout = datetime.timedelta(hours=4)
        _default_timeout = datetime.timedelta(minutes=10)
        torch.distributed.distributed_c10d._set_pg_timeout(_eval_timeout)

        # Save temporal embed weights (collective op — all ranks participate)
        tmp_dir = os.path.join(self.training_args.output_dir, "_eval_tmp")
        if self.global_rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
        dist.barrier()

        tmp_ckpt = os.path.join(tmp_dir, "diffusion_pytorch_model.safetensors")
        self._save_temporal_embed_for_eval(tmp_ckpt)
        dist.barrier()  # ensure rank 0 has written before subprocess reads

        # Move training model to CPU on all ranks to free GPU memory for eval subprocess.
        # plain .cpu() is not enough for FSDP2: _sharded_param_data inside each
        # FSDPParam still holds a GPU tensor.  Walk the module tree and free it.
        self.transformer.cpu()
        try:
            import gc
            from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
            for module in self.transformer.modules():
                state = _get_module_fsdp_state(module)
                if state is None or state._fsdp_param_group is None:
                    continue
                for fsdp_param in state._fsdp_param_group.fsdp_params:
                    if hasattr(fsdp_param, '_sharded_param_data') and \
                            fsdp_param._sharded_param_data is not None and \
                            fsdp_param._sharded_param_data.is_cuda:
                        fsdp_param._sharded_param_data = fsdp_param._sharded_param_data.cpu()
                    if hasattr(fsdp_param, '_sharded_post_forward_param_data') and \
                            fsdp_param._sharded_post_forward_param_data is not None and \
                            fsdp_param._sharded_post_forward_param_data.is_cuda:
                        fsdp_param._sharded_post_forward_param_data = \
                            fsdp_param._sharded_post_forward_param_data.cpu()
            gc.collect()
        except Exception as e:
            logger.warning("FSDP2 explicit GPU free failed (non-fatal): %s", e)
        torch.cuda.empty_cache()
        dist.barrier()  # all ranks offloaded before rank 0 launches subprocess

        # All ranks wait here while rank 0 drives the eval subprocess(es).
        # Ranks 1-3 sit at the final barrier; TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
        # in the training script is set high enough to cover eval time.
        if self.global_rank == 0:
            action_ckpt = self.training_args.eval_action_ckpt or ""
            num_eval_gpus = len(self.training_args.eval_gpus.split(","))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.training_args.eval_gpus
            log = {}
            import json as _json

            for tag, pose_json, image_path in eval_trajectories:
                out_dir = os.path.join(self.training_args.output_dir, f"eval_step{step}_{tag}")
                os.makedirs(out_dir, exist_ok=True)
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_eval_gpus}",
                    "--master_port=29799",
                    "scripts/eval/run_eval_temporal_embed.py",
                    "--model_path", self.training_args.pretrained_model_name_or_path,
                    "--action_ckpt", action_ckpt,
                    "--temporal_embed_ckpt", tmp_ckpt,
                    "--pose_json", pose_json,
                    "--image_path", image_path,
                    "--output_dir", out_dir,
                    "--num_inference_steps", "30",
                    "--seed", "42",
                ]
                logger.info("step %d  launching %s video eval on GPUs %s", step, tag, self.training_args.eval_gpus)
                result = subprocess.run(cmd, env=env)

                if result.returncode == 0:
                    video_path = os.path.join(out_dir, "gen.mp4")
                    if os.path.exists(video_path):
                        log[f"eval_{tag}_video"] = wandb.Video(video_path)
                    metrics_path = os.path.join(out_dir, "metrics.json")
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            for k, v in _json.load(f).items():
                                log[f"eval_{tag}_{k}"] = v
                else:
                    logger.warning("step %d  %s eval subprocess exited with code %d", step, tag, result.returncode)

            if log:
                wandb.log(log, step=step)
                logger.info("step %d  logged eval metrics to wandb: %s", step, log)

        # Reload training model back onto GPU on all ranks.
        self.transformer.cuda()
        dist.barrier()
        torch.distributed.distributed_c10d._set_pg_timeout(_default_timeout)


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
