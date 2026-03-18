"""
Quick test: load model with temporal_embed_per_block_training,
then immediately call save_checkpoint (no training steps).
Run via test_checkpoint_save.sh.
"""
import sys
import os

# repo root on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.multiprocessing as mp

from trainer.trainer_args import TrainingArgs, TrainerArgs
from trainer.utils import FlexibleArgumentParser
from trainer.logger import init_logger
from trainer.training.training_utils import save_checkpoint

logger = init_logger(__name__)


def main(args):
    from trainer.training.ar_hunyuan_w_mem_training_pipeline import HunyuanTrainingPipeline

    logger.info("Loading pipeline (model + FSDP + temporal embed params)...")
    pipeline = HunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)

    logger.info("Pipeline loaded. Calling save_checkpoint immediately at step 0...")
    save_checkpoint(
        pipeline.transformer,
        pipeline.global_rank,
        "outputs/test_checkpoint_save",
        step=0,
        optimizer=pipeline.optimizer,
        dataloader=None,
        scheduler=pipeline.lr_scheduler,
        noise_generator=None,
    )
    logger.info("save_checkpoint succeeded!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    from trainer.trainer_args import TrainingArgs, TrainerArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
