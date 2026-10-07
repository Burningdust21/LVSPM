import logging
import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

torch.serialization.add_safe_globals([getattr])
torch.serialization.add_safe_globals([torch.optim.lr_scheduler.OneCycleLR])
torch.autograd.set_detect_anomaly(False)
# Torch dynamo configs, requires Torch2.7+
torch._dynamo.config.capture_scalar_outputs = True
# The TTT module recompiles for validation and train seperatly
torch._dynamo.config.recompile_limit = 64
from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.encoder import EncoderLVSPM
from src.model.model_wrapper import ModelWrapper
from src.evaluation.evaluator import configure_evaluation
from src.utils import load_encoder_weights
from src.misc.validate_evaluation_index import sha256

app_logger = logging.getLogger(__name__)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    if cfg_dict.mode == "test":
        configure_evaluation(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    app_logger.info(cyan("Saving outputs to %s."), output_dir)
    if cfg.mode == "train":
        latest_run = output_dir.parents[1] / "latest-run"
        if latest_run.exists() or latest_run.is_symlink():
            latest_run.unlink()
        latest_run.symlink_to(output_dir)

    # Evaluation must not create training telemetry.
    callbacks = []
    if cfg.mode == "test":
        lightning_logger = False
    elif cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        lightning_logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        if cfg.mode == "train":
            callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        lightning_logger = LocalLogger(output_dir / "local", cfg_dict.tfboard_dir)

    # Evaluation is read-only with respect to model checkpoints.
    if cfg.mode == "train":
        callbacks.append(
            ModelCheckpoint(
                output_dir / "checkpoints",
                every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                save_top_k=cfg.checkpointing.save_top_k,
                monitor="info/global_step",
                mode="max",
            )
        )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    if cfg.mode == "test" and cfg.evaluation["runtime"]["strict_protocol"]:
        preset = cfg.evaluation["matrix"][cfg.evaluation["preset"]]
        expected = cfg.evaluation["checkpoints"][preset["checkpoint"]]["sha256"]
        if sha256(Path(checkpoint_path)) != expected:
            raise ValueError("Checkpoint does not match the selected evaluation preset.")

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    val_check_interval = (
        cfg.trainer.val_check_interval if cfg.mode == "train" else None
    )
    check_val_every_n_epoch = (
        None
        if cfg.mode == "train" and isinstance(val_check_interval, int)
        else 1
    )
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=lightning_logger,
        devices="auto",
        strategy="ddp",
        callbacks=callbacks,
        val_check_interval=val_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=-1,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_nodes=int(os.environ.get('NUM_NODE', 1)),
        log_every_n_steps=5,
        detect_anomaly=False,
        inference_mode=cfg.mode != "test",
    )
    seed = cfg_dict.seed if cfg.mode == "test" else cfg_dict.seed + trainer.global_rank
    seed_everything(seed)
    torch.manual_seed(seed)

    encoder = EncoderLVSPM(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        get_losses(cfg.loss),
        step_tracker,
        cfg_dict.compile_model,
        cfg.evaluation if cfg.mode == "test" else None,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
        evaluation_cfg=cfg.evaluation if cfg.mode == "test" else None,
    )
    if cfg.finetune_from is not None:
        app_logger.info("Finetuning model from: %s", cfg.finetune_from)
        state_dict_ori = torch.load(cfg.finetune_from, map_location=torch.device('cpu'), weights_only=True)["state_dict"]
        state_dict = {}
        for k,v in state_dict_ori.items():
            new_key = k.replace("_orig_mod.", "")
            state_dict[new_key] = v
        missing, existing = model_wrapper.load_state_dict(state_dict, strict=False)
        app_logger.info("Missing keys: %s, unexpected keys: %s", missing, existing)

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        load_encoder_weights(encoder, checkpoint_path)
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=None,
        )
        model_wrapper.finish_evaluation()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
