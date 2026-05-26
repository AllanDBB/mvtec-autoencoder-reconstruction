import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
# PyTorch 2.6+ requires explicit allowlist for non-tensor globals in checkpoints
torch.serialization.add_safe_globals([DictConfig, ListConfig])

import pytorch_lightning as pl
import hydra
import wandb

from src.datasets.mvtec import MVTecDataModule
from src.lightning_module import AutoencoderModule


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    datamodule = MVTecDataModule(
        data_dir=cfg.data.data_dir,
        classes=list(cfg.data.classes),
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_val_split=cfg.data.train_val_split,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    module = AutoencoderModule(model_cfg=cfg.model, data_cfg=cfg.data)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = pl.loggers.WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity if cfg.logger.entity else None,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=cfg.logger.log_model,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    run_name = f"{cfg.model.name}_{cfg.model.loss}_z{cfg.model.latent_dim}"
    ckpt_dir = os.path.join(cfg.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        filename='best-{epoch:03d}-{val_loss:.4f}',
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, ckpt_path='best')

    wandb.finish()


if __name__ == '__main__':
    main()
