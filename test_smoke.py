"""
Smoke test: generates fake MVTec structure with 4 images and runs 2 epochs.
Run in Colab before the full training loop to catch errors fast.

Usage:
    !python test_smoke.py
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf import DictConfig, ListConfig
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
torch.set_float32_matmul_precision('high')

from PIL import Image
import numpy as np
import pytorch_lightning as pl
import wandb

from src.datasets.mvtec import MVTecDataModule
from src.lightning_module import AutoencoderModule


def make_fake_mvtec(root, classes=('cable',), n_train=4, n_test=4):
    for cls in classes:
        train_good = os.path.join(root, cls, 'train', 'good')
        test_good  = os.path.join(root, cls, 'test',  'good')
        test_bad   = os.path.join(root, cls, 'test',  'broken')
        for d in (train_good, test_good, test_bad):
            os.makedirs(d, exist_ok=True)
        for i in range(n_train):
            arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(train_good, f'{i:03d}.png'))
        for i in range(n_test // 2):
            arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(test_good, f'{i:03d}.png'))
            Image.fromarray(arr).save(os.path.join(test_bad,  f'{i:03d}.png'))


def run_smoke(model_name='vae', loss='l1'):
    tmp = tempfile.mkdtemp()
    data_dir  = os.path.join(tmp, 'mvtec')
    ckpt_dir  = os.path.join(tmp, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    make_fake_mvtec(data_dir)

    model_cfg = OmegaConf.create({
        'name': model_name,
        'latent_dim': 8,
        'loss': loss,
        'beta': 1.0,
        'lr': 1e-4,
        'encoder_channels': [8, 16],
    })
    data_cfg = OmegaConf.create({
        'image_size': 128,
        'batch_size': 2,
        'num_workers': 0,
        'classes': ['cable'],
        'train_val_split': 0.75,
    })

    pl.seed_everything(42)

    datamodule = MVTecDataModule(
        data_dir=data_dir,
        classes=['cable'],
        image_size=128,
        batch_size=2,
        num_workers=0,
        train_val_split=0.75,
    )

    module = AutoencoderModule(model_cfg=model_cfg, data_cfg=data_cfg)

    run_name = f"smoke_{model_name}_{loss}"
    ckpt_path = os.path.join(ckpt_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_path,
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        filename='best-{epoch:03d}',
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        logger=False,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, ckpt_path='best')

    shutil.rmtree(tmp)
    print(f"\n✓ SMOKE PASS: {model_name} + {loss}")


if __name__ == '__main__':
    combos = [
        ('vae',  'l1'),
        ('vae',  'l2'),
        ('vae',  'ssim'),
        ('vae',  'ssim_l1'),
        ('unet', 'l1'),
        ('unet', 'l2'),
        ('unet', 'ssim'),
        ('unet', 'ssim_l1'),
    ]
    failed = []
    for model, loss in combos:
        try:
            run_smoke(model, loss)
        except Exception as e:
            print(f"\n✗ FAIL: {model} + {loss}: {e}")
            failed.append((model, loss))

    print("\n--- RESULTS ---")
    print(f"Passed: {len(combos) - len(failed)}/{len(combos)}")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
    else:
        print("All good.")
