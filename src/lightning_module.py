import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import wandb
from sklearn.manifold import TSNE

from src.models.vae import VAE
from src.models.unet_ae import UNetAE
from src.losses.losses import ReconstructionLoss


class AutoencoderModule(pl.LightningModule):
    def __init__(self, model_cfg, data_cfg):
        super().__init__()
        self.save_hyperparameters(ignore=['model_cfg', 'data_cfg'])
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

        if model_cfg.name == 'vae':
            self.model = VAE(
                latent_dim=model_cfg.latent_dim,
                encoder_channels=list(model_cfg.encoder_channels),
            )
            self.is_vae = True
        else:
            self.model = UNetAE(
                latent_dim=model_cfg.latent_dim,
                encoder_channels=list(model_cfg.encoder_channels),
            )
            self.is_vae = False

        self.loss_fn = ReconstructionLoss(model_cfg.loss)
        self.beta = float(getattr(model_cfg, 'beta', 1.0))

        # Accumulation buffers cleared in on_X_epoch_end
        self._val_x = None
        self._val_recon = None
        self._val_latents = []
        self._val_labels = []
        self._test_results = []

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _forward(self, x):
        if self.is_vae:
            recon, mu, logvar = self.model(x)
            recon_loss = self.loss_fn(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl
            latent = mu
            return loss, recon, recon_loss, kl, latent
        else:
            recon, z = self.model(x)
            loss = self.loss_fn(recon, x)
            return loss, recon, loss, torch.tensor(0.0, device=x.device), z

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, *_ = batch
        loss, _, recon_loss, kl, _ = self._forward(x)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', recon_loss, on_step=False, on_epoch=True)
        if self.is_vae:
            self.log('train/kl_loss', kl, on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        x, labels, *_ = batch
        loss, recon, recon_loss, kl, latent = self._forward(x)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        self._val_latents.append(latent.detach().cpu())
        self._val_labels.append(labels.cpu())

        if batch_idx == 0:
            n = min(16, x.size(0))
            self._val_x = x[:n].detach().cpu()
            self._val_recon = recon[:n].detach().cpu()

        return loss

    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        if self._val_x is not None:
            self._log_recon_grid(self._val_x, self._val_recon, tag='val/reconstructions')

        # t-SNE on test set every 5 epochs
        if (epoch % 5 == 0 or epoch == self.trainer.max_epochs - 1):
            self._log_tsne_on_test(epoch)

        self._val_latents.clear()
        self._val_labels.clear()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        x, labels, defect_types, classes = batch
        _, recon, _, _, latent = self._forward(x)

        per_img_err = torch.mean((recon - x) ** 2, dim=[1, 2, 3]).detach().cpu()

        self._test_results.append({
            'errors': per_img_err,
            'labels': labels.cpu(),
            'defect_types': list(defect_types),
            'classes': list(classes),
        })

        if batch_idx == 0:
            good_mask = labels == 0
            bad_mask = labels == 1
            imgs, recons = [], []
            if good_mask.any():
                imgs.append(x[good_mask][:8].detach().cpu())
                recons.append(recon[good_mask][:8].detach().cpu())
            if bad_mask.any():
                imgs.append(x[bad_mask][:8].detach().cpu())
                recons.append(recon[bad_mask][:8].detach().cpu())
            if imgs:
                combined = torch.cat(imgs, dim=0)[:16]
                combined_r = torch.cat(recons, dim=0)[:16]
                self._log_recon_grid(combined, combined_r, tag='test/reconstructions_good_bad')

    def on_test_epoch_end(self):
        pass  # Histograms computed in the notebook for full flexibility

    def get_test_results(self):
        return self._test_results

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _log_recon_grid(self, originals, reconstructions, tag, n=16):
        n = min(n, originals.size(0))
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        if n == 1:
            axes = axes[:, None]
        for i in range(n):
            axes[0, i].imshow(originals[i].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).clamp(0, 1).numpy())
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=9)
        axes[1, 0].set_ylabel('Reconstruida', fontsize=9)
        plt.suptitle(f'Epoch {self.current_epoch} — {self.model_cfg.name.upper()} ({self.model_cfg.loss})')
        plt.tight_layout()
        if wandb.run is not None:
            wandb.log({tag: wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)

    def _log_tsne_on_test(self, epoch):
        if self.trainer.datamodule is None:
            return
        test_dl = self.trainer.datamodule.test_dataloader()
        latents, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in test_dl:
                x, lbl, *_ = batch
                x = x.to(self.device)
                if self.is_vae:
                    mu, _ = self.model.encode(x)
                    latents.append(mu.cpu())
                else:
                    z = self.model.get_latent(x)
                    latents.append(z.cpu())
                labels.append(lbl)

        latents = torch.cat(latents, 0).numpy()
        labels = torch.cat(labels, 0).numpy()

        n = min(500, len(latents))
        idx = np.random.choice(len(latents), n, replace=False)
        emb = TSNE(n_components=2, random_state=42,
                   perplexity=min(30, n - 1)).fit_transform(latents[idx])
        lbl_s = labels[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        for val, color, name in [(0, 'green', 'Good'), (1, 'red', 'Defective')]:
            m = lbl_s == val
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], c=color, label=name, alpha=0.6, s=15)
        ax.legend()
        ax.set_title(f'Distribución de Embeddings (t-SNE) - Prueba - Epoch {epoch}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        plt.tight_layout()
        if wandb.run is not None:
            wandb.log({'test/tsne': wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.model_cfg.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val/loss'}}
