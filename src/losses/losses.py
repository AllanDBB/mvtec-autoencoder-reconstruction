import torch
import torch.nn as nn
from pytorch_msssim import ssim


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss supporting l1, l2, ssim, ssim_l1.
    Expects inputs in [0, 1] range.
    """

    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ('l1', 'l2', 'ssim', 'ssim_l1'):
            raise ValueError(f"Unknown loss: {loss_type}. Choose l1 | l2 | ssim | ssim_l1")

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'l1':
            return self.l1(recon, target)
        elif self.loss_type == 'l2':
            return self.l2(recon, target)
        elif self.loss_type == 'ssim':
            return 1.0 - ssim(recon, target, data_range=1.0, size_average=True)
        elif self.loss_type == 'ssim_l1':
            ssim_loss = 1.0 - ssim(recon, target, data_range=1.0, size_average=True)
            return 0.85 * ssim_loss + 0.15 * self.l1(recon, target)
