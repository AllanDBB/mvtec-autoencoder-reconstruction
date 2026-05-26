import torch
import torch.nn as nn


class _EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAE(nn.Module):
    """
    U-Net autoencoder for 128x128x3 images.
    encoder_channels define the 4 downsampling stages.
    Bottleneck adds one extra stride-2 conv, then projects to latent_dim for t-SNE.
    Skip connections from each encoder stage are concatenated in the decoder.
    """

    def __init__(self, latent_dim=128, encoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        self.latent_dim = latent_dim
        self.enc_chs = list(encoder_channels)
        bottleneck_ch = self.enc_chs[-1] * 2  # 1024

        # Encoder blocks: 128->64->32->16->8
        self.enc_blocks = nn.ModuleList()
        in_ch = 3
        for out_ch in self.enc_chs:
            self.enc_blocks.append(_EncBlock(in_ch, out_ch))
            in_ch = out_ch

        # Bottleneck: 8->4, [B, bottleneck_ch, 4, 4]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.enc_chs[-1], bottleneck_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Latent projection for t-SNE (not used in reconstruction path)
        self.latent_proj = nn.Linear(bottleneck_ch * 4 * 4, latent_dim)

        # Decoder blocks (reverse order, each uses skip from encoder)
        self.dec_blocks = nn.ModuleList()
        dec_in = bottleneck_ch
        for out_ch in reversed(self.enc_chs):
            self.dec_blocks.append(_DecBlock(dec_in, out_ch))
            dec_in = out_ch

        # Final upsample to 128x128 + output conv
        self.final_up = nn.ConvTranspose2d(self.enc_chs[0], 32, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        # Latent vector for t-SNE
        z = self.latent_proj(h.flatten(1))

        for i, block in enumerate(self.dec_blocks):
            skip = skips[-(i + 1)]
            h = block(h, skip)

        h = self.final_up(h)
        recon = self.final_conv(h)
        return recon, z

    def get_latent(self, x):
        _, z = self.forward(x)
        return z
