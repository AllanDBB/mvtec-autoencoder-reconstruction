import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        if last:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class VAE(nn.Module):
    """
    Variational Autoencoder for 128x128x3 images.
    5 stride-2 conv layers: 128 -> 64 -> 32 -> 16 -> 8 -> 4
    Latent space: latent_dim-dimensional Gaussian.
    """

    def __init__(self, latent_dim=128, encoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 512]

        self.latent_dim = latent_dim
        self.encoder_channels = list(encoder_channels)

        # Encoder
        enc_layers = []
        in_ch = 3
        for out_ch in self.encoder_channels:
            enc_layers.append(_EncoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # After 5 stride-2 convs on 128x128: spatial = 4x4
        self._flat_dim = self.encoder_channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)
        dec_channels = list(reversed(self.encoder_channels))
        dec_layers = []
        for i in range(len(dec_channels) - 1):
            dec_layers.append(_DecoderBlock(dec_channels[i], dec_channels[i + 1]))
        dec_layers.append(_DecoderBlock(dec_channels[-1], 3, last=True))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        h = self.fc_decode(z).view(-1, self.encoder_channels[-1], 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu
