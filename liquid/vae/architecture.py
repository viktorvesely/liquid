import numpy as np
import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512,
        num_layers: int = 5
    ):
        super().__init__()
        self.latent_dim = latent_dim

        channels = np.linspace(in_channels, latent_dim * 2, num=(num_layers + 1), endpoint=True).round().astype(int)

        enc_blocks = []
        for prev, fol in zip(channels[:-1], channels[1:], strict=True):
            enc_blocks.extend([
                nn.Conv2d(in_channels=prev, out_channels=fol, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_features=fol),
                nn.LeakyReLU()
            ])


        self.encoder = nn.Sequential(*enc_blocks)

        channels = np.linspace(latent_dim, in_channels, num=(num_layers + 1), endpoint=True).round().astype(int)
        dec_blocks = []
        for prev, fol in zip(channels[:-1], channels[1:], strict=True):
            dec_blocks.extend([
                nn.ConvTranspose2d(in_channels=prev, out_channels=fol, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=fol, out_channels=fol, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=fol),
                nn.LeakyReLU(),
            ])

        dec_blocks.pop()
        dec_blocks.pop()

        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
