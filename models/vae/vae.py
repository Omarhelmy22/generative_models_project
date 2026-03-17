import torch
import torch.nn as nn
from models.vae.encoder import Encoder
from models.vae.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder.

    Combines an encoder (image -> mu, logvar), the reparameterization trick,
    and a decoder (z -> reconstructed image).
    """

    def __init__(self, in_channels=3, encoder_channels=None, latent_dim=128,
                 decoder_channels=None):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels))

        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, encoder_channels, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_channels, in_channels)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * epsilon using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples, device):
        """Generate images by sampling z ~ N(0, I) and decoding."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
