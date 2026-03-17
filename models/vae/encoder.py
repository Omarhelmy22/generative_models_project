import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    CNN encoder that maps an input image to a mean and log-variance
    in latent space.

    Architecture for 32x32 input:
        Conv(3->32, 4x4, stride 2) -> 16x16
        Conv(32->64, 4x4, stride 2) -> 8x8
        Conv(64->128, 4x4, stride 2) -> 4x4
        Flatten -> FC -> (mu, logvar)
    """

    def __init__(self, in_channels=3, encoder_channels=None, latent_dim=128):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]

        layers = []
        ch_in = in_channels
        for ch_out in encoder_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out

        self.conv = nn.Sequential(*layers)

        # After 3 stride-2 convolutions on 32x32: spatial dims = 4x4
        flat_dim = encoder_channels[-1] * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
