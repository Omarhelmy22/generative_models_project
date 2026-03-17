import torch.nn as nn


class Decoder(nn.Module):
    """
    CNN decoder that maps a latent vector back to image space.

    Architecture:
        FC -> reshape to (128, 4, 4)
        ConvTranspose(128->64, 4x4, stride 2) -> 8x8
        ConvTranspose(64->32, 4x4, stride 2) -> 16x16
        ConvTranspose(32->3, 4x4, stride 2) -> 32x32
        Tanh activation to produce output in [-1, 1]
    """

    def __init__(self, latent_dim=128, decoder_channels=None, out_channels=3):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [128, 64, 32]

        self.initial_channels = decoder_channels[0]
        self.fc = nn.Linear(latent_dim, decoder_channels[0] * 4 * 4)

        layers = []
        for i in range(len(decoder_channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    decoder_channels[i], decoder_channels[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.BatchNorm2d(decoder_channels[i + 1]),
                nn.ReLU(inplace=True),
            ])

        layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1], out_channels,
                kernel_size=4, stride=2, padding=1,
            )
        )
        layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.initial_channels, 4, 4)
        return self.deconv(h)
