import torch
import torch.nn as nn

# GAN model for generating images without damage

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.ModuleList([
            self._block(in_channels, features, normalize=False),  # 64
            self._block(features, features * 2),  # 128
            self._block(features * 2, features * 4),  # 256
            self._block(features * 4, features * 8),  # 512
            self._block(features * 8, features * 8),
            self._block(features * 8, features * 8),
            self._block(features * 8, features * 8),
            self._block(features * 8, features * 8),
        ])

        self.decoder = nn.ModuleList([
            self._up_block(features * 8, features * 8, dropout=True),
            self._up_block(features * 16, features * 8, dropout=True),
            self._up_block(features * 16, features * 8, dropout=True),
            self._up_block(features * 16, features * 8),
            self._up_block(features * 16, features * 4),
            self._up_block(features * 8, features * 2),
            self._up_block(features * 4, features),
        ])

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _up_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = skips[-1]          # Deepest encoder output
        skips = skips[:-1][::-1]  # Reverse for skip connection order

        for i, up in enumerate(self.decoder):
            x = up(x)
            if i < len(skips):
                x = torch.cat((x, skips[i]), 1)

        return self.final(x)



class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(features, features * 2),
            self._block(features * 2, features * 4),
            self._block(features * 4, features * 8, stride=1),
            nn.Conv2d(features * 8, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


if __name__ == "__main__":
    # Test shapes
    gen = UNetGenerator()
    disc = PatchDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    y = gen(x)
    print("Generated image shape:", y.shape)
    print("Discriminator output shape:", disc(x, y).shape)
