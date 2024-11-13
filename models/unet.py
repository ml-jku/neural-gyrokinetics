from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self,
        n_fields: int,
        input_timesteps: int,
        output_timesteps: int,
        hidden_dim: int = 8,
    ):
        super().__init__()

        self.activation = nn.GELU()
        self.enc1 = UNet._block(n_fields * input_timesteps, hidden_dim, "enc1")
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = UNet._block(hidden_dim, hidden_dim * 2, "enc2")
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.enc3 = UNet._block(hidden_dim * 2, hidden_dim * 4, "enc3")
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.enc4 = UNet._block(hidden_dim * 4, hidden_dim * 8, "enc4")
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(hidden_dim * 8, hidden_dim * 16, "bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            hidden_dim * 16, hidden_dim * 8, kernel_size=(2, 2), stride=2
        )
        self.dec4 = UNet._block(hidden_dim * 8 * 2, hidden_dim * 8, "dec4")

        self.upconv3 = nn.ConvTranspose2d(
            hidden_dim * 8, hidden_dim * 4, kernel_size=(2, 2), stride=2, padding=1
        )
        self.dec3 = UNet._block(hidden_dim * 4 * 2, hidden_dim * 4, "dec3")

        self.upconv2 = nn.ConvTranspose2d(
            hidden_dim * 4, hidden_dim * 2, kernel_size=(2, 2), stride=2, padding=1
        )
        self.dec2 = UNet._block(hidden_dim * 2 * 2, hidden_dim * 2, "dec2")

        self.upconv1 = nn.ConvTranspose2d(
            hidden_dim * 2, hidden_dim, kernel_size=(2, 2), stride=2
        )
        self.dec1 = UNet._block(hidden_dim * 1 * 2, hidden_dim, "dec1")

        self.conv = nn.Conv2d(hidden_dim, n_fields * output_timesteps, kernel_size=1)

    def forward(self, x, grid, ts):
        # encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool1(enc1))
        enc3 = self.enc3(self.maxpool2(enc2))
        enc4 = self.enc4(self.maxpool3(enc3))

        # bottleneck
        bottleneck = self.bottleneck(self.maxpool4(enc4))

        # decoding path
        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        out = self.conv(dec1)
        return out

    @staticmethod
    def _block(in_channels, features, name, activation=nn.GELU()):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=(3, 3),
                            padding=1,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "act1", activation),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=(3, 3),
                            padding=1,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "act2", activation),
                ]
            )
        )
