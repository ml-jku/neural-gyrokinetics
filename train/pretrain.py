import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from .losses import autoencoder_loss


def pretrain_encoder_decoder(
    model: nn.Module,
    pretrain_epochs: int,
    trainloader: DataLoader,
    freeze: bool = False,
) -> nn.Module:
    if (conditioner := getattr(model, "conditioner", None)) is not None:
        autoencoder = [conditioner]
    else:
        autoencoder = []

    autoencoder += [model.encoder, model.decoder]
    autoencoder = nn.ModuleList(autoencoder)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=5e-4, weight_decay=1e-6)

    for epoch in range(pretrain_epochs):
        train_mse = 0
        for sample in trainloader:
            x, grid, _, timestep = sample

            loss = 0
            optimizer.zero_grad()

            loss += autoencoder_loss(
                model.encoder,
                model.decoder,
                x,
                grid,
                ts=timestep,
                conditioner=conditioner,
            )

            loss.backward()
            train_mse += loss.item()
            optimizer.step()

        train_mse /= len(trainloader)
        print(f"[Pretraining epoch {epoch}] loss: {train_mse:.5f}")

    # freeze encoder-decoder?
    if freeze is True:
        model.encoder.requires_grad_ = False
        model.decoder.requires_grad_ = False
        model.conditioner.requires_grad_ = False

        print("Encoder-Decoder-Conditioner params frozen")

    return model
