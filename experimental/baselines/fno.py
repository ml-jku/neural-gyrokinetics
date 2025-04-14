from typing import Sequence, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.utils import Film


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, dim, modes1=8, modes2=8, modes3=8):
        super().__init__()

        self.dim = dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(dim + 3, dim)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(dim, dim, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(dim, dim, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(dim, dim, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(dim, dim, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(dim, dim, 1)
        self.w1 = nn.Conv3d(dim, dim, 1)
        self.w2 = nn.Conv3d(dim, dim, 1)
        self.w3 = nn.Conv3d(dim, dim, 1)
        self.bn0 = torch.nn.BatchNorm3d(dim)
        self.bn1 = torch.nn.BatchNorm3d(dim)
        self.bn2 = torch.nn.BatchNorm3d(dim)
        self.bn3 = torch.nn.BatchNorm3d(dim)

        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, dim)

    def forward(self, x, grid):
        # x dim = [b, x1, x2, x3, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)


class FNOLayer(nn.Module):
    def __init__(
        self,
        space: int,
        dim: int,
        depth: int,
        grid_size: Sequence[int],
        drop_path: Union[Sequence[float], float] = 0.0,
        use_checkpoint: bool = False,
        init_weights: Optional[str] = None,
        **kwargs
    ) -> None:

        super().__init__()
        _ = space
        _ = kwargs

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.grid_size = grid_size
        self.drop_path = drop_path

        self.blocks = nn.ModuleList(
            [FNO3d(dim=dim, modes1=16, modes2=4, modes3=8) for _ in range(depth)]
        )

        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class FilmFNOLayer(FNOLayer):
    """Film-conditioned Vision Transformer layer."""

    def __init__(self, *args, cond_dim: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.conditioning = nn.ModuleList(
            [Film(cond_dim, self.dim) for _ in range(len(self.blocks))]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for blk, cond in zip(self.blocks, self.conditioning):
            x = cond(x, condition)
            x = blk(x)
        return x
