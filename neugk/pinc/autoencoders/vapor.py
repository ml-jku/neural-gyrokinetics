import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from functools import partial

from neugk.dataset import CycloneAEDataset


def get_density(df):
    return torch.sum(df, dim=(-2, -1))


def get_parallel_flux(df, v_parallel_grid):
    return torch.sum(df * v_parallel_grid, dim=(-2, -1))


def get_parallel_energy(df, v_parallel_grid, m_s):
    return torch.sum(df * (0.5 * m_s * (v_parallel_grid**2)), dim=(-2, -1))


def vapor_loss(df_pred, df_gt, geom, m_s: float = 1.0):
    v_parallel_grid = rearrange(geom["vpgr"], "par -> 1 1 par 1").to(df_pred.device)

    n_gt = get_density(df_gt)
    flux_gt = get_parallel_flux(df_gt, v_parallel_grid)
    e_para_gt = get_parallel_energy(df_gt, v_parallel_grid, m_s)

    v_para_grid_d = v_parallel_grid.detach()

    n_pred = get_density(df_pred)
    flux_pred = get_parallel_flux(df_pred, v_para_grid_d)
    e_para_pred = get_parallel_energy(df_pred, v_para_grid_d, m_s)

    return {
        "mass": F.mse_loss(n_pred, n_gt),
        "momentum": F.mse_loss(flux_pred, flux_gt),
        "energy": F.mse_loss(e_para_pred, e_para_gt),
    }


class SpectralConv2d(nn.Module):
    """SpectralConv2d class."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.view_as_real(torch.fft.rfft2(x, norm="ortho"))

        if self.modes1 > x_ft.shape[2] or self.modes2 > x_ft.shape[3]:
            raise ValueError(
                f"fno modes ({self.modes1}, {self.modes2}) are larger than input data's fourier modes ({x_ft.shape[2]}, {x_ft.shape[3]})."
            )

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            2,
            device=x.device,
        )
        op = partial(torch.einsum, "bixy,ioxy->boxy")

        out_ft[:, :, : self.modes1, : self.modes2] = torch.stack(
            [
                op(x_ft[:, :, : self.modes1, : self.modes2, 0], self.weights1[..., 0])
                - op(
                    x_ft[:, :, : self.modes1, : self.modes2, 1], self.weights1[..., 1]
                ),
                op(x_ft[:, :, : self.modes1, : self.modes2, 1], self.weights1[..., 0])
                + op(
                    x_ft[:, :, : self.modes1, : self.modes2, 0], self.weights1[..., 1]
                ),
            ],
            dim=-1,
        )

        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.stack(
            [
                op(x_ft[:, :, -self.modes1 :, : self.modes2, 0], self.weights2[..., 0])
                - op(
                    x_ft[:, :, -self.modes1 :, : self.modes2, 1], self.weights2[..., 1]
                ),
                op(x_ft[:, :, -self.modes1 :, : self.modes2, 1], self.weights2[..., 0])
                + op(
                    x_ft[:, :, -self.modes1 :, : self.modes2, 0], self.weights2[..., 1]
                ),
            ],
            dim=-1,
        )

        return torch.fft.irfft2(
            torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)), norm="ortho"
        )


class FNOBasicBlock(nn.Module):
    """FNOBasicBlock class."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, modes1=12, modes2=12):
        super().__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, modes1, modes2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, modes1, modes2)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = (
            nn.Sequential(
                SpectralConv2d(in_planes, self.expansion * planes, modes1, modes2),
                nn.BatchNorm2d(self.expansion * planes),
            )
            if stride != 1 or in_planes != self.expansion * planes
            else nn.Sequential()
        )

    def forward(self, x):
        return F.relu(
            self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x)
        )


class FNORefiner(nn.Module):
    """FNORefiner class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1=12,
        modes2=12,
        width=32,
        num_blocks=[2, 2, 2, 2],
    ):
        super().__init__()
        self.in_planes, self.modes1, self.modes2 = width, modes1, modes2
        self.conv1 = SpectralConv2d(in_channels, width, self.modes1, self.modes2)
        self.bn1 = nn.BatchNorm2d(width)

        self.layer1 = self._make_layer(FNOBasicBlock, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(FNOBasicBlock, width, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(FNOBasicBlock, width, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(FNOBasicBlock, width, num_blocks[3], stride=1)
        self.conv2 = SpectralConv2d(width, out_channels, self.modes1, self.modes2)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s, self.modes1, self.modes2))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv2(
            self.layer4(
                self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(x))))))
            )
        )


class VectorQuantizer(nn.Module):
    """VectorQuantizer class."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim, self._num_embeddings = embedding_dim, num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        ).scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        loss = F.mse_loss(
            quantized, inputs.detach()
        ) + self._commitment_cost * F.mse_loss(quantized.detach(), inputs)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """VectorQuantizerEMA class."""

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super().__init__()
        self._embedding_dim, self._num_embeddings = embedding_dim, num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay, self._epsilon = decay, epsilon

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        ).scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        loss = self._commitment_cost * F.mse_loss(quantized.detach(), inputs)

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    """Residual class."""

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Conv2d(
                in_channels,
                num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(True),
            nn.Conv2d(
                num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    """ResidualStack class."""

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super().__init__()
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.leaky_relu(x)


class Encoder(nn.Module):
    """Encoder class."""

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super().__init__()
        self._conv_1 = nn.Conv2d(
            in_channels, num_hiddens // 2, kernel_size=3, stride=2, padding=1
        )
        self._conv_2 = nn.Conv2d(
            num_hiddens // 2, num_hiddens, kernel_size=3, stride=2, padding=1
        )
        self._residual_stack = ResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, inputs):
        return self._residual_stack(self._conv_2(F.leaky_relu(self._conv_1(inputs))))


class Decoder(nn.Module):
    """Decoder class."""

    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        out_channels,
        padding=[1, 1, 1],
    ):
        super().__init__()
        self._conv_1 = nn.Conv2d(
            in_channels, num_hiddens, kernel_size=3, stride=1, padding=1
        )
        self._residual_stack = ResidualStack(
            num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._conv_trans_1 = nn.ConvTranspose2d(
            num_hiddens,
            num_hiddens // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=padding[0],
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            num_hiddens // 2,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=padding[1],
        )

    def forward(self, inputs):
        return self._conv_trans_2(
            F.leaky_relu(self._conv_trans_1(self._residual_stack(self._conv_1(inputs))))
        )


class VQVAE(nn.Module):
    """VQVAE class."""

    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
        padding=[1, 1, 1],
    ):
        super().__init__()
        self._encoder = Encoder(
            in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            num_hiddens, embedding_dim, kernel_size=1, stride=1
        )
        self._vq_vae = (
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            if decay > 0.0
            else VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        )
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            in_channels,
            padding=padding,
        )

    def forward(self, x):
        loss, quantized, _, _ = self._vq_vae(self._pre_vq_conv(self._encoder(x)))
        return loss, self._decoder(quantized), quantized


class VAPOR(nn.Module):
    """VAPOR class."""

    def __init__(self, vqvae: nn.Module, refiner: nn.Module):
        super().__init__()
        self.vqvae, self.refiner = vqvae, refiner

    def forward(self, df, **_):
        full_5d = df.ndim > 4
        if full_5d:
            df = rearrange(df, "b c vp mu s x y -> (b s x y) c vp mu")
        vq_loss, df_vae, indices = self.vqvae(df)
        df_out = self.refiner(df_vae)
        if full_5d:
            df_out = rearrange(
                df_out, "(b s x y) c vp mu -> b c vp mu s x y", s=16, x=85, y=32
            )
        return {"df": df_out, "df_vae": df_vae, "vq_loss": vq_loss, "indices": indices}


if __name__ == "__main__":
    device = "cuda"

    cyc_dataset = CycloneAEDataset(
        input_fields=["df", "phi", "flux"],
        split="train",
        normalization="zscore",
        normalization_scope="dataset",
        trajectories="iteration_{0-5,7-12,14-31,33-58,60-67,69-82,84-99}.h5",
        cond_filters={"last_fluxes": [1.0, np.inf], "first_fluxes": [1.0, np.inf]},
        separate_zf=False,
        real_potens=True,
        stage="ae",
        conditions=["itg"],
    )

    cyc_valset = CycloneAEDataset(
        input_fields=["df", "phi", "flux"],
        split="val",
        normalization="zscore",
        normalization_scope="dataset",
        trajectories=["iteration_13.h5"],
        separate_zf=False,
        real_potens=True,
        stage="ae",
        conditions=["itg"],
        normalization_stats=cyc_dataset.norm_stats,
    )

    data = torch.cat(
        [
            rearrange(cyc_dataset[i].df, "c vp mu s x y -> (s x y) c vp mu")
            for i in np.random.randint(0, 260, 512)
        ],
        dim=0,
    )
    dataloader = DataLoader(
        TensorDataset(data), batch_size=4 * 4096, shuffle=True, num_workers=0
    )

    print(f"total of {len(data)/1e6:.1f}M samples")

    vqvae = VQVAE(
        in_channels=2,
        num_hiddens=256,
        num_residual_layers=4,
        num_residual_hiddens=64,
        num_embeddings=512,
        embedding_dim=1,
        commitment_cost=0.25,
        decay=0.99,
        padding=[0, 0],
    ).to(device)
    fno_refiner = FNORefiner(
        in_channels=2,
        out_channels=2,
        modes1=8,
        modes2=4,
        width=32,
        num_blocks=[2, 2, 2, 2],
    ).to(device)
    vapor = VAPOR(vqvae, fno_refiner)

    optimizer = optim.AdamW(
        list(vqvae.parameters()) + list(fno_refiner.parameters()),
        1e-3,
        weight_decay=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-8)

    print(f"vqvae parameters: {sum(p.numel() for p in vqvae.parameters()):,}")
    print(f"fno parameters: {sum(p.numel() for p in fno_refiner.parameters()):,}")

    vqvae.train()
    fno_refiner.train()

    for epoch in range(10):
        progress_bar = tqdm(dataloader, desc=f"[e: {epoch+1}/{10}]")
        for df in progress_bar:
            df = df[0].to(device)
            outs = vapor(df)
            recon_loss = F.mse_loss(outs["df"], df)
            physics_losses = vapor_loss(outs["df"], df, cyc_dataset[0].geometry)
            total_loss = recon_loss + outs["vq_loss"] + sum(physics_losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cr = df.nbytes / outs["indices"].to(torch.int16).nbytes

            progress_bar.set_postfix(
                total=f"{total_loss.item():.4f}",
                recon=f"{recon_loss.item():.4f}",
                vq=f"{outs['vq_loss'].item():.4f}",
                mass=f"{physics_losses['mass']:.2f}",
                momentum=f"{physics_losses['momentum']:.2f}",
                energy=f"{physics_losses['energy']:.2f}",
                cr=f"{cr:.1f}x",
            )
        scheduler.step()
        torch.save(vapor.state_dict(), f"vapor{epoch}.pth")

    torch.save(vapor.state_dict(), "vapor.pth")
    from neugk.pinc.neural_fields.data import CycloneNFDataset
    from neugk.pinc.neural_fields.nf_train import eval_diagnose

    data = CycloneNFDataset("iteration_13", timesteps=64)
    with torch.no_grad():
        df = vapor(cyc_valset[64].df.to(device).unsqueeze(0))["df"]
        eval_diagnose(data, pred_df=df[0], device=device)
