import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from functools import partial

from neugk.dataset import CycloneAEDataset


def get_density(df):
    return torch.sum(df, dim=(-2, -1))


def get_parallel_flux(df, v_parallel_grid):
    integrand = df * v_parallel_grid
    return torch.sum(integrand, dim=(-2, -1))


def get_parallel_energy(df, v_parallel_grid, m_s):
    parallel_ke = 0.5 * m_s * (v_parallel_grid**2)
    integrand = df * parallel_ke
    return torch.sum(integrand, dim=(-2, -1))


def vapor_loss(df_pred, df_gt, geom, m_s: float = 1.0):
    v_parallel_grid = rearrange(geom["vpgr"], "par -> 1 1 par 1").to(df_pred.device)

    n_gt = get_density(df_gt)
    flux_gt = get_parallel_flux(df_gt, v_parallel_grid)
    e_para_gt = get_parallel_energy(df_gt, v_parallel_grid, m_s)

    v_para_grid_d = v_parallel_grid.detach()

    n_pred = get_density(df_pred)
    flux_pred = get_parallel_flux(df_pred, v_para_grid_d)
    e_para_pred = get_parallel_energy(df_pred, v_para_grid_d, m_s)

    # L_mass
    L_mass = F.mse_loss(n_pred, n_gt)
    L_v_parallel = F.mse_loss(flux_pred, flux_gt)
    L_E_parallel = F.mse_loss(e_para_pred, e_para_gt)

    return {"mass": L_mass, "momentum": L_v_parallel, "energy": L_E_parallel}


class SpectralConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, modes1, modes2
    ):  # <--- FIX: Added modes1 and modes2
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modes for height
        self.modes2 = modes2  # Modes for width (frequency)

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
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm="ortho")
        x_ft = torch.view_as_real(x_ft)

        # Check if data dimensions are smaller than mode dimensions
        data_modes1 = x_ft.shape[2]
        data_modes2 = x_ft.shape[3]

        if self.modes1 > data_modes1 or self.modes2 > data_modes2:
            raise ValueError(
                f"FNO modes ({self.modes1}, {self.modes2}) are larger than "
                f"input data's Fourier modes ({data_modes1}, {data_modes2}). "
                f"Reduce fno_modes_h or fno_modes_w."
            )

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            2,
            device=x.device,
        )

        op = partial(torch.einsum, "bixy,ioxy->boxy")

        # Slicing is now correct
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

        out_ft = torch.view_as_complex(out_ft)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class FNOBasicBlock(nn.Module):
    """ResNet-style block for FNO."""

    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, modes1=12, modes2=12
    ):  # <--- FIX: Added modes1 and modes2
        super().__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, modes1, modes2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, modes1, modes2)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, self.expansion * planes, modes1, modes2),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FNORefiner(nn.Module):
    """
    FNO model, adapted as a refiner.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1=12,  # <--- FIX: Added modes1
        modes2=12,  # <--- FIX: Added modes2
        width=32,
        num_blocks=[2, 2, 2, 2],
    ):
        super().__init__()
        self.in_planes = width
        self.modes1 = modes1
        self.modes2 = modes2

        self.conv1 = SpectralConv2d(in_channels, width, self.modes1, self.modes2)
        self.bn1 = nn.BatchNorm2d(width)

        self.layer1 = self._make_layer(FNOBasicBlock, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(FNOBasicBlock, width, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(FNOBasicBlock, width, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(FNOBasicBlock, width, num_blocks[3], stride=1)

        self.conv2 = SpectralConv2d(width, out_channels, self.modes1, self.modes2)

    def _make_layer(
        self, block, planes, num_blocks, stride
    ):  # <--- FIX: Pass modes to block
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.modes1, self.modes2))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv2(out)
        return out


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # quantized: BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
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

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # quantized: BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.leaky_relu(x)


class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super().__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.leaky_relu(x)
        x = self._conv_2(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
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
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=padding[0],
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=padding[1],
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.leaky_relu(x)
        x = self._conv_trans_2(x)
        return x


class VQVAE(nn.Module):
    """
    This is the main VQ-VAE model, renamed from 'Model' in the original script.
    """

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
            in_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            in_channels,  # Output channels must match input channels
            padding=padding,
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, _, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, quantized


class VAPOR(nn.Module):
    def __init__(self, vqvae: nn.Module, refiner: nn.Module):
        super().__init__()

        self.vqvae = vqvae
        self.refiner = refiner

    def forward(self, df, **_):
        full_5d = False
        if df.ndim > 4:
            full_5d = True
            df = rearrange(df, "b c vp mu s x y -> (b s x y) c vp mu")
        # vqvae
        vq_loss, df_vae, indices = self.vqvae(df)
        # fno
        df = self.refiner(df_vae)
        if full_5d:
            df = rearrange(df, "(b s x y) c vp mu -> b c vp mu s x y", s=16, x=85, y=32)
        return {"df": df, "df_vae": df_vae, "vq_loss": vq_loss, "indices": indices}


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
    GEOM = cyc_dataset[0].geometry
    BATCH_SIZE = 4 * 4096

    N_SAMPLES = 512

    data = torch.cat(
        [
            rearrange(cyc_dataset[i].df, "c vp mu s x y -> (s x y) c vp mu")
            for i in np.random.randint(0, 260, N_SAMPLES)
        ],
        dim=0,
    )
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Total of {len(dataset)/1e6:.1f}M samples")

    num_hiddens = 256
    num_residual_hiddens = 64
    num_residual_layers = 4
    embedding_dim = 1
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99

    # FNO Params
    fno_modes1 = 8
    fno_modes2 = 4
    fno_width = 32

    # Training Params
    EPOCHS = 10

    padding = [0, 0]

    vqvae = VQVAE(
        in_channels=2,
        num_hiddens=num_hiddens,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        decay=decay,
        padding=padding,
    ).to(device)

    fno_refiner = FNORefiner(
        in_channels=2,
        out_channels=2,
        modes1=fno_modes1,
        modes2=fno_modes2,
        width=fno_width,
        num_blocks=[2, 2, 2, 2],
    ).to(device)

    vapor = VAPOR(vqvae, fno_refiner)

    all_params = list(vqvae.parameters()) + list(fno_refiner.parameters())
    optimizer = optim.AdamW(all_params, 1e-3, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-8)

    print(f"VQVAE Model Parameters: {sum(p.numel() for p in vqvae.parameters()):,}")
    print(
        f"FNO Refiner Parameters: {sum(p.numel() for p in fno_refiner.parameters()):,}"
    )

    vqvae.train()
    fno_refiner.train()

    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"[e: {epoch+1}/{EPOCHS}]")

        for df in progress_bar:
            df = df[0].to(device)
            outs = vapor(df)
            recon_loss = F.mse_loss(outs["df"], df)
            physics_losses = vapor_loss(outs["df"], df, GEOM)
            # total loss
            total_loss = recon_loss + outs["vq_loss"] + sum(physics_losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            cr = df[0].nbytes / outs["indices"][0].to(torch.int16).nbytes

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
    from neural_fields.data import CycloneNFDataset
    from neural_fields.nf_train import eval_diagnose

    data = CycloneNFDataset("iteration_13", timesteps=64)

    with torch.no_grad():
        gt_df = cyc_valset[64].df.to(device).unsqueeze(0)
        df = vapor(gt_df)["df"]
        eval_diagnose(data, pred_df=df[0], device=device)
