"""Qualitative visualization functions for 5D and 3D data."""

from typing import Dict, Optional

import io
from PIL import Image as PILImage
import wandb
import warnings

import matplotlib
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt

import numpy as np
import torch


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plt_to_wandb_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return wandb.Image(img)


def distribution_5D(x, **kwargs):
    _ = kwargs
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    for i in range(5):
        for j in range(5):
            if [i, j] not in comb:
                ax[i, j].remove()

    c_map = matplotlib.colormaps["coolwarm"]
    c_map.set_bad("k")

    imin = -1
    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])
        xx = x[0].std(other)
        xx[xx == 0] = np.nan
        ax[i, j].matshow(xx, cmap=c_map)

        if i > imin:
            ax[i, j].set_ylabel(rf"${labels[i]}$", fontsize=20)
            ax[i, j].set_xlabel(rf"${labels[j]}$", fontsize=20)
            imin = i

        force_aspect(ax[i, j])

    return plt_to_wandb_image(fig)


def plot4x4_sided(x1, x2, title="", mark_bad=False, average=True):
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]
    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(30, 14))
    for i in range(5):
        for j in range(5):
            if j == 0:
                ax[i, j].remove()
                continue
            if i == 4:
                ax[i, j].remove()
                continue
            ax_ij = ax[i, j]
            ax_ij.set_frame_on(False)
            ax_ij.tick_params(labelleft=False, labelbottom=False)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])

    # fig.tight_layout()
    fig.suptitle(title)
    c_map = matplotlib.colormaps["RdBu"]
    c_map.set_bad("k")

    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])

        if average:
            x1_plot = x1[0].mean(other)
            x2_plot = x2[0].mean(other)
        else:
            x1_plot = torch.tensor(x1[0]).permute(i, j, *other).numpy()[:, :, 0, 0, 0]
            x2_plot = torch.tensor(x2[0]).permute(i, j, *other).numpy()[:, :, 0, 0, 0]

        if mark_bad:
            x1_std = x1.std(other)
            x2_std = x2.std(other)
            x1_plot[x1_std == 0] = np.nan
            x2_plot[x2_std == 0] = np.nan

        ax_ij = ax[i, j]
        pos = ax_ij.get_position()

        # create two new axes within the same space as the original subplot
        plot_width = 0.475 * pos.width
        left_margin = 0.0 * pos.width
        x_left_1 = pos.x0 + left_margin
        x_left_2 = x_left_1 + plot_width
        y = pos.y0
        h = pos.height
        ax1 = fig.add_axes([x_left_1, y, plot_width, h])
        ax2 = fig.add_axes([x_left_2, y, plot_width, h])

        # compute shared vmin and vmax
        vmin = min(x1_plot.min(), x2_plot.min())
        vmax = max(x1_plot.max(), x2_plot.max())

        im1 = ax1.matshow(x1_plot, cmap=c_map, vmin=vmin, vmax=vmax)
        ax2.matshow(x2_plot, cmap=c_map, vmin=vmin, vmax=vmax)

        # shared colourbar
        cbar = fig.colorbar(
            im1, ax=[ax_ij], format=tkr.FormatStrFormatter("%.2g"), pad=0, fraction=0.05
        )
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.ax.tick_params(labelsize=12)

        if i == 0:
            # Set axis labels
            ax1.set_title(r"PRED", fontsize=24)
            ax2.set_title(r"GT", fontsize=24)

        if j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3) or (i == 3 and j == 4):
            ax_ij.set_ylabel(rf"${labels[i]}$", fontsize=14)

        if i == 3 or j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3):
            ax_ij.set_xlabel(rf"${labels[j]}$", fontsize=14)

        # Remove axis ticks and labels
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.tick_params(labelleft=False, labelbottom=False)
        ax2.tick_params(labelleft=False, labelbottom=False)
        # Force aspect ratio
        force_aspect(ax1)
        force_aspect(ax2)

    return plt_to_wandb_image(fig)


def mse_time_histogram(losses):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    times = sorted(losses.keys())
    losses_mean = [np.mean(losses[t]) for t in times]
    losses_std = [np.std(losses[t]) for t in times]
    # Bar plot with error bars
    ax.bar(times, losses_mean, yerr=losses_std, alpha=0.7, capsize=5, color="blue")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("MSE by Time Step")
    ax.grid(True)
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def radially_averaged_power_spectrum_nd(image):
    warnings.warn("radially_averaged_power_spectrum_nd is wrong!")
    image = image - image.mean()
    fourier_transform = np.fft.fftn(image)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)
    power_spectrum = np.abs(fourier_transform_shifted) ** 2
    # Create a grid of radial distances from the center
    shape = image.shape
    center = np.array(shape) // 2
    indices = np.indices(shape)
    r = np.sqrt(((indices - center.reshape((-1,) + (1,) * len(shape))) ** 2).sum(0))
    r = r.astype(int)
    # Sum the power spectrum values at each radius
    radial_sum = np.bincount(r.ravel(), power_spectrum.ravel())
    # Count the number of pixels at each radius
    radial_count = np.bincount(r.ravel())
    return radial_sum / radial_count


def plot_5D_raspec(x, x2):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout="tight")
    raspec = radially_averaged_power_spectrum_nd(x.cpu().detach().numpy())
    gt_raspec = radially_averaged_power_spectrum_nd(x2.cpu().detach().numpy())
    ax.loglog(raspec, label="Pred spec", c="r")
    ax.loglog(gt_raspec, label="GT spec", c="k")
    ax.set_xlabel("Freq")
    ax.set_ylabel("A")
    ax.grid(True)
    return plt_to_wandb_image(fig)


def plot_4x4_2D_raspec(x1, x2=None, **kwargs):
    from pysteps.utils.spectral import rapsd

    _ = kwargs
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]

    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    for i in range(5):
        for j in range(5):
            if [i, j] not in comb:
                ax[i, j].remove()

    imin = -1
    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])
        xx = np.stack(
            [x1[0].permute(i, j, *other).numpy(), x1[1].permute(i, j, *other).numpy()],
            axis=-1,
        )
        xx = np.nan_to_num(xx)
        xx = np.complex64(xx)

        slices = [tuple(np.random.randint(0, dim, size=100)) for dim in xx.shape[2:]]
        slices = list(zip(*slices))
        # slices = np.ndindex(*xx.shape[2:])  # all slices

        # radially averaged power spectrum for each slice
        xx_raspec = [rapsd(xx[:, :, *sl], fft_method=np.fft) for sl in slices]
        xx_raspec_avg = np.mean(xx_raspec, axis=0)

        ax[i, j].loglog(xx_raspec_avg, label="Pred spec", c="r", lw=3)
        ax[i, j].grid(True)

        if x2 is not None:
            yy = np.stack(
                [
                    x2[0].permute(i, j, *other).numpy(),
                    x2[1].permute(i, j, *other).numpy(),
                ],
                axis=-1,
            )
            yy = np.complex64(yy)
            yy_raspec = [rapsd(yy[:, :, *sl], fft_method=np.fft) for sl in slices]
            yy_raspec_avg = np.mean(yy_raspec, axis=0)
            ax[i, j].loglog(yy_raspec_avg, label="GT spec", c="k", lw=3)

        if i > imin:
            ax[i, j].set_ylabel(rf"${labels[i]}$ (A)", fontsize=20)
            ax[i, j].set_xlabel(rf"${labels[j]}$ ($\phi$)", fontsize=20)
            imin = i

    return plt_to_wandb_image(fig)


def plot_potentials(x1, x2):
    from matplotlib import colormaps

    c_map = colormaps["plasma"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.05)

    # select only real part if we predicted both real/imag parts of phi
    x1 = x1[0] if x1.ndim > 3 else x1
    x2 = x2[0] if x2.ndim > 3 else x2
    ax[0].matshow(x1.squeeze()[:, 8, :].T, cmap=c_map)
    ax[0].set_title(r"$\phi_{pred}$", fontsize=24)
    ax[0].set_ylabel(r"$y_{\phi}$", fontsize=20)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].matshow(x2.squeeze()[:, 8, :].T, cmap=c_map)
    ax[1].set_title(r"$\phi_{GT}$", fontsize=24)
    ax[1].set_xlabel(r"$x_{\phi}$", fontsize=20)
    ax[1].set_ylabel(r"$y_{\phi}$", fontsize=20)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    return plt_to_wandb_image(fig)


def avg_flux_confidence(pred_means, pred_stds, tgt_vals, traj_ids):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(traj_ids))
    ax.errorbar(
        x_pos,
        pred_means,
        yerr=pred_stds,
        fmt="o",
        capsize=5,
        label="predicted (mean ± std)",
        color="blue",
        alpha=0.7,
    )
    ax.plot(x_pos, tgt_vals, "rx", markersize=8, label="ground truth", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(traj_ids, rotation=45)
    ax.set_xlabel("trajectory id")
    ax.set_ylabel("average flux")
    ax.set_title("average flux predictions per trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return plt_to_wandb_image(fig)


def generate_val_plots(
    rollout: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    phase: str,
    ts: Optional[torch.Tensor] = None,
):
    plots = {}
    ts = f"T={ts[0].item():.2f}, " if ts is not None else ""
    val_plots_dict = {
        "df": {f"df ({ts}{phase})": plot4x4_sided},
        # "phi": {f"phi ({ts}{phase})": plot_potentials},
        # "phi_int": {f"phi int ({ts}{phase})": plot_potentials},
    }

    for key in rollout.keys():
        if key not in val_plots_dict:
            # skip flux
            continue

        gt_key = key
        if "int" in key:
            gt_key = key.replace("_int", "")

        x = rollout[key].clone()
        y = gt[gt_key].clone()

        if x.ndim != 7:
            # add dummy time dimension
            x = x.unsqueeze(0)

        if y.shape[0] != 2 and key == "df":
            y = torch.cat(
                [
                    y[0::2].sum(axis=0, keepdims=True),
                    y[1::2].sum(axis=0, keepdims=True),
                ],
                dim=0,
            )

        if x.shape[1] != 2 and key == "df":  # separate zonal flow, sum and recompose
            x = torch.cat(
                [
                    x[:, 0::2].sum(axis=1, keepdims=True),
                    x[:, 1::2].sum(axis=1, keepdims=True),
                ],
                dim=1,
            )

        for name, plot_fn in val_plots_dict[key].items():
            # x[0] for first rolled timestep (if it's there)
            plots[name] = plot_fn(x[0], x2=y)

    return plots
