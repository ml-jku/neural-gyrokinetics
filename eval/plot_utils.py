import io
from PIL import Image as PILImage
import matplotlib
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import numpy as np
import wandb
from einops import rearrange

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
    labels = ["vpar", "vmu", "s", "x", "y"]

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
            ax[i, j].set_ylabel(labels[i], fontsize=20)
            ax[i, j].set_xlabel(labels[j], fontsize=20)
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


def to_fourier(x_rollout, y):
    # TODO tmp, move somewhere else
    x_rollout = rearrange(x_rollout, "t b c ... -> c t b ...")
    x_rollout = torch.complex(real=x_rollout[0], imag=x_rollout[1])
    x_rollout = torch.fft.fftn(x_rollout, dim=(-2, -1))
    x_rollout = torch.stack([x_rollout.real, x_rollout.imag])
    x_rollout = rearrange(x_rollout, "c t b ... -> t b c ...")

    if y.ndim == 8:
        y = rearrange(y, "t b c ... -> c t b ...")
    else:
        y = rearrange(y, "b c ... -> c b ...")

    y = torch.complex(real=y[0], imag=y[1])
    y = torch.fft.fftn(y, dim=(-2, -1))
    y = torch.stack([y.real, y.imag]).squeeze()

    if y.ndim == 8:
        y = rearrange(y, "c t b ... -> t b c ...")
    else:
        y = rearrange(y, "c b ... -> b c ...")

    return x_rollout, y


def generate_val_plots(x_rollout, y, ts, phase, val_plots):
    val_plots_dict = {
        f"pred (T={ts[0].item():.2f}, {phase})": plot4x4_sided,
        f"std (T={ts[0].item():.2f}, {phase})": distribution_5D,
    }
    for name, plot_fn in val_plots_dict.items():
        # first timestep and batch
        y_first = (y[0] if y.ndim == 7 else y[0, 0]).to("cpu")
        val_plots[name] = plot_fn(
            x_rollout[0, 0],
            x2=y_first,
        )
