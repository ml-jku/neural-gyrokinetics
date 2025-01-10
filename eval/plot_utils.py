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
    labels = ["vpar", "vmu", "s", "x", "y"]
    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(30, 14))
    for i in range(5):
        for j in range(5):
            ax[i, j].remove()

    fig.tight_layout()
    fig.suptitle(title)
    c_map = matplotlib.colormaps["coolwarm"]
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

        # Clear the axis and directly plot two images side by side
        ax_ij = ax[i, j]

        # Get the position of the original axis
        pos = ax_ij.get_position()

        # Create two new axes within the same space as the original subplot
        displ = pos.width / 2
        width = 0.92 * (displ)  # Split the width into two halves
        ax1 = fig.add_axes([pos.x0, pos.y0, width, pos.height])
        ax2 = fig.add_axes([pos.x0 + displ, pos.y0, width, pos.height])

        # Plot x1 and xp side by side
        im1 = ax1.matshow(x1_plot, cmap=c_map)
        im2 = ax2.matshow(x2_plot, cmap=c_map)

        cbar1 = fig.colorbar(im1, ax=ax1, format=tkr.FormatStrFormatter("%.2g"))
        cbar2 = fig.colorbar(im2, ax=ax2, format=tkr.FormatStrFormatter("%.2g"))

        cbar1.set_ticks([x1_plot.min(), x1_plot.max()])
        cbar2.set_ticks([x2_plot.min(), x2_plot.max()])

        if i == 0:
            # Set axis labels
            ax1.set_title("PRED")
            ax2.set_title("GT")

        ax1.set_xlabel(labels[j])
        ax1.set_ylabel(labels[i])
        ax2.set_xlabel(labels[j])
        # ax2.set_ylabel(labels[i])

        # Force aspect ratio
        force_aspect(ax1)
        force_aspect(ax2)

        # Set only one tick at the maximum value for both axes
        ax1.set_xticks([x1_plot.shape[1] - 1])
        ax1.set_yticks([x1_plot.shape[0] - 1])
        ax2.set_xticks([x2_plot.shape[1] - 1])
        ax2.set_yticks([x2_plot.shape[0] - 1])

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
    x_rollout = torch.stack([x_rollout.real, x_rollout.imag]).squeeze()
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
