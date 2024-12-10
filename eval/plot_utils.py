import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def distribution_5D(x):
    labels = ["v1", "v2", "s", "x", "y"]

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
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

    for i in range(5):
        for j in range(5):
            if [i, j] not in comb:
                ax[i, j].remove()
    return fig


def plot4x4_sided(x1, x2, title="", mark_bad=False, average=True):
    labels = ["v1", "v2", "s", "x", "y"]
    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(30, 12))
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
        width = pos.width / 2  # Split the width into two halves
        ax1 = fig.add_axes([pos.x0, pos.y0, width, pos.height])
        ax2 = fig.add_axes([pos.x0 + width, pos.y0, width, pos.height])

        # Plot x1 and xp side by side
        im1 = ax1.matshow(x1_plot, cmap=c_map)
        im2 = ax2.matshow(x2_plot, cmap=c_map)

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)

        if i == 0:
            # Set axis labels
            ax1.set_title("GT")
            ax2.set_title("PRED")

        ax1.set_xlabel(labels[j])
        ax1.set_ylabel(labels[i])
        ax2.set_xlabel(labels[j])
        ax2.set_ylabel(labels[i])

        # Force aspect ratio
        force_aspect(ax1)
        force_aspect(ax2)

    return fig
