import os
import numpy as np
import torch

import matplotlib
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt


directory = "/restricteddata/ukaea/gyrokinetics/raw/cyclone4_2_2"


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot4x4_3D(x, title="", mode="mean"):
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]
    comb = torch.combinations(torch.arange(5), 3).tolist()

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(title)

    cmap = "RdBu_r"

    for idx, (i, j, k) in enumerate(comb):
        ax_main = fig.add_subplot(4, 4, idx + 1, projection="3d")
        other = tuple([o for o in range(5) if o != i and o != j and o != k])

        if mode == "mean":
            xx = x.mean(other)
        elif mode == "std":
            cmap = "plasma"
            xx = x.std(other)
        else:
            xx = x.permute(i, j, k, *other)
            slice_point_1 = xx.shape[3] // 2
            slice_point_2 = xx.shape[4] // 2
            xx = xx[..., slice_point_1, slice_point_2]

            ax_imshow = inset_axes(
                ax_main, width="30%", height="30%", loc="upper right", borderpad=-1.5
            )
            sliced_dims = x.mean((i, j, k)).numpy().T
            ax_imshow.matshow(sliced_dims, cmap="viridis", origin="lower")
            ax_imshow.scatter(slice_point_1, slice_point_2, color="red", marker="x")
            ax_imshow.set_xticks([])
            ax_imshow.set_yticks([])
            ax_imshow.grid(False)
            ax_imshow.set_xlabel(rf"${labels[other[0]]}$", fontsize=12)
            ax_imshow.set_ylabel(rf"${labels[other[1]]}$", fontsize=12)
            force_aspect(ax_imshow)

        X, Y, Z = np.meshgrid(
            np.arange(xx.shape[0]),
            np.arange(xx.shape[1]),
            np.arange(xx.shape[2]),
            indexing="ij",
        )
        xx = xx.numpy()

        ax_main.scatter(
            X.flatten(), Y.flatten(), Z.flatten(), c=xx.flatten(), cmap=cmap, alpha=0.4
        )

        # Add axis labels
        ax_main.set_xlabel(rf"${labels[i]}$", fontsize=20)
        ax_main.set_ylabel(rf"${labels[j]}$", fontsize=20)
        ax_main.set_zlabel(rf"${labels[k]}$", fontsize=20)

        # Remove ticks and gridlines
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_zticks([])
        ax_main.grid(False)

    return fig


def stuff(fname, mode="mean", slice="mid", fft_norm="ortho"):
    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    krho = np.loadtxt(os.path.join(directory, "krho"))
    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]

    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

    with open(os.path.join(directory, fname), "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)

    x = (
        np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky), order="F")
        .copy()
        .astype("float32")
    )

    def do_ifft(x, sl):
        x = np.ascontiguousarray(np.moveaxis(x, 0, -1))
        x = x.view(dtype=np.complex64)
        x = np.fft.fftshift(x, axes=(3, 4))
        x[..., sl, :] = 0.0
        x = np.fft.ifftn(x, axes=(3, 4), norm=fft_norm)
        x = np.stack([x.real, x.imag]).squeeze().astype("float32")
        return torch.from_numpy(x)

    x_zf = do_ifft(x.copy(), np.arange(1, 32))
    x_turb1 = do_ifft(x.copy(), 0)

    plot4x4_3D(x_zf[0], title="ZF", mode=mode)
    plot4x4_3D(x_turb1[0], title="TURB", mode=mode)


def velocity_space_sample_3D(x, title=""):
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(title)

    fixed_axes = (2, 3, 4)
    slice_axes = (0, 1)

    global_min = np.inf
    global_max = -np.inf

    npar, nmu = 8, 2
    slice_idx = [
        [int(v_par), int(v_mu)]
        for v_par in np.linspace(9, 23, npar)
        for v_mu in np.linspace(1, 3, nmu)
    ]

    for idx in range(16):
        slice_point_0, slice_point_1 = slice_idx[idx]
        xx = x[slice_point_0, slice_point_1, :, :, :].numpy()
        global_min = min(global_min, xx.min())
        global_max = max(global_max, xx.max())

    cmap = "RdBu_r"

    for idx in range(16):
        ax_main = fig.add_subplot(4, 4, idx + 1, projection="3d")

        slice_point_0, slice_point_1 = slice_idx[idx]
        xx = x[slice_point_0, slice_point_1, :, :, :].numpy()

        ax_imshow = inset_axes(
            ax_main, width="30%", height="30%", loc="upper right", borderpad=-2.5
        )
        sliced_dims = x.mean(fixed_axes).T
        ax_imshow.matshow(sliced_dims, cmap="viridis", origin="lower")
        ax_imshow.scatter(slice_point_0, slice_point_1, color="red", marker="x")
        ax_imshow.set_xticks([])
        ax_imshow.set_yticks([])
        ax_imshow.grid(False)
        force_aspect(ax_imshow)

        X, Y, Z = np.meshgrid(
            np.arange(xx.shape[0]),
            np.arange(xx.shape[1]),
            np.arange(xx.shape[2]),
            indexing="ij",
        )

        scatter = ax_main.scatter(
            X.flatten(),
            Y.flatten(),
            Z.flatten(),
            c=xx.flatten(),
            cmap=cmap,
            alpha=0.4,
            vmin=global_min,
            vmax=global_max,
        )

        ax_main.set_xlabel(rf"${labels[fixed_axes[0]]}$", fontsize=20)
        ax_main.set_ylabel(rf"${labels[fixed_axes[1]]}$", fontsize=20)
        ax_main.set_zlabel(rf"${labels[fixed_axes[2]]}$", fontsize=20)

        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.set_zticks([])
        ax_main.grid(False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax)

    return fig


def plot4x4_sided4times(x1, x2, x3, x4, average=True):
    labels = [r"v_{par}", r"v_{\mu}", r"s", r"k_x", r"k_y"]
    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(36, 20))
    for i in range(5):
        for j in range(5):
            if j == 0 or i == 4:
                ax[i, j].remove()
                continue
            ax_ij = ax[i, j]
            ax_ij.set_frame_on(False)
            ax_ij.tick_params(labelleft=False, labelbottom=False)
            ax_ij.set_xticks([])
            ax_ij.set_yticks([])

    c_map = matplotlib.colormaps["RdBu"]

    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])

        if average:
            x1_plot = x1[0].mean(other)
            x2_plot = x2[0].mean(other)
            x3_plot = x3[0].mean(other)
            x4_plot = x4[0].mean(other)
        else:
            x1_plot = x1[0].permute(i, j, *other).numpy()
            x1_plot = x1_plot[
                :,
                :,
                x1_plot.shape[2] // 2,
                x1_plot.shape[3] // 2,
                x1_plot.shape[4] // 2,
            ]
            x2_plot = x2[0].permute(i, j, *other).numpy()
            x2_plot = x2_plot[
                :,
                :,
                x2_plot.shape[2] // 2,
                x2_plot.shape[3] // 2,
                x2_plot.shape[4] // 2,
            ]
            x3_plot = x3[0].permute(i, j, *other).numpy()
            x3_plot = x3_plot[
                :,
                :,
                x3_plot.shape[2] // 2,
                x3_plot.shape[3] // 2,
                x3_plot.shape[4] // 2,
            ]
            x4_plot = x4[0].permute(i, j, *other).numpy()
            x4_plot = x4_plot[
                :,
                :,
                x4_plot.shape[2] // 2,
                x4_plot.shape[3] // 2,
                x4_plot.shape[4] // 2,
            ]

        ax_ij = ax[i, j]
        pos = ax_ij.get_position()

        # Define the positions for four subplots in a 2x2 layout
        plot_width = 0.45 * pos.width
        plot_height = 0.45 * pos.height
        x_left_1 = pos.x0
        x_left_2 = x_left_1 + plot_width
        y_bottom_1 = pos.y0
        y_bottom_2 = y_bottom_1 + plot_height

        ax1 = fig.add_axes([x_left_1, y_bottom_2, plot_width, plot_height])  # Top-left
        ax2 = fig.add_axes([x_left_2, y_bottom_2, plot_width, plot_height])  # Top-right
        ax3 = fig.add_axes(
            [x_left_1, y_bottom_1, plot_width, plot_height]
        )  # Bottom-left
        ax4 = fig.add_axes(
            [x_left_2, y_bottom_1, plot_width, plot_height]
        )  # Bottom-right

        # Compute shared vmin and vmax
        vmin = min(x1_plot.min(), x2_plot.min(), x3_plot.min(), x4_plot.min())
        vmax = max(x1_plot.max(), x2_plot.max(), x3_plot.max(), x4_plot.max())

        im1 = ax1.matshow(x1_plot, cmap=c_map, vmin=vmin, vmax=vmax)
        im2 = ax2.matshow(x2_plot, cmap=c_map, vmin=vmin, vmax=vmax)
        im3 = ax3.matshow(x3_plot, cmap=c_map, vmin=vmin, vmax=vmax)
        im4 = ax4.matshow(x4_plot, cmap=c_map, vmin=vmin, vmax=vmax)

        # Shared colorbar
        cbar = fig.colorbar(
            im1, ax=[ax_ij], format=tkr.FormatStrFormatter("%.2g"), pad=0, fraction=0.05
        )
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.ax.tick_params(labelsize=12)

        if i == 0:
            ax1.set_title("ZF (0)", fontsize=18)
            ax2.set_title("[1:10]", fontsize=18)
            ax3.set_title("[11:20]", fontsize=18)
            ax4.set_title("[21:31]", fontsize=18)

        if j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3) or (i == 3 and j == 4):
            ax_ij.set_ylabel(rf"${labels[i]}$", fontsize=14)

        if i == 3 or j == 1 or (i == 1 and j == 2) or (i == 2 and j == 3):
            ax_ij.set_xlabel(rf"${labels[j]}$", fontsize=14)

        # Remove axis ticks and labels
        for axx in [ax1, ax2, ax3, ax4]:
            axx.set_xticks([])
            axx.set_yticks([])
            axx.tick_params(labelleft=False, labelbottom=False)
            force_aspect(axx)

    return fig


def plot4x4_sided(x1, x2, mark_bad=False, average=True):
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

    c_map = matplotlib.colormaps["RdBu"]
    c_map.set_bad("k")

    for i, j in comb:
        other = tuple([o for o in range(5) if o != i and o != j])

        if average:
            x1_plot = x1[0].mean(other)
            x2_plot = x2[0].mean(other)
        else:
            x1_plot = x1[0].permute(i, j, *other).numpy()[:, :, 0, 0, 0]
            x2_plot = x2[0].permute(i, j, *other).numpy()[:, :, 0, 0, 0]

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
            ax1.set_title("ZF", fontsize=24)
            ax2.set_title("TURB", fontsize=24)

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

    return fig


def frequency_filter_4x4(fname, mode="mean", filter_method="fft"):
    # load data files
    sgrid = np.loadtxt(os.path.join(directory, "sgrid"))
    krho = np.loadtxt(os.path.join(directory, "krho"))
    vpgr = np.loadtxt(os.path.join(directory, "vpgr.dat"))
    ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]

    nkx, nky = krho.shape[1], krho.shape[0]
    nvpar, nmu = vpgr.shape[1], vpgr.shape[0]

    with open(os.path.join(directory, fname), "rb") as fid:
        ff = np.fromfile(fid, dtype=np.float64)

    x = (
        np.reshape(ff, (2, nvpar, nmu, ns, nkx, nky), order="F")
        .copy()
        .astype("float32")
    )

    def do_ifft(x, sl=None):
        x = np.ascontiguousarray(np.moveaxis(x, 0, -1))
        x = x.view(dtype=np.complex64)
        x = np.fft.fftshift(x, axes=(3, 4))
        if sl is not None:
            x[..., sl, :] = 0.0
        x = np.fft.ifftn(x, axes=(3, 4))
        x = np.stack([x.real, x.imag]).squeeze().astype("float32")
        return x

    if filter_method == "fft":

        x_zf = do_ifft(x.copy(), np.arange(1, 32))
        x_turb1 = do_ifft(x.copy(), [0] + list(range(11, 32)))
        x_turb2 = do_ifft(x.copy(), list(range(0, 10)) + list(range(22, 32)))
        x_turb3 = do_ifft(x.copy(), list(range(0, 22)))

    if filter_method == "gauss":
        from scipy.ndimage import gaussian_filter

        x = do_ifft(x)
        x = x / x.std()
        # apply separate on real / imag
        x_zf = x.copy()
        x_zf[0] = gaussian_filter(x_zf[0], sigma=(2.0, 2.0, 2.0, 4.5, 4.5))
        # x_zf[1] = gaussian_filter(x_zf[1], sigma=0.3)

        x_turb2 = x.copy()
        x_turb2[0] = gaussian_filter(x_turb2[0], sigma=(0.3, 0.3, 0.3, 0.8, 0.8))
        # x_turb2[1] = gaussian_filter(x_turb2[1], sigma=2.0)

        # band pass via gaussian difference
        x_turb1 = x - x_zf - x_turb2
        x_turb3 = x_zf + x_turb1 + x_turb2

    return plot4x4_sided4times(
        x_zf[0][None],
        x_turb1[0][None],
        x_turb2[0][None],
        x_turb3[0][None],
        average=(mode == "mean"),
    )


def plot3D(
    x,
    fixed_axes=(2, 3),
    title="",
    subs_2d=(1, 1),
    subs_3d=(1, 1, 1),
    alpha=1.0,
    cmap="RdBu_r",
    bg_alpha=0.2,
    surface_slices=None,  # Number of Z-slices to show as surfaces
    edgecolor="white",  # Color of the surface grid lines
    linewidth=0.3,  # Width of the grid lines
):
    # Validate subsampling parameters
    if len(subs_2d) != 2:
        raise ValueError("subs_2d must be a tuple of length 2")
    if len(subs_3d) != 3:
        raise ValueError("subs_3d must be a tuple of length 3")

    fixed_axis1, fixed_axis2 = fixed_axes
    slice_axes = [i for i in range(5) if i not in fixed_axes]

    # Determine grid size with axis-specific subsampling
    n1, n2 = x.shape[fixed_axis1], x.shape[fixed_axis2]
    plane_indices1 = range(0, n1, subs_2d[0])
    plane_indices2 = range(0, n2, subs_2d[1])
    n_rows = len(plane_indices1)
    n_cols = len(plane_indices2)

    fig = plt.figure(figsize=(2 * len(plane_indices2), 2 * len(plane_indices1)))

    # Find global min/max for consistent coloring
    global_min = np.inf
    global_max = -np.inf
    global_mean_min = np.inf
    global_mean_max = -np.inf
    slice_means = []  # Store means for background colors

    for i in plane_indices1:
        for j in plane_indices2:
            idx = [slice(None)] * 5
            idx[fixed_axis1] = i
            idx[fixed_axis2] = j
            xx = x[tuple(idx)].numpy()
            global_min = min(global_min, xx.min())
            global_max = max(global_max, xx.max())
            slice_means.append(xx.mean())
            global_mean_min = min(global_mean_min, xx.mean())
            global_mean_max = max(global_mean_max, xx.mean())

    norm = plt.Normalize(global_mean_min, global_mean_max)

    def make_faint_color(mean_val):
        rgba = matplotlib.colormaps[cmap](norm(mean_val))
        return (rgba[0], rgba[1], rgba[2], bg_alpha)

    bg_colors = [make_faint_color(mean) for mean in slice_means]

    # Create a gridspec with no spacing between subplots
    gs = fig.add_gridspec(
        n_rows, n_cols, left=0, right=1, bottom=0, top=1, wspace=0, hspace=0
    )

    # Create all plots
    plot_idx = 0
    for i, row in enumerate(plane_indices1):
        for j, col in enumerate(plane_indices2):
            ax = fig.add_subplot(
                gs[i, j], projection="3d", facecolor=bg_colors[plot_idx], alpha=bg_alpha
            )
            plot_idx += 1

            # Get the data slice
            idx = [slice(None)] * 5
            idx[fixed_axis1] = row
            idx[fixed_axis2] = col
            xx = x[tuple(idx)].numpy()

            # Subsample the data
            xx = xx[:: subs_3d[0], :: subs_3d[1], :: subs_3d[2]]

            # Create meshgrid for X and Y
            X, Y = np.meshgrid(
                np.arange(0, xx.shape[0]), np.arange(0, xx.shape[1]), indexing="ij"
            )

            # Determine which Z-slices to plot as surfaces
            if surface_slices is None:
                # Default to showing 3 surfaces (min, middle, max)
                z_indices = [0, xx.shape[2] // 2, xx.shape[2] - 1]
            else:
                z_indices = np.linspace(0, xx.shape[2] - 1, surface_slices, dtype=int)

            # Plot each surface with decreasing alpha based on Z position
            for k, z in enumerate(z_indices):
                Z_val = np.full_like(X, z)
                ax.plot_surface(
                    X,
                    Y,
                    Z_val,
                    facecolors=matplotlib.colormaps[cmap](
                        plt.Normalize(global_min, global_max)(xx[:, :, z])
                    ),
                    edgecolor=edgecolor,  # Set grid line color
                    linewidth=linewidth,  # Set grid line width
                    alpha=alpha * (1 - 0.2 * k),  # Slightly fade further surfaces
                    shade=False,
                    zorder=z,
                )

            # Remove all axes and frames
            ax.set_axis_off()
            ax.grid(False)

    # Adjust the suptitle position to account for no padding
    fig.subplots_adjust(top=0.95 if title else 1.0)

    return fig