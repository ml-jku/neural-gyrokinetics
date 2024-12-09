from typing import Dict, List

import io

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from pysteps.utils import spectral

import torch
import wandb


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def print_shapes(data):
    for k in list(data.keys()):
        if data[k].ndim == 2:
            desc = "(x, y)"
        elif data[k].ndim == 3:
            desc = "(t, x, y)"
        else:
            desc = "(scalar)"
        print(f"{k} -> {desc} = {data[k].shape}")


def get_gifs(filename=None, rollout=None, name=""):
    matplotlib.use("Agg")
    if rollout is None:
        data = h5py.File(filename, "r")
    else:
        filename = f"rollout_{name}"
        data = rollout

    all_keys = list(data.keys())
    n_times = data["T"].shape[0]
    frames = []

    print_shapes(data)

    plot_vars = list(set(["rho", "T", "omega", "zj"]) & set(all_keys))
    colormaps = ["jet", "inferno", "seismic", "RdGy"]

    for i_time in tqdm(n_times):
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_axis_off()
        for i_var, var in enumerate(plot_vars):
            data_tmp = np.asarray(data[var][i_time])
            ax[i_var // 2, i_var % 2].imshow(data_tmp, cmap=colormaps[i_var])
            ax[i_var // 2, i_var % 2].set_title(var, fontsize=20)
        image = fig2img(fig)
        plt.close()
        frames.append(image)

    frame_one = frames[0]
    gif_filename = "gifs/" + filename.split(".h5")[0].split("/")[-1] + ".gif"
    frame_one.save(
        gif_filename,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=100,
        loop=0,
    )

    return gif_filename


def get_gifs3x3(
    pred,
    gt,
    out_file=None,
    colormaps=["jet", "inferno", "seismic", "RdGy"],
    use_tqdm=True,
):
    # dont display
    matplotlib.use("Agg")
    all_keys = list(pred.keys())
    n_times = pred[all_keys[0]].shape[0]
    frames = []

    # plot_vars = list(set(["rho", "T", "omega", "zj"]) & set(all_keys))
    plot_vars = all_keys
    N = len(plot_vars)
    if use_tqdm:
        tsteps = tqdm(range(n_times), "Creating gif")
    else:
        tsteps = range(n_times)
    for ts in tsteps:
        fig, ax = plt.subplots(3, N, figsize=(12, 12))
        for i in range(3):
            for j in range(N):
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                # ax[i, j].set_axis_off()
            ax[i, 0].set_ylabel(["Pred", "GT", "MSE"][i], fontsize=20)

        for i, var in enumerate(plot_vars):
            pred_ts = np.asarray(pred[var][ts])
            gt_ts = np.asarray(gt[var][ts])
            # set same vmin-vmax
            vmin = min(pred_ts.min(), gt_ts.min())
            vmax = max(pred_ts.max(), gt_ts.max())
            # pred on row 0
            ax[0, i].imshow(pred_ts, cmap=colormaps[i], vmax=vmax, vmin=vmin)
            ax[0, i].set_title(var, fontsize=20)
            # gt on row 1
            ax[1, i].imshow(gt_ts, cmap=colormaps[i], vmax=vmax, vmin=vmin)
            ax[1, i].set_title(var, fontsize=20)
            # mse on row 2
            mse = (gt_ts - pred_ts) ** 2
            ax[2, i].imshow(mse, cmap=colormaps[i])
            ax[2, i].set_title(var, fontsize=20)

        if out_file is not None:
            image = fig2img(fig)
            plt.close()
            frames.append(image)
        else:
            frames.append(fig)

    if out_file is not None:
        frame_one = frames[0]
        # TODO figure out animation duration
        frame_one.save(
            f"{out_file}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=100,
            loop=0,
        )
        print(f"Saving gif with {n_times} frames to {out_file}.gif")

        return f"{out_file}.gif"
    else:
        return frames


def get_val_trajectory_plots(
    prediction: torch.Tensor, ground_truth: torch.Tensor, tsteps: List[int]
) -> List[plt.Figure]:
    # reshape plotting_data in order to use get_gifs3x3
    pred_out = {}
    gt_out = {}
    for ts in tsteps:
        # take only the first trajectory of the batch
        pred_out[f"Timestep {ts}"] = prediction[ts - 1][0, ...].cpu()
        gt_out[f"Timestep {ts}"] = ground_truth[ts - 1][0, ...].cpu()

    return get_gifs3x3(
        pred_out, gt_out, use_tqdm=False, colormaps=["viridis"] * len(tsteps)
    )


def get_wandb_tables(
    metrics: torch.Tensor, metric_names: List[str]
) -> Dict[str, wandb.Table]:
    tables = {}
    # loop through each metric and create separate tables
    for metric_idx, metric in enumerate(metric_names):
        table = wandb.Table(columns=["t", metric])
        # populate the table with values
        for t in range(metrics.shape[1]):
            table.add_data(t + 1, metrics[metric_idx, t])
        tables[metric] = table

    return tables


def get_power_spectra(prediction: torch.Tensor, ground_truth: torch.Tensor):
    # calculate radially averaged power spectral density for the 1 step prediction
    bs = ground_truth[0].shape[0]
    problem_dims = ground_truth[0].shape[1]
    pred = prediction[0, ...]  # 1 step prediction
    gt = ground_truth[0, ...]  # 1 step groundtruth

    spectra_pred = []
    spectra_gt = []

    for sample_idx in range(bs):
        rapsd_pred_per_dim = []
        rapsd_gt_per_dim = []

        for problem_dim in range(problem_dims):
            rapsd_pred, frequencies = spectral.rapsd(
                pred[sample_idx, problem_dim, ...].cpu().numpy(),
                fft_method=np.fft,
                return_freq=True,
            )
            rapsd_pred_per_dim.append(rapsd_pred)

            rapsd_gt, frequencies = spectral.rapsd(
                gt[sample_idx, problem_dim, ...].cpu().numpy(),
                fft_method=np.fft,
                return_freq=True,
            )
            rapsd_gt_per_dim.append(rapsd_gt)

        # log-transform and average across the problem dimensions for this sample
        avg_log_rapsd_pred = np.mean(
            [np.log(s + 1e-30) for s in rapsd_pred_per_dim], axis=0
        )
        avg_log_rapsd_gt = np.mean(
            [np.log(s + 1e-30) for s in rapsd_gt_per_dim], axis=0
        )

        spectra_pred.append(avg_log_rapsd_pred)
        spectra_gt.append(avg_log_rapsd_gt)

    # calculate mean log RAPS for all samples
    mean_log_rapsd_pred = np.mean(spectra_pred, axis=0)
    mean_log_rapsd_gt = np.mean(spectra_gt, axis=0)

    return (
        frequencies[1:],
        np.exp(mean_log_rapsd_pred)[1:],
        np.exp(mean_log_rapsd_gt)[1:],
    )


def distribution_5D(x):
    labels = ["v1", "v2", "s", "x", "y"]

    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()

    comb = torch.combinations(torch.arange(5), 2).tolist()

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    c_map = matplotlib.colormaps["rainbow"]
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
        im1 = ax1.imshow(x1_plot.T, cmap=c_map)
        im2 = ax2.imshow(x2_plot.T, cmap=c_map)

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)

        if i == 0:
            # Set axis labels
            ax1.set_title("GT")
            ax2.set_title("PRED")

        ax1.set_xlabel(labels[i])
        ax1.set_ylabel(labels[j])
        ax2.set_xlabel(labels[i])
        ax2.set_ylabel(labels[j])

        # Force aspect ratio
        force_aspect(ax1)
        force_aspect(ax2)

    return fig
