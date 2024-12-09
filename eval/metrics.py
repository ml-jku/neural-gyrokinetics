import torch
from skimage.metrics import structural_similarity as ssim


def SSIM(prediction, ground_truth, keys, average=True):
    t_max_pred = prediction[keys[0]].shape[0]
    if average:
        out = {}
        for k in prediction.keys():
            data_range = max(
                prediction[k].max(), ground_truth[k][:t_max_pred, ...].max()
            ) - min(prediction[k].min(), ground_truth[k][:t_max_pred, ...].min())
            out[k] = ssim(
                prediction[k], ground_truth[k][:t_max_pred, ...], data_range=data_range
            )
    else:
        out = {}
        for k in prediction.keys():
            ssim_list = []
            for t in range(t_max_pred):
                data_range = max(
                    prediction[k][t].max(), ground_truth[k][t].max()
                ) - min(prediction[k][t].min(), ground_truth[k][t].min())
                ssim_value = ssim(
                    prediction[k][t], ground_truth[k][t], data_range=data_range
                )
                ssim_list.append(ssim_value)
            out[k] = torch.tensor(ssim_list)
    return out


def ssim_tensor(x, y):
    bs = x.shape[1]
    xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    tsteps = xn.shape[0]
    out = torch.zeros((tsteps, bs))
    for batch in range(bs):
        for t in range(tsteps):
            val = ssim(
                xn[t, batch],
                yn[t, batch],
                channel_axis=0,
                data_range=xn[t, batch].max() - yn[t, batch].min(),  # TODO why x and y?
            )
            out[t, batch] = torch.tensor(val)
    # average batches
    return torch.mean(out, dim=1)
