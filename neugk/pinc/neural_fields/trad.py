import torch
import numpy as np
from einops import rearrange

import os
import io
import pywt
import zfpy
from sklearn.decomposition import PCA
import glymur


def zfp_recon(df: torch.Tensor, tolerance: float = 2500.0):
    vp, s = df.shape[1], df.shape[3]
    df = rearrange(df, "c vp vm s x y -> c (vp vm) (s y) x").cpu().numpy()

    zfp_compressed = zfpy.compress_numpy(df, tolerance=tolerance)
    zf_df = zfpy.decompress_numpy(zfp_compressed)
    zf_df = rearrange(zf_df, "c (vp vm) (s y) x -> c vp vm s x y", vp=vp, s=s)

    compressed_size = len(zfp_compressed)
    # compression_ratio = df.nbytes / compressed_size
    # print(f"ZFP compression ratio: {compression_ratio:.2f}x")
    return torch.from_numpy(zf_df), zfp_compressed, compressed_size


def wavelet_recon(df: torch.Tensor, threshold: float = 28.0, level: int = 1):
    df = df.cpu().numpy()
    vp, vm, s, x, y = df.shape[1:]

    coeffs = []
    for c in range(2):
        dec = pywt.wavedecn(df[c], wavelet="haar", mode="periodization", level=level)
        coeff, slices = pywt.coeffs_to_array(dec)
        coeff[np.abs(coeff) < threshold] = 0
        coeffs.append((coeff, slices))

    wt_df = []
    for coeff, slices in coeffs:
        recon = pywt.array_to_coeffs(coeff, slices, output_format="wavedecn")
        recon = pywt.waverecn(recon, wavelet="haar", mode="periodization")
        wt_df.append(recon)

    wt_df = np.stack(wt_df, axis=0)
    wt_df = wt_df[:, :vp, :vm, :s, :x, :y]

    compressed_size = sum(c[0][c[0].astype(bool)].nbytes for c in coeffs)
    # compression_ratio = df.nbytes / compressed_size
    # print(f"Wavelet compression ratio: {compression_ratio:.2f}x")
    return torch.from_numpy(wt_df), coeffs, compressed_size


def pca_recon(df: torch.Tensor, n_components: int = 2, level: int = 1):
    vp, vm, s, x, y = df.shape[1:]
    if level == 1:
        df = rearrange(df, "c vp vm s x y -> c (vp vm s) (x y)").cpu().numpy()
    if level == 2:
        df = rearrange(df, "c vp vm s x y -> c (vp vm s y) x").cpu().numpy()
    if level == 3:
        df = rearrange(df, "c vp vm s x y -> c (vp vm s x y)").cpu().numpy()

    pca_results = []
    compressed = []
    compressed_size = 0

    for c in range(2):
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(df[c])
        reconstructed = pca.inverse_transform(transformed)
        pca_results.append(reconstructed)

        # components + PCA parameters
        compressed_version = {
            "components": transformed,
            "mean": pca.mean_,
            "explained_variance": pca.explained_variance_,
        }
        compressed.append(compressed_version)
        compressed_size += (
            transformed.nbytes + pca.mean_.nbytes + pca.explained_variance_.nbytes
        )

    pca_df = np.stack(pca_results, axis=0)
    if level == 1:
        pca_df = rearrange(
            pca_df,
            "c (vp vm s) (x y) -> c vp vm s x y",
            vp=vp,
            vm=vm,
            s=s,
            x=x,
            y=y,
        )
    if level == 2:
        pca_df = rearrange(
            pca_df,
            "c (vp vm s y) x -> c vp vm s x y",
            vp=vp,
            vm=vm,
            s=s,
            x=x,
            y=y,
        )
    if level == 3:
        pca_df = rearrange(
            pca_df,
            "c (vp vm s x y) -> c vp vm s x y",
            vp=vp,
            vm=vm,
            s=s,
            x=x,
            y=y,
        )

    # compression_ratio = df.nbytes / compressed_size
    # print(f"PCA compression ratio: {compression_ratio:.2f}x")
    return torch.from_numpy(pca_df), compressed, compressed_size


def quantization_recon(df: torch.Tensor, num_bits: int = 8):
    df = df.cpu().numpy()
    flat = df.flatten()

    min_val, max_val = flat.min(), flat.max()
    q_levels = 2**num_bits
    norm = (flat - min_val) / (max_val - min_val + 1e-12)
    quantized = np.round(norm * (q_levels - 1)).astype(np.uint16)

    compressed = {
        "quantized": quantized,
        "min_val": float(min_val),
        "max_val": float(max_val),
        "num_bits": num_bits,
        "shape": df.shape,
    }

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **compressed)
    compressed_bytes = buffer.getvalue()
    compressed_size = len(compressed_bytes)

    dequantized = quantized.astype(np.float32) / (q_levels - 1)
    recon = dequantized * (max_val - min_val) + min_val
    recon = recon.reshape(df.shape)

    return torch.from_numpy(recon), compressed, compressed_size


def jpeg2000_recon(df: torch.Tensor, quality: float = 0.2):
    c, vp, vm, _, x, _ = df.shape
    df_np = df.cpu().numpy()

    df_flat = rearrange(df_np, "c vp vm s x y -> c (vp vm s) (x y)")

    compressed_data = []
    compressed_size = 0.0
    recon_flat = np.zeros_like(df_flat)

    for ch in range(c):
        slice_ = df_flat[ch]

        mn, mx = slice_.min(), slice_.max()
        if mx - mn == 0:
            norm_slice = np.zeros_like(slice_)
        else:
            norm_slice = (slice_ - mn) / (mx - mn)

        img_uint16 = (norm_slice * 65535).astype(np.uint16)

        # NOTE: does not work with tempfile or io buffer for some reason
        try:
            os.remove("/tmp/df.jp2")
        except OSError:
            pass
        glymur.Jp2k("/tmp/df.jp2", data=img_uint16, cratios=[100.0 / quality])
        compressed_data.append({"bytes": None, "min": mn, "max": mx})
        compressed_size += os.path.getsize("/tmp/df.jp2")
        jp2 = glymur.Jp2k("/tmp/df.jp2")
        recon_uint16 = jp2[:]
        recon_norm = recon_uint16.astype(np.float32) / 65535.0
        recon_flat[ch] = recon_norm * (mx - mn) + mn
        try:
            os.remove("/tmp/df.jp2")
        except OSError:
            pass

    recon_np = rearrange(
        recon_flat, "c (vp vm s) (x y) -> c vp vm s x y", vp=vp, vm=vm, x=x
    )
    return torch.from_numpy(recon_np), compressed_data, compressed_size
