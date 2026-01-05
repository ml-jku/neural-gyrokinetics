from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from collections import defaultdict


class ComplexMetrics:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def to_complex(self, tensor):
        """convert tensor from [BS, C, ...] to complex representation"""
        if tensor.dtype == torch.bfloat16:
            # BFloat16 can not be converted to complex directly, convert to float32
            tensor = tensor.float()

        channels = tensor.shape[1]
        if channels == 2:
            return torch.complex(tensor[:, 0], tensor[:, 1])
        elif channels % 2 == 0:  # if zonal flow is split
            # Sum even indices (real parts: 0, 2, 4, ...) and odd indices (imag parts: 1, 3, 5, ...)
            real_parts = tensor[:, 0::2].sum(dim=1)
            imag_parts = tensor[:, 1::2].sum(dim=1)
            return torch.complex(real_parts, imag_parts)
        else:
            raise ValueError(
                f"Expected even number of channels for complex conversion, got {channels}"
            )

    def complex_pearson(self, z1, z2, dims=None):
        """complex Pearson correlation coefficient"""
        if dims is None:
            dims = list(range(1, z1.dim()))

        # compute mean
        mu1 = z1.mean(dim=dims, keepdim=True)
        mu2 = z2.mean(dim=dims, keepdim=True)

        # normalize
        z1_c = z1 - mu1
        z2_c = z2 - mu2

        z1_c_d = z1_c.to(torch.complex64)
        z2_c_d = z2_c.to(torch.complex64)

        # complex covariance
        cov = (z1_c_d * z2_c_d.conj()).mean(dim=dims)

        # variance
        var1 = (z1_c_d.abs() ** 2).mean(dim=dims)
        var2 = (z2_c_d.abs() ** 2).mean(dim=dims)

        # calculate Pearson correlation coefficient
        denom = torch.sqrt(var1 * var2 + self.epsilon)
        corr = cov / denom

        # convert back to original dtype
        corr = corr.to(z1.dtype)

        # for identical tensors, return ones for magnitude and zeros for phase
        if torch.allclose(z1, z2, rtol=1e-10, atol=1e-10):
            return torch.ones_like(corr.abs()), torch.zeros_like(corr.angle())

        return corr.abs(), corr.angle()

    def phase_locking_value(self, z1, z2, dims=None):
        """Phase Locking Value (PLV)"""
        if dims is None:
            dims = list(range(1, z1.dim()))

        # create mask for non-zero elements to avoid division by zero
        mask = (z1.abs() > self.epsilon) & (z2.abs() > self.epsilon)

        # normalize to unit vectors (extract phase) only for non-zero elements
        z1_unit = torch.where(mask, z1 / z1.abs(), torch.zeros_like(z1))
        z2_unit = torch.where(mask, z2 / z2.abs(), torch.zeros_like(z2))

        # count valid elements for proper averaging
        valid_count = mask.sum(dim=dims, keepdim=True).float()

        # phase difference
        phase_diff = z1_unit * z2_unit.conj()

        # average only over valid elements
        plv = torch.where(
            valid_count > 0,
            phase_diff.sum(dim=dims).abs() / valid_count.squeeze(),
            torch.ones_like(valid_count.squeeze()),  # If no valid elements, return 1
        )

        return plv

    def complex_ssim(self, z1, z2, dims=None, c1=0.01, c2=0.03):
        """Complex Structural Similarity Index (CSSIM)"""
        if dims is None:
            dims = list(range(1, z1.dim()))

        # Means
        mu1 = z1.mean(dim=dims)
        mu2 = z2.mean(dim=dims)

        # For variance calculation, we need to keepdim for broadcasting
        mu1_keepdim = z1.mean(dim=dims, keepdim=True)
        mu2_keepdim = z2.mean(dim=dims, keepdim=True)

        # Variances and covariance
        var1 = ((z1 - mu1_keepdim).abs() ** 2).mean(dim=dims)
        var2 = ((z2 - mu2_keepdim).abs() ** 2).mean(dim=dims)
        cov12 = ((z1 - mu1_keepdim) * (z2 - mu2_keepdim).conj()).mean(dim=dims)

        # SSIM components
        mu1_abs = mu1.abs()
        mu2_abs = mu2.abs()

        # Dynamic range for c1 and c2 (based on data range)
        data_range = max(z1.abs().max().item(), z2.abs().max().item())
        c1 = (c1 * data_range) ** 2
        c2 = (c2 * data_range) ** 2

        numerator = (2 * mu1_abs * mu2_abs + c1) * (2 * cov12.abs() + c2)
        denominator = (mu1_abs**2 + mu2_abs**2 + c1) * (var1 + var2 + c2)

        ssim = numerator / denominator

        return ssim

    def complex_mse(self, z1, z2, dims=None):
        """Complex Mean Squared Error"""
        if dims is None:
            dims = list(range(1, z1.dim()))

        diff = z1 - z2
        cmse = torch.mean(diff.real**2 + diff.imag**2, dim=dims)
        return cmse.mean()

    def complex_l1(self, z1, z2, dims=None):
        """Complex L1 norm (mean absolute error)"""
        if dims is None:
            dims = list(range(1, z1.dim()))

        diff = z1 - z2
        l1 = torch.mean(diff.real.abs() + diff.imag.abs(), dim=dims)
        return l1.mean()

    def spectral_energy_metric(self, z1, z2, dims=None):
        """Spectral energy difference metric."""
        if dims is None:
            dims = list(range(1, z1.dim()))

        # Compute spectral energy (sum of squared magnitudes)
        energy1 = (z1.abs() ** 2).sum(dim=dims)
        energy2 = (z2.abs() ** 2).sum(dim=dims)

        # Relative energy difference
        energy_diff = torch.abs(energy1 - energy2)
        energy_denom = energy1 + self.epsilon

        return (energy_diff / energy_denom).mean()

    def _to_spectral_domain(self, z, spatial_dims=(-2, -1)):
        """
        Transform complex tensor to spectral domain using FFT
            1. Given f(x,y) in spatial domain
            2. Apply 2D FFT: F̂(kₓ,kᵧ) = ∫∫ f(x,y) e^(-i(kₓx + kᵧy)) dx dy
            3. Apply fftshift for proper frequency ordering: DC at center

        Args:
            z: Complex tensor [..., x, y] (supports 5D data: [BS, v_par, mu, s, x, y])
            spatial_dims: Dimensions to transform (default: last 2 dimensions)
        """
        # Apply FFT to spatial dimensions following existing codebase pattern
        z_fft = torch.fft.fftn(z, dim=spatial_dims, norm="forward")

        # Apply fftshift for proper frequency ordering (following train/integrals.py)
        z_fft = torch.fft.fftshift(z_fft, dim=spatial_dims)

        return z_fft

    def magnitude_weighted_phase_coherence(self, z1, z2, spatial_dims=(-2, -1)):
        """
        Magnitude-weighted phase coherence (MWPC) for complex-valued data
            Step 1: Extract phases: φ₁(k) = arg(F̂₁(k)), φ₂(k) = arg(F̂₂(k))
            Step 2: Phase difference: Δφ(k) = φ₁(k) - φ₂(k)
            Step 3: Magnitude weights: w(k) = |F̂₁(k)| · |F̂₂(k)|
            Step 4: Weighted phase coherence:
                    γw = |Σₖ w(k) · e^{iΔφ(k)}| / Σₖ w(k)
        """
        z1_fft = self._to_spectral_domain(z1, spatial_dims)
        z2_fft = self._to_spectral_domain(z2, spatial_dims)

        # Extract magnitude weights
        weights = z1_fft.abs() * z2_fft.abs()

        # Phase difference (using complex division for phase extraction)
        # e^{i(φ₁-φ₂)} = (F̂₁/|F̂₁|) · (F̂₂/|F̂₂|)* = F̂₁ · F̂₂* / (|F̂₁| · |F̂₂|)
        magnitude_product = z1_fft.abs() * z2_fft.abs()
        phase_factor = torch.where(
            magnitude_product > self.epsilon,
            (z1_fft * z2_fft.conj()) / magnitude_product,
            torch.zeros_like(z1_fft),
        )

        # Weighted phase coherence
        numerator = (weights * phase_factor).sum(dim=spatial_dims)
        denominator = weights.sum(dim=spatial_dims)

        weighted_plv = numerator.abs() / (denominator + self.epsilon)

        return weighted_plv.mean()

    def complex_psnr(self, z1, z2, spatial_dims=(-2, -1)):
        """
        Peak Signal-to-Noise Ratio for complex-valued data

        Mathematical derivation:
        Step 1: Peak value: PEAK = max(|z₁(x,y)|) over spatial domain
        Step 2: Mean squared error: MSE = ⟨|z₁(x,y) - z₂(x,y)|²⟩
        Step 3: PSNR = 10 · log₁₀(PEAK² / MSE) = 20 · log₁₀(PEAK / √MSE)

        For complex data, we use magnitude for peak calculation.
        """
        # Calculate peak value from ground truth (z1)
        z1_magnitude = z1.abs()
        peak_value = z1_magnitude.flatten(start_dim=1).max(dim=1)[0]  # Max per batch

        # Calculate MSE in complex domain
        mse = ((z1 - z2).abs() ** 2).flatten(start_dim=1).mean(dim=1)  # Mean per batch

        # PSNR calculation
        psnr = 20 * torch.log10(peak_value / (torch.sqrt(mse) + self.epsilon))

        return psnr.mean()

    def kx_ky_analysis(self, z1, z2, spatial_dims=(-2, -1)):
        """
        Compute kx and ky directional spectral analysis with rigorous derivation.

        Mathematical formulation:
        Step 1: Power spectral density in 2D k-space
                PSD(kₓ,kᵧ) = |F̂(kₓ,kᵧ)|²
        Step 2: Marginal spectra (projection onto axes)
                PSD_kₓ(kₓ) = Σ_{kᵧ} PSD(kₓ,kᵧ)  [integrate over kᵧ]
                PSD_kᵧ(kᵧ) = Σ_{kₓ} PSD(kₓ,kᵧ)  [integrate over kₓ]
        Step 3: Directional correlations
                ρₓ = cosine_similarity(PSD_kₓ¹, PSD_kₓ²)
                ρᵧ = cosine_similarity(PSD_kᵧ¹, PSD_kᵧ²)
        Step 4: Radial spectrum for isotropic analysis
                k = √(kₓ² + kᵧ²)
                PSD_radial(k) = ⟨PSD(kₓ,kᵧ)⟩_{kₓ²+kᵧ²=k²}

        Args:
            z1, z2: Complex tensors (supports 5D: [BS, v_par, mu, s, x, y])
            spatial_dims: Spatial dimensions for FFT

        Returns:
            Dictionary with directional spectral metrics
        """
        # Compute power spectral densities directly
        z1_fft = self._to_spectral_domain(z1, spatial_dims)
        z2_fft = self._to_spectral_domain(z2, spatial_dims)
        psd1 = z1_fft.abs() ** 2
        psd2 = z2_fft.abs() ** 2

        # For 5D data [BS, v_par, mu, s, x, y], spatial_dims are typically (-2, -1)
        # Sum over dimensions to get directional spectra
        # kx spectrum: sum over ky (last spatial dimension)
        kx_spectrum1 = psd1.sum(dim=spatial_dims[-1], keepdim=False)
        kx_spectrum2 = psd2.sum(dim=spatial_dims[-1], keepdim=False)

        # ky spectrum: sum over kx (second-to-last spatial dimension)
        ky_spectrum1 = psd1.sum(dim=spatial_dims[-2], keepdim=False)
        ky_spectrum2 = psd2.sum(dim=spatial_dims[-2], keepdim=False)

        # kx correlation: flatten over [BS, v_par, mu, s] dimensions
        kx_flat1 = (
            kx_spectrum1.flatten(start_dim=0, end_dim=-2)
            if len(kx_spectrum1.shape) > 1
            else kx_spectrum1
        )
        kx_flat2 = (
            kx_spectrum2.flatten(start_dim=0, end_dim=-2)
            if len(kx_spectrum2.shape) > 1
            else kx_spectrum2
        )

        kx_correlation = F.cosine_similarity(
            kx_flat1.unsqueeze(0) if kx_flat1.dim() == 1 else kx_flat1.unsqueeze(1),
            kx_flat2.unsqueeze(0) if kx_flat2.dim() == 1 else kx_flat2.unsqueeze(1),
            dim=-1,
        ).mean()

        # ky correlation: flatten over [BS, v_par, mu, s] dimensions
        ky_flat1 = (
            ky_spectrum1.flatten(start_dim=0, end_dim=-2)
            if len(ky_spectrum1.shape) > 1
            else ky_spectrum1
        )
        ky_flat2 = (
            ky_spectrum2.flatten(start_dim=0, end_dim=-2)
            if len(ky_spectrum2.shape) > 1
            else ky_spectrum2
        )

        ky_correlation = F.cosine_similarity(
            ky_flat1.unsqueeze(0) if ky_flat1.dim() == 1 else ky_flat1.unsqueeze(1),
            ky_flat2.unsqueeze(0) if ky_flat2.dim() == 1 else ky_flat2.unsqueeze(1),
            dim=-1,
        ).mean()

        # Radial binning for k-space analysis
        # Create k-space grid for the spatial dimensions
        ny, nx = psd1.shape[spatial_dims[-2]], psd1.shape[spatial_dims[-1]]
        kx_vals = torch.fft.fftfreq(nx, device=psd1.device).view(1, -1)
        ky_vals = torch.fft.fftfreq(ny, device=psd1.device).view(-1, 1)
        k_magnitude = torch.sqrt(kx_vals**2 + ky_vals**2)

        # Simple radial correlation (averaging over non-spatial dimensions)
        # Average PSD over all non-spatial dimensions: [v_par, mu, s]
        non_spatial_dims = tuple(range(1, len(psd1.shape) - 2))
        if non_spatial_dims:
            psd1_spatial_avg = psd1.mean(dim=non_spatial_dims)  # [BS, x, y]
            psd2_spatial_avg = psd2.mean(dim=non_spatial_dims)  # [BS, x, y]
        else:
            psd1_spatial_avg = psd1
            psd2_spatial_avg = psd2

        # Average over batch dimension and flatten for correlation
        psd1_radial = psd1_spatial_avg.mean(dim=0).flatten()  # [x*y]
        psd2_radial = psd2_spatial_avg.mean(dim=0).flatten()  # [x*y]

        radial_correlation = F.cosine_similarity(
            psd1_radial.unsqueeze(0), psd2_radial.unsqueeze(0), dim=1
        ).item()

        # Total spectral power ratio: power2 / power1 (predictions / ground truth)
        # Sum over spatial dimensions, then average over all other dimensions
        total_power1 = psd1.sum(dim=spatial_dims).mean()
        total_power2 = psd2.sum(dim=spatial_dims).mean()
        total_power_ratio = total_power2 / (total_power1 + self.epsilon)

        return {
            "radial_spectral_correlation": radial_correlation,
            "kx_correlation": (
                kx_correlation.item()
                if hasattr(kx_correlation, "item")
                else kx_correlation
            ),
            "ky_correlation": (
                ky_correlation.item()
                if hasattr(ky_correlation, "item")
                else ky_correlation
            ),
            "total_spectral_power_ratio": total_power_ratio.item(),
        }

    def evaluate_all(
        self,
        preds,
        gts,
        dims=None,
        return_dict=True,
        include_spectral=True,
        spatial_dims=(-2, -1),
    ):
        """
        Evaluate all metrics for complex predictions vs ground truth
            preds: Predictions tensor [BS, 2, v_par, mu, s, x, y] (5D gyrokinetic data)
            gts: Ground truth tensor [BS, 2, v_par, mu, s, x, y] (5D gyrokinetic data)
            dims: Dimensions to aggregate over. If None, aggregates all except batch
            return_dict: If True, returns dict. If False, returns tensor
            include_spectral: If True, include spectral analysis metrics
            spatial_dims: Spatial dimensions for FFT (default: last 2 dimensions)
        """
        # Convert to complex (handles both 2-channel and 4-channel zonal flow data)
        z_preds = self.to_complex(preds)
        z_gts = self.to_complex(gts)

        # Compute standard metrics
        pearson_mag, pearson_phase = self.complex_pearson(z_preds, z_gts, dims)
        plv = self.phase_locking_value(z_preds, z_gts, dims)
        cssim = self.complex_ssim(z_preds, z_gts, dims)
        mse = self.complex_mse(z_preds, z_gts, dims)
        l1 = self.complex_l1(z_preds, z_gts, dims)
        spectral_energy = self.spectral_energy_metric(z_preds, z_gts, dims)
        psnr = self.complex_psnr(z_gts, z_preds, spatial_dims)

        results = {
            "pearson_magnitude": pearson_mag.mean().item(),
            "pearson_phase": pearson_phase.mean().item(),
            "phase_locking": plv.mean().item(),
            "ssim": cssim.mean().item(),
            "mse": mse.mean().item(),
            "l1": l1.mean().item(),
            "spectral_energy": spectral_energy.item(),
            "psnr": psnr.item(),
        }

        # Add spectral analysis if requested
        if include_spectral:
            try:
                # Transform to spectral domain and use existing complex functions
                z_preds_fft = self._to_spectral_domain(z_preds, spatial_dims)
                z_gts_fft = self._to_spectral_domain(z_gts, spatial_dims)

                # Use existing complex_pearson in frequency domain (equivalent to spectral correlation)
                avg_dims = list(range(1, z_preds_fft.dim()))
                spectral_pearson_mag, _ = self.complex_pearson(
                    z_preds_fft, z_gts_fft, dims=avg_dims
                )

                # Use existing phase_locking_value in frequency domain
                spectral_plv = self.phase_locking_value(
                    z_preds_fft, z_gts_fft, dims=avg_dims
                )

                # Compute kx/ky directional analysis (unique spectral info)
                kx_ky_results = self.kx_ky_analysis(z_preds, z_gts, spatial_dims)

                # Magnitude-weighted phase coherence (advanced spectral metric)
                weighted_plv = self.magnitude_weighted_phase_coherence(
                    z_preds, z_gts, spatial_dims
                )

                # Add to results
                results.update(
                    {
                        "spectral_pearson_magnitude": spectral_pearson_mag.mean().item(),
                        "spectral_phase_locking": spectral_plv.mean().item(),
                        "magnitude_weighted_phase_coherence": weighted_plv.item(),
                        "kx_correlation": kx_ky_results["kx_correlation"],
                        "ky_correlation": kx_ky_results["ky_correlation"],
                        "radial_spectral_correlation": kx_ky_results[
                            "radial_spectral_correlation"
                        ],
                        "total_spectral_power_ratio": kx_ky_results[
                            "total_spectral_power_ratio"
                        ],
                    }
                )

            except Exception as e:
                # If spectral analysis fails, add warning but continue
                print(f"Warning: Spectral analysis failed: {e}")
                results["spectral_analysis_error"] = str(e)

        if return_dict:
            return results
        else:
            # Stack main metrics into a tensor (excluding spectral for tensor output)
            return torch.stack(
                [
                    pearson_mag.mean(),
                    pearson_phase.mean(),
                    plv.mean(),
                    cssim.mean(),
                    mse.mean(),
                    l1.mean(),
                    spectral_energy,
                ]
            )


def validation_metrics(
    preds: Dict[str, torch.Tensor],
    tgts: Dict[str, torch.Tensor],
    geometry: Dict[str, torch.Tensor],
    loss_wrap: nn.Module,
    eval_integrals: bool = True,
):
    # detect sequence length
    is_sequence = False
    if preds["df"].ndim == 1 + 1 + 5 + 1:
        is_sequence = True

    n_steps = preds["df"].shape[0] if is_sequence else 1

    metrics_all = defaultdict(list)
    integrated_all = []

    complex_metrics_calc = None
    if "df" in preds:
        complex_metrics_calc = ComplexMetrics()

    for n in range(n_steps):
        if is_sequence:
            nth_pred = {k: v[n] for k, v in preds.items()}
            nth_tgt = {k: v[n] for k, v in tgts.items()}
        else:
            nth_pred = preds
            nth_tgt = tgts

        common_keys = set(nth_pred.keys()) & set(nth_tgt.keys())
        for k in common_keys:
            assert nth_pred[k].shape == nth_tgt[k].shape, f"Shape mismatch for {k}[{n}]"

        # variable returns (3 for GyroSwin, 4 for PINC)
        ret = loss_wrap(
            preds=nth_pred,
            tgts=nth_tgt,
            geometry=geometry,
            compute_integrals=eval_integrals,
        )

        nth_losses, nth_integrated = ret[1], ret[2]

        integrated_all.append(nth_integrated)

        for k, v in nth_losses.items():
            # detach and cpu for storage
            val = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v)
            metrics_all[k].append(val)

        # complex metrics (df only)
        if "df" in nth_pred and "df" in nth_tgt and complex_metrics_calc:
            c_res = complex_metrics_calc.evaluate_all(nth_pred["df"], nth_tgt["df"])
            for ck, cv in c_res.items():
                metrics_all[f"complex_{ck}"].append(
                    torch.tensor(cv, dtype=torch.float32)
                )

    metrics_all = {k: torch.stack(v) for k, v in metrics_all.items()}
    if not is_sequence:
        metrics_all = {k: v.squeeze(0) for k, v in metrics_all.items()}
        integrated_all = integrated_all[0]
    return metrics_all, integrated_all
