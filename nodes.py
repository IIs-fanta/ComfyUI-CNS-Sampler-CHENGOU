"""
ComfyUI Custom Node: Colored Noise Sampler (CNS)
Based on "Colored Noise Diffusion Sampling" (Davidson et al., 2026)
https://arxiv.org/abs/2605.30332

CNS replaces uniform white noise injection in SDE sampling with a
dynamic, frequency-decoupled schedule that routes energy toward
structurally unresolved frequency bands — exploiting the spectral
bias of diffusion models to improve generation quality.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import comfy.samplers
import comfy.sample
import latent_preview
from tqdm.auto import trange

# Path to the bundled gamma matrix sitting next to this file
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_GAMMA_PATH = os.path.join(_NODE_DIR, "gamma_matrix_scaled.pt")

def _load_bundled_gamma():
    if os.path.exists(_BUNDLED_GAMMA_PATH):
        try:
            gm = torch.load(_BUNDLED_GAMMA_PATH, map_location="cpu", weights_only=True)
            print(f"[CNS] Loaded bundled gamma matrix: {_BUNDLED_GAMMA_PATH}, shape={gm.shape}")
            return gm
        except Exception as e:
            print(f"[CNS] Warning: could not load bundled gamma matrix ({e}). Using built-in approximation.")
    else:
        print(f"[CNS] gamma_matrix_scaled.pt not found in node directory. Using built-in approximation.")
    return None

# Load once at import time
_BUNDLED_GAMMA = _load_bundled_gamma()


# ─────────────────────────────────────────────────────────────────────────────
# Gamma Matrix: encodes per-frequency-band progress γ(f, t) ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def compute_radial_freq_bins(height, width, num_bins=32):
    """
    Compute radial frequency bin indices for a latent of given spatial size.
    Returns a (H, W) integer tensor mapping each (h, w) position to a freq bin.
    """
    fy = torch.fft.fftfreq(height)
    fx = torch.fft.fftfreq(width)
    fy2d, fx2d = torch.meshgrid(fy, fx, indexing='ij')
    r = torch.sqrt(fy2d ** 2 + fx2d ** 2)           # radial frequency, [0, ~0.7]
    r_max = r.max().item() + 1e-8
    bins = (r / r_max * (num_bins - 1)).long()        # map to [0, num_bins-1]
    return bins                                        # (H, W)


def build_gamma_matrix_from_sigmas(sigmas, height, width, num_bins=32):
    """
    Build an approximate γ(f, t) matrix from sigma schedule.

    The progress of a frequency band f at step t is approximated as:
        γ(f, t) = 1 - σ(t) / σ(0)   (same for all bands in this approximation)

    For a true gamma matrix you'd run an ODE analysis on the actual model
    (see the official repo). This approximation gives reasonable results
    and requires no precomputation.
    """
    T = len(sigmas) - 1          # number of steps (sigmas has T+1 elements)
    gamma = torch.zeros(num_bins, T)
    sigma_max = sigmas[0].item()
    for t in range(T):
        progress = 1.0 - (sigmas[t].item() / (sigma_max + 1e-8))
        progress = max(0.0, min(1.0, progress))
        gamma[:, t] = progress
    return gamma                  # (num_bins, T)


def load_gamma_matrix(path):
    """Load a precomputed gamma matrix saved as a .pt file."""
    return torch.load(path, map_location='cpu', weights_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# CNS Noise Schedule: β_f(t) per the paper's formula
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta_schedule(gamma_t, power_gamma=1.0, gamma_divider=1.0,
                           alpha_tilt=0.0, use_fnorm=False, num_bins=32):
    """
    Compute the per-frequency noise scaling β_f(t) for a single timestep.

    Formula (from paper):
        β_f(t) = sqrt(1 - γ_f(t)) / sqrt( mean_f'[(1 - γ_f'(t))] )

    Args:
        gamma_t:       (num_bins,) tensor, γ values at current timestep
        power_gamma:   exponent applied to residual (1-γ) before normalising
        gamma_divider: divides γ before computing residuals (weakens effect)
        alpha_tilt:    frequency tilt — positive boosts high-freq, negative boosts low-freq
        use_fnorm:     weight tilt by normalised frequency position
        num_bins:      number of radial frequency bins
    """
    gamma_t = (gamma_t / gamma_divider).clamp(0.0, 1.0)
    residual = (1.0 - gamma_t).clamp(min=1e-8) ** power_gamma   # (num_bins,)

    # Frequency tilt: shift energy toward high or low frequencies
    if alpha_tilt != 0.0:
        freqs = torch.linspace(0.0, 1.0, num_bins, device=gamma_t.device)
        if use_fnorm:
            tilt = torch.exp(alpha_tilt * freqs)
        else:
            tilt = torch.ones(num_bins, device=gamma_t.device) * (1.0 + alpha_tilt)
        residual = residual * tilt

    # Normalise so total injected energy is preserved (mean β² = 1)
    mean_residual = residual.mean().clamp(min=1e-8)
    beta = torch.sqrt(residual / mean_residual)                  # (num_bins,)
    return beta


# ─────────────────────────────────────────────────────────────────────────────
# Apply CNS scaling to a noise tensor in frequency space
# ─────────────────────────────────────────────────────────────────────────────

def apply_cns_to_noise(noise, beta, freq_bins, energy_scale=1.0):
    """
    Modulate noise in 2D FFT space according to the CNS β schedule.

    Args:
        noise:      (..., H, W) noise tensor
        beta:       (num_bins,) per-frequency-bin scaling factors
        freq_bins:  (H, W) integer tensor mapping each pixel to a freq bin
        energy_scale: global energy multiplier (fine-tuning knob)

    Returns:
        (..., H, W) colored noise tensor with same total variance as input
    """
    H, W = noise.shape[-2:]
    device = noise.device
    beta = beta.to(device)
    freq_bins = freq_bins.to(device)

    # Build a (H, W) scaling map from the per-bin β values
    scale_map = beta[freq_bins]                      # (H, W)
    scale_map = scale_map * energy_scale

    # FFT → scale per frequency → iFFT
    noise_f = torch.fft.fft2(noise)                  # (..., H, W) complex
    scale_map = scale_map.reshape((1,) * (noise.ndim - 2) + (H, W))
    noise_f_colored = noise_f * scale_map

    colored = torch.fft.ifft2(noise_f_colored).real  # back to real space

    # Re-normalise to match the original noise std (variance-preserving)
    orig_std   = noise.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    colored_std = colored.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    colored = colored * (orig_std / colored_std)

    return colored


# ─────────────────────────────────────────────────────────────────────────────
# Custom sampler that wraps Euler SDE with CNS noise injection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_euler_cns(model, x, sigmas, extra_args=None, callback=None,
                     disable=None, s_churn=0.5, gamma_matrix=None,
                     power_gamma=1.0, gamma_divider=1.0,
                     alpha_tilt_start=0.0, alpha_tilt_end=None,
                     alpha_use_fnorm=False, alpha_exp_interp=False,
                     alpha_exp_sharpness=0.75, energy_scale=1.0,
                     num_bins=32):
    """
    Euler SDE sampler with Colored Noise Sampling (CNS) injection.
    """
    extra_args = extra_args or {}
    if x.ndim < 4:
        raise ValueError(f"CNS sampler expected a tensor with spatial dimensions, got shape {tuple(x.shape)}")
    batch_size = x.shape[0]
    H, W = x.shape[-2:]
    T = len(sigmas) - 1

    # Pre-compute freq bins for this latent size
    freq_bins = compute_radial_freq_bins(H, W, num_bins=num_bins)

    # Build or use provided gamma matrix
    if gamma_matrix is None:
        gamma_matrix = build_gamma_matrix_from_sigmas(sigmas, H, W, num_bins=num_bins)
    gamma_matrix = gamma_matrix.float()

    # Resize gamma matrix to (num_bins, T) using 1D linear interp on each axis.
    # Always use (N, 1, L) → 1D linear → squeeze to avoid dimension mismatch errors.
    if gamma_matrix.shape[1] != T:
        # Resize time axis: treat bins as batch dim
        gamma_matrix = F.interpolate(
            gamma_matrix.unsqueeze(1),   # (bins, 1, T_orig)
            size=T,
            mode='linear',
            align_corners=False
        ).squeeze(1)                     # (bins, T)
    if gamma_matrix.shape[0] != num_bins:
        # Resize freq-bin axis: transpose so bins become the last dim, interp, transpose back
        gamma_matrix = F.interpolate(
            gamma_matrix.t().unsqueeze(1),   # (T, 1, orig_bins)
            size=num_bins,
            mode='linear',
            align_corners=False
        ).squeeze(1).t()                     # (num_bins, T)
    gamma_matrix = gamma_matrix[:num_bins, :T]

    alpha_tilt_end = alpha_tilt_end if alpha_tilt_end is not None else alpha_tilt_start

    for i in trange(T, disable=disable):
        # Current and next sigma
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Model denoising prediction
        denoised = model(x, sigma * torch.ones(batch_size, device=x.device), **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma,
                      'denoised': denoised})

        # ODE step: Euler along the probability flow direction
        # dx = (x - denoised) / sigma * (sigma_next - sigma)
        d = (x - denoised) / sigma
        dt = sigma_next - sigma
        x = x + d * dt

        # SDE stochastic term — inject noise then re-add to keep variance on track
        # sigma_up: how much noise to inject so total variance matches sigma_next
        # Formula: sigma_up = sqrt(sigma_next^2 - sigma_next^2 * sigma_next^2 / sigma^2)
        #        = sigma_next * sqrt(1 - (sigma_next/sigma)^2)   [Karras et al.]
        if i < T - 1 and sigma_next > 0 and s_churn > 0:
            ratio = (sigma_next / sigma).clamp(max=1.0)
            sigma_up = sigma_next * (1 - ratio ** 2).clamp(min=0).sqrt()
            sigma_up = sigma_up * s_churn

            # Generate base white noise
            noise = torch.randn_like(x)

            # ── CNS: compute per-frequency β for this timestep ──────────────
            t_norm = i / max(T - 1, 1)                   # normalise t to [0,1]
            if alpha_exp_interp:
                w = (torch.tensor(t_norm) * alpha_exp_sharpness).exp()
                w = (w - 1) / (torch.tensor(alpha_exp_sharpness).exp() - 1 + 1e-8)
                w = w.item()
            else:
                w = t_norm

            alpha_t = alpha_tilt_start + w * (alpha_tilt_end - alpha_tilt_start)

            gamma_t = gamma_matrix[:, i]                  # (num_bins,)
            beta = compute_beta_schedule(
                gamma_t,
                power_gamma=power_gamma,
                gamma_divider=gamma_divider,
                alpha_tilt=alpha_t,
                use_fnorm=alpha_use_fnorm,
                num_bins=num_bins,
            )

            # Apply CNS frequency modulation
            colored_noise = apply_cns_to_noise(noise, beta, freq_bins,
                                               energy_scale=energy_scale)
            # ────────────────────────────────────────────────────────────────

            x = x + colored_noise * sigma_up

    return x


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node: CNSSamplerNode
# ─────────────────────────────────────────────────────────────────────────────

class CNSSamplerNode:
    """
    Colored Noise Sampler (CNS) for ComfyUI.

    Drop-in replacement for standard SDE samplers. Connects to
    SamplerCustomAdvanced (or KSampler via the SAMPLER output).

    Based on: "Colored Noise Diffusion Sampling" (Davidson et al., 2026)
    https://arxiv.org/abs/2605.30332
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_churn": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "SDE noise strength. 0 = ODE (no stochasticity). 0.5 is a good default."
                }),
                "power_gamma": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Exponent on residual (1-γ). Lower = gentler coloring. Paper uses 0.75 for unguided."
                }),
                "gamma_divider": ("FLOAT", {
                    "default": 1.73, "min": 0.1, "max": 50.0, "step": 0.01,
                    "tooltip": "Divides γ values, weakening the coloring effect. Paper: 1.73 unguided, 25.0 guided."
                }),
                "energy_scale": ("FLOAT", {
                    "default": 0.98, "min": 0.5, "max": 1.5, "step": 0.005,
                    "tooltip": "Global energy multiplier after normalisation. Paper uses 0.98 (unguided) / 0.998 (guided)."
                }),
                "alpha_tilt_start": ("FLOAT", {
                    "default": 0.15, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Frequency tilt at the start of sampling. Positive = boost high-freq. Paper: 0.15 unguided."
                }),
                "alpha_tilt_end": ("FLOAT", {
                    "default": -0.5, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Frequency tilt at the end of sampling. Paper: -0.5 unguided, 0.03 guided."
                }),
                "alpha_use_fnorm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Weight tilt by normalised frequency position. Recommended when using two-value tilting."
                }),
                "alpha_exp_interp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use exponential (vs linear) interpolation between alpha_start and alpha_end."
                }),
                "alpha_exp_sharpness": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Sharpness of exponential alpha interpolation. Paper uses 0.75."
                }),
                "num_freq_bins": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Number of radial frequency bands. 32 is a good default."
                }),
            },

        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, s_churn, power_gamma, gamma_divider, energy_scale,
                    alpha_tilt_start, alpha_tilt_end, alpha_use_fnorm,
                    alpha_exp_interp, alpha_exp_sharpness, num_freq_bins):

        # Use the bundled gamma matrix (loaded at import time)
        gamma_matrix = _BUNDLED_GAMMA

        sampler_fn = lambda model, x, sigmas, extra_args, callback, disable: \
            sample_euler_cns(
                model, x, sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                s_churn=s_churn,
                gamma_matrix=gamma_matrix,
                power_gamma=power_gamma,
                gamma_divider=gamma_divider,
                alpha_tilt_start=alpha_tilt_start,
                alpha_tilt_end=alpha_tilt_end,
                alpha_use_fnorm=alpha_use_fnorm,
                alpha_exp_interp=alpha_exp_interp,
                alpha_exp_sharpness=alpha_exp_sharpness,
                energy_scale=energy_scale,
                num_bins=num_freq_bins,
            )

        sampler = comfy.samplers.KSAMPLER(sampler_fn)
        return (sampler,)


# ─────────────────────────────────────────────────────────────────────────────
# Node Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CNSSampler_CHENGOU": CNSSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CNSSampler_CHENGOU": "CNS Sampler (Colored Noise) | CHENGOU",
}
