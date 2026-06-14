"""
ComfyUI custom node for Colored Noise Sampling (CNS).

The frequency shaping follows the public implementation from
HadarDavidson/colored-noise-sampling, adapted to ComfyUI's sampler API.
"""

import math
import os

import torch
import torch.nn.functional as F
import comfy.samplers
from tqdm.auto import trange


_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_GAMMA_PATH = os.path.join(_NODE_DIR, "gamma_matrix_scaled.pt")

_BUNDLED_GAMMA = None
_BUNDLED_GAMMA_LOADED = False


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_bundled_gamma():
    if not os.path.exists(_BUNDLED_GAMMA_PATH):
        print("[CNS] gamma_matrix_scaled.pt not found. Using sigma-schedule approximation.")
        return None

    try:
        gamma = _torch_load_cpu(_BUNDLED_GAMMA_PATH)
    except Exception as exc:
        print(f"[CNS] Could not load bundled gamma matrix: {exc}. Using approximation.")
        return None

    if not torch.is_tensor(gamma) or gamma.ndim != 2:
        print("[CNS] Bundled gamma matrix has an unexpected format. Using approximation.")
        return None

    print(f"[CNS] Loaded gamma matrix: {_BUNDLED_GAMMA_PATH}, shape={tuple(gamma.shape)}")
    return gamma


def _get_bundled_gamma(use_bundled_gamma_matrix):
    global _BUNDLED_GAMMA, _BUNDLED_GAMMA_LOADED

    if not use_bundled_gamma_matrix:
        return None

    if not _BUNDLED_GAMMA_LOADED:
        _BUNDLED_GAMMA = _load_bundled_gamma()
        _BUNDLED_GAMMA_LOADED = True

    return _BUNDLED_GAMMA


def compute_radial_freq_bins(height, width, num_bins=32):
    fy = torch.fft.fftfreq(height)
    fx = torch.fft.fftfreq(width)
    fy2d, fx2d = torch.meshgrid(fy, fx, indexing="ij")

    radius = torch.sqrt(fx2d.square() + fy2d.square())
    radius = radius / radius.max().clamp(min=1e-8)
    return (radius * (num_bins - 1)).long().clamp(0, num_bins - 1)


def build_gamma_matrix_from_sigmas(sigmas, num_bins=32):
    """Fallback gamma matrix in official layout: rows are steps, columns are bins."""
    steps = len(sigmas) - 1
    gamma = torch.zeros(steps, num_bins, dtype=torch.float32)

    sigma_max = sigmas[0].detach().float().cpu().item()
    sigma_max = max(sigma_max, 1e-8)

    for step in range(steps):
        sigma = sigmas[step].detach().float().cpu().item()
        progress = 1.0 - sigma / sigma_max
        gamma[step, :] = max(0.0, min(1.0, progress))

    return gamma


def load_gamma_matrix(path):
    return _torch_load_cpu(path)


def _gamma_as_steps_by_bins(gamma_matrix, num_bins):
    """Accept official [steps, bins] matrices and older [bins, steps] matrices."""
    if not torch.is_tensor(gamma_matrix) or gamma_matrix.ndim != 2:
        raise ValueError("gamma matrix must be a 2D torch.Tensor")

    rows, cols = gamma_matrix.shape
    if cols == num_bins:
        gamma = gamma_matrix
    elif rows == num_bins:
        gamma = gamma_matrix.t()
    elif rows > cols:
        gamma = gamma_matrix
    else:
        gamma = gamma_matrix.t()

    return gamma.float()


def _resize_gamma_matrix(gamma_matrix, steps, num_bins):
    """Resize a [steps, bins] matrix without swapping its axes."""
    gamma = gamma_matrix

    if gamma.shape[0] != steps:
        gamma = F.interpolate(
            gamma.t().unsqueeze(0),
            size=steps,
            mode="linear",
            align_corners=False,
        ).squeeze(0).t()

    if gamma.shape[1] != num_bins:
        gamma = F.interpolate(
            gamma.unsqueeze(1),
            size=num_bins,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    return gamma[:steps, :num_bins]


def prepare_gamma_matrix(gamma_matrix, sigmas, num_bins):
    steps = len(sigmas) - 1

    if gamma_matrix is None:
        gamma_matrix = build_gamma_matrix_from_sigmas(sigmas, num_bins=num_bins)

    try:
        gamma = _gamma_as_steps_by_bins(gamma_matrix, num_bins)
    except ValueError as exc:
        print(f"[CNS] {exc}. Using sigma-schedule approximation.")
        gamma = build_gamma_matrix_from_sigmas(sigmas, num_bins=num_bins)

    return _resize_gamma_matrix(gamma, steps, num_bins)


def _interpolate_alpha(step, steps, start, end, use_exp, sharpness):
    progress = step / max(steps - 1, 1)

    if use_exp:
        denom = math.exp(sharpness) - 1.0
        if abs(denom) > 1e-8:
            progress = (math.exp(sharpness * progress) - 1.0) / denom

    return start + progress * (end - start)


def compute_noise_scaling(
    gamma_step,
    power_gamma=1.0,
    gamma_divider=1.0,
    alpha_tilt=0.0,
    use_fnorm=False,
):
    """
    Per-frequency CNS scaling for one step.

    This mirrors the official cns_sde residual-energy path. The final global
    variance conservation happens after FFT filtering in apply_cns_to_noise().
    """
    gamma_divider = max(float(gamma_divider), 1e-8)
    base_residual = 1.0 - gamma_step / gamma_divider

    if alpha_tilt != 0.0:
        if use_fnorm:
            f_norm = torch.linspace(
                0.0,
                1.0,
                steps=gamma_step.numel(),
                device=gamma_step.device,
                dtype=gamma_step.dtype,
            )
            residual_energy = torch.exp(alpha_tilt * f_norm) * base_residual
        else:
            residual_energy = torch.exp(alpha_tilt * base_residual)
    else:
        residual_energy = base_residual

    residual_energy = residual_energy.clamp(min=0.0)
    if power_gamma != 1.0:
        residual_energy = residual_energy.pow(power_gamma)

    return residual_energy


def compute_beta_schedule(
    gamma_t,
    power_gamma=1.0,
    gamma_divider=1.0,
    alpha_tilt=0.0,
    use_fnorm=False,
    num_bins=None,
):
    """Backward-compatible name kept for old imports/workflows."""
    return compute_noise_scaling(
        gamma_t,
        power_gamma=power_gamma,
        gamma_divider=gamma_divider,
        alpha_tilt=alpha_tilt,
        use_fnorm=use_fnorm,
    )


def apply_cns_to_noise(noise, noise_scaling, freq_bins, energy_scale=1.0):
    height, width = noise.shape[-2:]
    dtype = noise.dtype
    device = noise.device

    scale_grid = noise_scaling.to(device=device, dtype=torch.float32)[freq_bins.to(device)]
    scale_grid = scale_grid.reshape((1,) * (noise.ndim - 2) + (height, width))

    noise_float = noise.to(torch.float32)
    filtered = torch.fft.ifft2(torch.fft.fft2(noise_float) * scale_grid).real

    filtered_std = filtered.std()
    if filtered_std > 1e-9:
        filtered = filtered / filtered_std

    filtered = filtered * float(energy_scale)
    return filtered.to(dtype=dtype)


@torch.no_grad()
def sample_euler_cns(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.5,
    gamma_matrix=None,
    power_gamma=1.0,
    gamma_divider=1.0,
    alpha_tilt_start=0.0,
    alpha_tilt_end=None,
    alpha_use_fnorm=False,
    alpha_exp_interp=False,
    alpha_exp_sharpness=0.75,
    energy_scale=1.0,
    num_bins=32,
):
    extra_args = extra_args or {}
    if x.ndim < 4:
        raise ValueError(f"CNS sampler expected a spatial latent tensor, got shape {tuple(x.shape)}")

    batch_size = x.shape[0]
    height, width = x.shape[-2:]
    steps = len(sigmas) - 1

    freq_bins = compute_radial_freq_bins(height, width, num_bins=num_bins)
    gamma_matrix = prepare_gamma_matrix(gamma_matrix, sigmas, num_bins=num_bins)
    alpha_tilt_end = alpha_tilt_start if alpha_tilt_end is None else alpha_tilt_end

    for step in trange(steps, disable=disable):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]

        sigma_in = sigma * torch.ones(batch_size, device=x.device)
        denoised = model(x, sigma_in, **extra_args)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )

        d = (x - denoised) / sigma
        x = x + d * (sigma_next - sigma)

        if step >= steps - 1 or sigma_next <= 0 or s_churn <= 0:
            continue

        ratio = (sigma_next / sigma).clamp(max=1.0)
        sigma_up = sigma_next * (1.0 - ratio.square()).clamp(min=0.0).sqrt()
        sigma_up = sigma_up * s_churn

        alpha_t = _interpolate_alpha(
            step,
            steps,
            alpha_tilt_start,
            alpha_tilt_end,
            alpha_exp_interp,
            alpha_exp_sharpness,
        )

        gamma_step = gamma_matrix[step].to(device=x.device, dtype=torch.float32)
        noise_scaling = compute_noise_scaling(
            gamma_step,
            power_gamma=power_gamma,
            gamma_divider=gamma_divider,
            alpha_tilt=alpha_t,
            use_fnorm=alpha_use_fnorm,
        )

        noise = torch.randn_like(x)
        colored_noise = apply_cns_to_noise(
            noise,
            noise_scaling,
            freq_bins,
            energy_scale=energy_scale,
        )
        x = x + colored_noise * sigma_up

    return x


class CNSSamplerNode:
    """
    Colored Noise Sampler for ComfyUI.

    The sampler is meant for SamplerCustomAdvanced and returns a standard
    ComfyUI SAMPLER object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s_churn": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "SDE noise strength. 0 disables the stochastic CNS term.",
                    },
                ),
                "power_gamma": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Power applied to the residual energy. Official unguided setting: 0.75.",
                    },
                ),
                "gamma_divider": (
                    "FLOAT",
                    {
                        "default": 1.73,
                        "min": 0.1,
                        "max": 50.0,
                        "step": 0.01,
                        "tooltip": "Divides gamma before residual energy is computed. Official unguided setting: 1.73.",
                    },
                ),
                "energy_scale": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.5,
                        "max": 1.5,
                        "step": 0.005,
                        "tooltip": "Applied after FFT filtering and unit-std normalization. Official unguided setting: 0.98.",
                    },
                ),
                "alpha_tilt_start": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency tilt at the first step. Positive values favor higher frequencies.",
                    },
                ),
                "alpha_tilt_end": (
                    "FLOAT",
                    {
                        "default": -0.5,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency tilt at the final step.",
                    },
                ),
                "alpha_use_fnorm": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use normalized radial frequency for alpha tilting, matching the official published settings.",
                    },
                ),
                "alpha_exp_interp": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use exponential interpolation between alpha_tilt_start and alpha_tilt_end.",
                    },
                ),
                "alpha_exp_sharpness": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Sharpness for exponential alpha interpolation. Official unguided setting: 0.75.",
                    },
                ),
                "num_freq_bins": (
                    "INT",
                    {
                        "default": 32,
                        "min": 8,
                        "max": 128,
                        "step": 8,
                        "tooltip": "Number of radial frequency bins. Official matrices use 32.",
                    },
                ),
                "use_bundled_gamma_matrix": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Load gamma_matrix_scaled.pt from this node folder. Disable to use the sigma-schedule fallback.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(
        self,
        s_churn,
        power_gamma,
        gamma_divider,
        energy_scale,
        alpha_tilt_start,
        alpha_tilt_end,
        alpha_use_fnorm,
        alpha_exp_interp,
        alpha_exp_sharpness,
        num_freq_bins,
        use_bundled_gamma_matrix,
    ):
        gamma_matrix = _get_bundled_gamma(use_bundled_gamma_matrix)

        def sampler_fn(model, x, sigmas, extra_args, callback, disable):
            return sample_euler_cns(
                model,
                x,
                sigmas,
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

        return (comfy.samplers.KSAMPLER(sampler_fn),)


NODE_CLASS_MAPPINGS = {
    "CNSSampler_CHENGOU": CNSSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CNSSampler_CHENGOU": "CNS Sampler (Colored Noise) | CHENGOU",
}
