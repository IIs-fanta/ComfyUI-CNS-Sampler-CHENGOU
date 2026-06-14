
# ComfyUI-CNS-Sampler-CHENGOU

A ComfyUI custom node for **Colored Noise Sampling (CNS)**, based on
[Colored Noise Diffusion Sampling](https://arxiv.org/abs/2605.30332) by Hadar Davidson, Noam Issachar, and Sagie Benaim.

This plugin adapts the public CNS noise-shaping code from
[HadarDavidson/colored-noise-sampling](https://github.com/HadarDavidson/colored-noise-sampling)
to ComfyUI's `SAMPLER` interface.

## What's New

- Aligned gamma matrix handling with the official implementation: bundled matrices are read as `[step, freq_bin]`.
- Moved `energy_scale` after FFT filtering and unit-std normalization, matching the official CNS code path.
- Updated alpha tilting to follow the official residual-energy behavior.
- Kept compatibility with ComfyUI latent tensors that have extra leading dimensions; the last two dimensions are treated as spatial `H/W`.
- Cleaned up README/code comments and kept the node API stable for existing workflows.
<img width="7087" height="5197" alt="CNSTEST_Jc" src="https://github.com/user-attachments/assets/21ad0cac-3480-47af-bd23-a729b778b646" />

## Install

Place this folder in your ComfyUI `custom_nodes` directory:

```text
ComfyUI/
+-- custom_nodes/
    +-- ComfyUI-CNS-Sampler-CHENGOU/
        +-- __init__.py
        +-- nodes.py
        +-- gamma_matrix_scaled.pt
        +-- README.md
```

Restart ComfyUI. The node appears under:

```text
sampling / custom_sampling / samplers / CNS Sampler (Colored Noise) | CHENGOU
```

## Workflow

Use this node with **SamplerCustomAdvanced**. It outputs a normal ComfyUI `SAMPLER`.

```text
BasicScheduler
ModelPatcher
Conditioning (positive/negative)
RandomNoise / DisableNoise
CNS Sampler (Colored Noise)
        |
        +-- SamplerCustomAdvanced -- VAEDecode -- Image
```

`KSampler` does not expose the custom sampler slot in the same way, so `SamplerCustomAdvanced` is the intended path.

## Parameters

| Parameter | Default | Notes |
| --- | ---: | --- |
| `s_churn` | `0.5` | Stochastic strength for the ComfyUI Euler adaptation. `0` disables CNS noise injection. |
| `power_gamma` | `0.75` | Power on residual energy. Official unguided setting: `0.75`. |
| `gamma_divider` | `1.73` | Divides gamma before residual energy is computed. Official unguided setting: `1.73`; guided setting: `25.0`. |
| `energy_scale` | `0.98` | Applied after FFT filtering and unit-std normalization. Official unguided setting: `0.98`; guided setting: `0.998`. |
| `alpha_tilt_start` | `0.15` | Frequency tilt at the beginning of sampling. |
| `alpha_tilt_end` | `-0.5` | Frequency tilt at the end of sampling. |
| `alpha_use_fnorm` | `True` | Uses normalized radial frequency for alpha tilting. This matches the published settings. |
| `alpha_exp_interp` | `True` | Uses exponential interpolation between start/end alpha values. |
| `alpha_exp_sharpness` | `0.75` | Official unguided setting: `0.75`. |
| `num_freq_bins` | `32` | Official matrices use 32 radial frequency bins. |
| `use_bundled_gamma_matrix` | `True` | Loads `gamma_matrix_scaled.pt` from this folder. Disable to use the sigma-schedule fallback. |

## Suggested Settings

Unguided / low CFG:

```text
s_churn          = 0.5
power_gamma      = 0.75
gamma_divider    = 1.73
energy_scale     = 0.98
alpha_tilt_start = 0.15
alpha_tilt_end   = -0.5
alpha_use_fnorm  = True
alpha_exp_interp = True
```

Guided / higher CFG:

```text
s_churn          = 0.5
power_gamma      = 0.5
gamma_divider    = 25.0
energy_scale     = 0.998
alpha_tilt_start = -0.1
alpha_tilt_end   = 0.03
alpha_use_fnorm  = True
alpha_exp_interp = False
```

The bundled `gamma_matrix_scaled.pt` is the official unguided matrix from
`gamma_matrix/gamma_matrix_scaled.pt`. For guided experiments, you can replace it
with the official guided matrix, but keep the filename as `gamma_matrix_scaled.pt`
unless you also edit the loader path.

## Implementation Notes

The official repository implements CNS inside its SiT `transport` SDE integrator.
This plugin keeps the same CNS frequency-shaping path while fitting into ComfyUI's
Euler-style sampler call:

- gamma matrix layout is `[step, freq_bin]`, matching the official code;
- residual energy is computed from `1 - gamma / gamma_divider`;
- alpha tilting uses `exp(alpha * f_norm) * residual` when `alpha_use_fnorm=True`;
- FFT-filtered noise is normalized to unit std, then `energy_scale` is applied;
- the final colored noise is injected through ComfyUI's sigma schedule.

If `gamma_matrix_scaled.pt` is missing or disabled, the plugin falls back to a
simple sigma-schedule approximation. That fallback is useful for booting, but it
is not a replacement for an ODE-derived gamma matrix.

## 中文说明

这是一个 ComfyUI 自定义采样器节点，基于 CNS 论文和官方代码实现。它不是官方 SiT 训练/评估仓库的完整移植，而是把官方 CNS 的频域噪声调度接到 ComfyUI 的 `SAMPLER` 接口上。

本版重点修正：

- `gamma_matrix_scaled.pt` 按官方 `[step, freq_bin]` 方向读取；
- `energy_scale` 放在频域滤波和标准差归一化之后，参数会真正生效；
- alpha tilt 逻辑按官方 residual energy 写法实现；
- 保留 `SamplerCustomAdvanced` 工作流和原节点参数。

推荐先使用默认参数。如果你使用 guided / 高 CFG，可以参考上面的 guided 参数，但最好同时替换为官方 guided gamma matrix。

## Citation

```bibtex
@misc{davidson2026colorednoisediffusionsampling,
      title={Colored Noise Diffusion Sampling},
      author={Hadar Davidson and Noam Issachar and Sagie Benaim},
      year={2026},
      eprint={2605.30332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30332},
}
```
