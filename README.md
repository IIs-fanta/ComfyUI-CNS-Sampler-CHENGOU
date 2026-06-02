# ComfyUI-CNS-Sampler-CHENGOU

A ComfyUI custom node implementing **Colored Noise Sampling (CNS)** — a plug-and-play SDE sampler that actively exploits the spectral bias of diffusion models to improve image quality. | by **CHENGOU**

Based on the paper [Colored Noise Diffusion Sampling](https://arxiv.org/abs/2605.30332) (Davidson et al., 2026).

## Latest Fixes

- Fixed a sampler crash when ComfyUI passes latent tensors with extra leading dimensions, such as video or XPU backend shapes. CNS now treats the last two tensor dimensions as spatial `H/W` and supports both standard 4D latents and 5D+ latents. See [issue #3](https://github.com/IIs-fanta/ComfyUI-CNS-Sampler-CHENGOU/issues/3).
- Restored terminal progress output for the custom CNS sampling loop using `tqdm`, while still respecting ComfyUI's `disable_pbar` flag. The terminal now shows step progress, elapsed time, and ETA. See [issue #1](https://github.com/IIs-fanta/ComfyUI-CNS-Sampler-CHENGOU/issues/1).

<details>
<summary>📖 中文说明 / Chinese README</summary>

## ComfyUI-CNS-Sampler-CHENGOU

ComfyUI 自定义节点：**Colored Noise Sampler（彩色噪声采样器）** | CHENGOU

基于论文 [Colored Noise Diffusion Sampling](https://arxiv.org/abs/2605.30332)（Davidson et al., 2026）实现。

---

### 安装

将整个 `ComfyUI-CNS-Sampler-CHENGOU` 文件夹放入你的 ComfyUI `custom_nodes` 目录：

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-CNS-Sampler-CHENGOU/
        ├── __init__.py
        ├── nodes.py
        └── README.md
```

重启 ComfyUI，节点会出现在：
`Add Node → sampling → custom_sampling → samplers → CNS Sampler (Colored Noise) | CHENGOU`

---

### 工作流接法

CNS 节点输出一个 `SAMPLER`，配合 **`SamplerCustomAdvanced`** 使用：

```
BasicScheduler ──────────────────────────────┐
                                             ▼
ModelPatcher ─────────────────────► SamplerCustomAdvanced ──► VAEDecode ──► 图片
                                             ▲
CNS Sampler (Colored Noise) ─────────────────┘
Conditioning (正向/负向) ──────────────────────┘
RandomNoise / DisableNoise ───────────────────┘
```

**关键：** 一定要用 `SamplerCustomAdvanced`（不是 KSampler），这样才能插入自定义采样器。

---

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `s_churn` | 0.5 | SDE 噪声强度。0 = 纯 ODE（无随机性），1.0 = 强随机性 |
| `power_gamma` | 0.75 | 残差能量指数。越低颜色效果越温和。无引导时论文用 0.75 |
| `gamma_divider` | 1.73 | 削弱彩色化效果的除数。无引导 1.73，有引导（CFG）25.0 |
| `energy_scale` | 0.98 | 全局能量缩放。无引导 0.98，有引导 0.998 |
| `alpha_tilt_start` | 0.15 | 采样开始时的频率倾斜（正值=增强高频） |
| `alpha_tilt_end` | -0.5 | 采样结束时的频率倾斜 |
| `alpha_use_fnorm` | True | 按归一化频率位置加权倾斜（推荐开启） |
| `alpha_exp_interp` | True | 用指数插值替代线性插值 |
| `alpha_exp_sharpness` | 0.75 | 指数插值曲线的锐度 |
| `num_freq_bins` | 32 | 径向频率分箱数量 |
| `gamma_matrix_pt` | （空） | 可选：官方预计算 gamma 矩阵路径（.pt 文件） |

---

### FLUX 推荐参数

**无引导 / 低 CFG（FLUX.1-dev, CFG ≈ 1.0）：**
```
s_churn          = 0.5
power_gamma      = 0.75
gamma_divider    = 1.73
energy_scale     = 0.98
alpha_tilt_start = 0.15
alpha_tilt_end   = -0.5
alpha_use_fnorm  = True
alpha_exp_interp = True
```

**有引导（FLUX.1-dev, CFG 3.5）：**
```
s_churn          = 0.5
power_gamma      = 0.5
gamma_divider    = 25.0
energy_scale     = 0.998
alpha_tilt_start = -0.1
alpha_tilt_end   = 0.03
alpha_use_fnorm  = True
alpha_exp_interp = False
```

---

### 关于 Gamma Matrix

本节点内置了一个基于 sigma schedule 的近似 gamma matrix，开箱即用无需额外文件。

如果你有官方仓库里的 gamma_matrix_scaled.pt，放到 node 文件夹里，它会自动加载。如果不行，内置的基于西格玛的近似方法也没问题。
https://github.com/HadarDavidson/colored-noise-sampling

下载 `gamma_matrix/gamma_matrix_scaled.pt` 

---

### 原理简述

标准 SDE 采样在每一步都注入均匀白噪声，浪费了有限的能量预算。

CNS 的做法：
1. 追踪每个频率波段的"建立进度" γ(f, t)
2. 计算每个频段还缺少多少能量
3. 把噪声能量重新分配到还没建立好的频段

结果是相同的步数、相同的模型，出图质量更好（FID 更低，细节更清晰，结构更连贯）。

---

### 引用

```bibtex
@misc{davidson2026colorednoisediffusionsampling,
      title={Colored Noise Diffusion Sampling},
      author={Hadar Davidson and Noam Issachar and Sagie Benaim},
      year={2026},
      eprint={2605.30332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

</details>

---

## Installation

Drop the `ComfyUI-CNS-Sampler-CHENGOU` folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-CNS-Sampler-CHENGOU/
        ├── __init__.py
        ├── nodes.py
        └── README.md
```

Restart ComfyUI. The node will appear at:
`Add Node → sampling → custom_sampling → samplers → CNS Sampler (Colored Noise) | CHENGOU`

---

## Workflow

CNS outputs a `SAMPLER` — use it with **`SamplerCustomAdvanced`** (not KSampler):

```
BasicScheduler ──────────────────────────────┐
                                             ▼
ModelPatcher ─────────────────────► SamplerCustomAdvanced ──► VAEDecode ──► Image
                                             ▲
CNS Sampler (Colored Noise) ─────────────────┘
Conditioning (pos / neg) ─────────────────────┘
RandomNoise / DisableNoise ───────────────────┘
```

> **Important:** `SamplerCustomAdvanced` is required — it exposes the sampler slot that CNS plugs into.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s_churn` | 0.5 | SDE noise strength. `0` = pure ODE (deterministic), `1.0` = strong stochasticity |
| `power_gamma` | 0.75 | Exponent on residual energy `(1-γ)`. Lower = gentler coloring. Paper uses `0.75` unguided |
| `gamma_divider` | 1.73 | Divides γ values to weaken the coloring effect. `1.73` unguided, `25.0` guided (CFG) |
| `energy_scale` | 0.98 | Global energy multiplier after normalisation. `0.98` unguided, `0.998` guided |
| `alpha_tilt_start` | 0.15 | Frequency tilt at sampling start. Positive = boost high-freq |
| `alpha_tilt_end` | -0.5 | Frequency tilt at sampling end |
| `alpha_use_fnorm` | True | Weight tilt by normalised frequency position (recommended) |
| `alpha_exp_interp` | True | Exponential (vs linear) interpolation between alpha values |
| `alpha_exp_sharpness` | 0.75 | Sharpness of exponential alpha interpolation |
| `num_freq_bins` | 32 | Number of radial frequency bins |
| `gamma_matrix_pt` | *(empty)* | Optional: path to a precomputed `.pt` gamma matrix from the official repo |

---

## Recommended Settings for FLUX

**Unguided / low CFG (FLUX.1-dev, CFG ≈ 1.0) — use defaults:**
```
s_churn          = 0.5
power_gamma      = 0.75
gamma_divider    = 1.73
energy_scale     = 0.98
alpha_tilt_start = 0.15
alpha_tilt_end   = -0.5
alpha_use_fnorm  = True
alpha_exp_interp = True
```

**Guided (FLUX.1-dev, CFG 3.5):**
```
s_churn          = 0.5
power_gamma      = 0.5
gamma_divider    = 25.0
energy_scale     = 0.998
alpha_tilt_start = -0.1
alpha_tilt_end   = 0.03
alpha_use_fnorm  = True
alpha_exp_interp = False
```

---

## Gamma Matrix

The node ships with a built-in approximation of the gamma matrix derived from the sigma schedule — no extra files needed to get started.

If you have gamma_matrix_scaled.pt from the official repo, put it in the node folder and it loads automatically. If not, there's a built-in sigma-based approximation that works fine.

https://github.com/HadarDavidson/colored-noise-sampling

Download `gamma_matrix/gamma_matrix_scaled.pt` 

---

## How It Works

Standard SDE samplers inject uniform white noise at every step, wasting their finite energy budget on frequency bands that are already structurally resolved.

CNS instead:
1. Tracks each frequency band's "resolution progress" γ(f, t) across the sampling trajectory
2. Computes how much structural energy each band still lacks
3. Dynamically routes injected noise energy toward the most underbuilt bands

The result is the same model, same step count — with better image quality (lower FID, sharper detail, more coherent structure).

---

## Citation

```bibtex
@misc{davidson2026colorednoisediffusionsampling,
      title={Colored Noise Diffusion Sampling},
      author={Hadar Davidson and Noam Issachar and Sagie Benaim},
      year={2026},
      eprint={2605.30332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
