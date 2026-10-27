<h1 align="center">Fast ANTs</h1>

<div align="center">

**基于 ANTs 配准算法的 Python 加速项目。**

[![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-5D5FEF)](docs/zh-CN/changelog.md)
[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![CPU: Numba](https://img.shields.io/badge/CPU-Numba-00A3E0)](docs/zh-CN/optimizations.md)
[![GPU: Metal](https://img.shields.io/badge/GPU-Metal-333333?logo=apple&logoColor=white)](docs/zh-CN/installation.md)
[![Precision: FP64 / FP32](https://img.shields.io/badge/Precision-FP64%20%2F%20FP32-6D4AFF)](docs/zh-CN/architecture.md)
[![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA_FP32-76B900?logo=nvidia&logoColor=white)](docs/zh-CN/installation.md#cuda-环境)

[English](README.md) · [简体中文](README.zh-CN.md) · [文档](docs/zh-CN/index.md) · [算法](docs/zh-CN/algorithms.md) · [优化实现](docs/zh-CN/optimizations.md) · [性能测试](docs/zh-CN/performance.md)

🧠 **刚体 / 仿射 / SyN** · ⚙️ **CPU 多线程** · ⚡ **Metal / CUDA** · 🐍 **Python 与命令行**

</div>

## ⚡ 性能

| 平台 / 线程 | 后端 | 耗时 / s | 加速比 | 前景 NCC | 对应点平均误差 / mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Apple M5 / 10 | Metal FP32 | **15.34** | **33.1×** | 0.6626 | 1.430 |
| Apple M5 / 10 | CPU FP32 | 42.57 | 11.9× | 0.6645 | 1.438 |
| Apple M5 / 10 | CPU FP64 | 60.44 | 8.37× | 0.6644 | 1.434 |
| Apple M5 / 10 | 官方 ANTs FP32 | 507.48 | — | 0.6644 | 1.432 |
| Apple M5 / 10 | 官方 ANTs FP64 | 505.85 | — | 0.6644 | 1.434 |
| RTX 2080 Ti + 双路 Xeon / 32 | CUDA FP32 | **8.24** | **40.6×** | 0.6630–0.6644 | 1.431–1.434 |
| RTX 2080 Ti + 双路 Xeon / 32 | CPU FP32 | 51.79 | 6.46× | 0.6644 | 1.431 |
| RTX 2080 Ti + 双路 Xeon / 32 | CPU FP64 | 85.58 | 4.63× | 0.6644 | 1.434 |
| RTX 2080 Ti + 双路 Xeon / 32 | 官方 ANTs FP32 | 334.32 | — | 0.6629–0.6644 | 1.425–1.433 |
| RTX 2080 Ti + 双路 Xeon / 32 | 官方 ANTs FP64 | 396.34 | — | 0.6639 | 1.434 |

使用同一组 256×256×150 MRI 图像；Apple M5 的 v0.1.0 结果为单次测量，Linux 的 v0.2.0 结果为两次缓存运行的中位数。耗时包含启动、读取、配准和完整写出，Fast ANTs 还包含质量检查。加速比与相同精度的官方 ANTs 比较；对应点误差是配准后四组粗略标注位置的平均距离，越小表示对齐越好。

Apple M5 质量指标通过同一 CPU FP64 采样器重新应用 v0.1.0 已保存变换计算，前景包含 3,236,832 个体素，地标来自四组非专家粗略标注。Metal 保存的正逆场 Jacobian 均为正，两个复合方向中较大的逆一致性 RMS 为 0.02297 mm。

[性能测试详情](docs/zh-CN/performance.md) · [质量评估](docs/zh-CN/validation.md)

## 📚 目录

- [性能](#-性能)
- [功能](#-功能)
- [算法原理](#-算法原理)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [Python 接口](#-python-接口)
- [文档](#-文档)
- [许可与致谢](#-许可与致谢)

项目参考 ANTs/ITK 的刚体、仿射和 greedy SyN 配准流程，用 Python 重写采样、相似性度量、滤波和位移场运算等核心计算，通过 Numba 多线程、MLX 自定义 Metal 内核及可选 CuPy CUDA 内核加速执行，并支持 ANTs 兼容的变换读写。

## 🧠 功能

- 刚体 → 仿射 → SyN 配准，支持多分辨率图像金字塔。
- 线性配准使用 Mattes 互信息，SyN 使用局部交叉相关。
- 提供 CPU FP64、CPU FP32、Apple GPU FP32 和 CUDA FP32 后端。
- 支持 ANTs 兼容的仿射变换与形变场读写。
- 提供命令行和 Python 接口、英文运行日志及输出质量检查。

CPU 内核采用多线程、SIMD、度量计算融合和工作区复用。Metal 后端在 GPU 数组中保存图像与形变场，融合采样和直方图计算，并批量提交独立任务以减少同步。详见[优化实现](docs/zh-CN/optimizations.md)。

CUDA 后端使用 CuPy 自定义 kernel 和 CUDA Graph，在 GPU 上保留图像和位移场，并使用 FP64 完成最终质量检查。CPU 导入不依赖可选 GPU 运行库。详见[安装说明](docs/zh-CN/installation.md)和[实现优化](docs/zh-CN/optimizations.md)。

## 🧮 算法原理

[算法原理文档](docs/zh-CN/algorithms.md)介绍刚体与仿射变换、Mattes 互信息、局部 CC 和 SyN 的数学实现，并说明图像梯度、多分辨率优化、形变场复合及逆场迭代。

## 📦 安装

Python 发行包名、导入名和命令行入口均为 `fast_ants`。

要求 Python 3.11 或更新版本。可从 [v0.2.0 发布页](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.2.0)下载 wheel，或在项目根目录构建并安装：

```bash
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_ants-0.2.0-py3-none-any.whl
```

Apple Silicon macOS 使用 Metal 后端时：

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[metal]'
```

NVIDIA CUDA 12.x 使用：

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[cuda]'
```

CPU 后端依赖 NumPy、NiBabel 和 Numba，Metal 使用 MLX，CUDA 使用 CuPy。wheel 构建、源码安装、conda 环境和保留的 v0.1.0 发行包见[安装说明](docs/zh-CN/installation.md)。

## 🚀 快速开始

使用 CPU 将移动图像配准到固定图像：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-cpu --backend cpu --precision fp64 \
  --preset brain-syn --threads 10 --verbose
```

使用 Apple GPU：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-metal --backend metal --precision fp32 \
  --preset brain-syn --threads 10 --verbose
```

使用 NVIDIA GPU：

```bash
fast_ants register fixed.nii.gz moving.nii.gz \
  --output registration-cuda --backend cuda --precision fp32 \
  --preset brain-syn --threads 32
```

输出目录需要是新目录，线程数应按机器配置设置。默认输入选项为 `itk_scanner` 图像几何和 `itk` 强度缩放；使用标准 NiBabel NIfTI 读取方式时，添加 `--geometry-policy nifti --scaling-policy nifti`。

命令行默认输出进度日志；Metal 和 CUDA 在交互式终端按层显示 `tqdm` 进度条，`console.log` 保存纯文本诊断。使用 `--quiet` 可关闭进度和详细日志。

输出包括仿射变换、正向和逆向形变场、重采样图像、迭代记录、质量检查结果及 `console.log`。详见[使用说明](docs/zh-CN/usage.md)和[输入与输出](docs/zh-CN/inputs-outputs.md)。

## 🐍 Python 接口

```python
from fast_ants import load_images, register_metal

fixed, moving = load_images(
    ("fixed.nii.gz", "moving.nii.gz"),
    geometry_policy="itk_scanner", scaling_policy="itk",
)
result = register_metal(
    fixed, moving, output_directory="registration-api",
    verbose=True, log_every=1,
)
warped = result.warped
```

`register` 使用 CPU FP64，`register32` 使用 CPU FP32，`register_metal` 使用 Metal，`register_cuda` 使用 CUDA。Python 接口默认采用两层配准参数，命令行的 `brain-syn` 预设采用四层参数。[此示例](examples/register_api.py)演示如何在 Python 中运行完整预设。

## 📖 文档

- [参数配置](configs/README.zh-CN.md)：预设参数和自定义 JSON 文件。
- [算法](docs/zh-CN/algorithms.md)：变换模型、相似性度量与 SyN 更新。
- [优化实现](docs/zh-CN/optimizations.md)：CPU 并行、算子融合、内存复用与 GPU 调度。
- [代码结构](docs/zh-CN/architecture.md)：模块组织和各后端实现。
- [测试与质量评估](docs/zh-CN/validation.md)。
- [开发说明](docs/zh-CN/development.md)与[贡献指南](.github/CONTRIBUTING.zh-CN.md)。
- [开发计划](docs/zh-CN/roadmap.md)。

## 📄 许可与致谢

Fast ANTs 采用 [GNU General Public License v3.0](LICENSE)，许可证标识为 `GPL-3.0-only`。

项目参考 ANTs 和 ITK 中的配准方法，保留其 Apache-2.0 许可证原文：[LICENSE-ANTs](licenses/LICENSE-ANTs) 和 [LICENSE-ITK](licenses/LICENSE-ITK)。版权声明与实现来源见 [第三方声明](licenses/NOTICE.zh-CN.md)。
