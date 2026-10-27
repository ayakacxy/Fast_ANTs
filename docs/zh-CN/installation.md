# 安装

[English](../en/installation.md) · [简体中文](installation.md) · [文档](index.md)

## 环境要求

- Python 3.11 或更新版本。
- CPU 后端依赖 NumPy、NiBabel 和 Numba。
- Metal 后端需要 Apple Silicon macOS 和 MLX。
- CUDA 后端需要 NVIDIA GPU、CUDA 12.x toolkit 和 CuPy。

CPU 和 GPU 配准实现在软件包中提供；`validation/` 中的对照工具使用官方 ANTs。

发行包名、Python 导入名和命令行入口均为 `fast_ants`，源码仓库为 `Fast_ANTs`。

## 安装 wheel

从 [v0.2.0 发布页](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.2.0)下载 wheel，或在源码树中创建虚拟环境、构建并安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_ants-0.2.0-py3-none-any.whl
fast_ants --version
```

Windows PowerShell 中使用 `.venv\Scripts\Activate.ps1` 激活环境。

Apple Silicon macOS 启用 Metal 后端：

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[metal]'
fast_ants register --help
```

兼容的 NVIDIA 系统启用 CUDA 后端：

```bash
python -m pip install './dist/fast_ants-0.2.0-py3-none-any.whl[cuda]'
```

Metal 和 CUDA 拟合使用 FP32；CPU 支持 FP64 和 FP32，可独立于 MLX 和 CuPy 加载。[v0.1.0 发布页](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.1.0)保留了 CPU 和 Metal wheel。

## 从源码安装

项目源码维护在 [Fast_ANTs 仓库](https://github.com/ayakacxy/Fast_ANTs)。在 v0.2.0 源码树根目录安装 CPU 依赖：

```bash
python -m pip install -e .
```

Metal 使用 `python -m pip install -e '.[metal]'`，CUDA 使用 `python -m pip install -e '.[cuda]'`。构建 wheel 可运行 `python -m pip wheel . --no-deps -w dist`。完整源码树包含示例、测试和验证工具；Python 源码发行包包含可安装的软件包和文档。

## Conda 环境

CPU 配准使用以下环境：

```bash
conda env create -f environments/cpu.yml
conda activate fast-ants-cpu
python -m pip install -e .
```

Apple Silicon macOS 使用 Metal 环境：

```bash
conda env create -f environments/metal.yml
conda activate fast-ants-metal
python -m pip install -e '.[metal]'
```

[CPU](../../environments/cpu.yml) 和 [Metal](../../environments/metal.yml) 环境文件指定了依赖版本。[参考环境](../../environments/reference.yml)还安装了 ANTs，用于数值对照。

## CUDA 环境

[CUDA 环境](../../environments/cuda.yml)名为 `ants`，使用 Python 3.11。在项目根目录中先运行以下命令，再创建环境和安装依赖。已忽略的 `setup/cuda` 目录用于保存本项目的环境、依赖下载、临时文件和编译缓存：

```bash
mkdir -p "$PWD/setup/cuda"/{tmp,envs,pkgs,cache/{xdg,pip,cupy,cuda,numba,python}}
export CONDA_ENVS_PATH="$PWD/setup/cuda/envs"
export CONDA_PKGS_DIRS="$PWD/setup/cuda/pkgs"
export CONDA_REGISTER_ENVS=false
export TMPDIR="$PWD/setup/cuda/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/setup/cuda/cache/xdg"
export PIP_CACHE_DIR="$PWD/setup/cuda/cache/pip"
export CUPY_CACHE_DIR="$PWD/setup/cuda/cache/cupy"
export CUDA_CACHE_PATH="$PWD/setup/cuda/cache/cuda"
export NUMBA_CACHE_DIR="$PWD/setup/cuda/cache/numba"
export PYTHONPYCACHEPREFIX="$PWD/setup/cuda/cache/python"
conda env create -f environments/cuda.yml
conda activate ants
python -m pip install -e '.[cuda]'
```

在新 shell 中激活 `ants` 前，重新设置上述环境变量。可选依赖为 `cupy-cuda12x>=14,<15`；CPU 导入不需要 CuPy。该依赖使用已安装的 CUDA 12.x toolkit。存在多个 toolkit 时，通过 `CUDA_PATH` 指定所需安装目录，详见 [CuPy 安装指南](https://docs.cupy.dev/en/stable/install.html)。

通过 `--backend cuda --precision fp32` 运行完整刚体、仿射与 SyN 配准，或在 Python 中调用 `register_cuda`。CUDA 运行时或设备不可用时会报错，不会切换到 CPU。详见[使用说明](usage.md)和[开发指南](development.md)。

CUDA 拟合使用 FP32，最终场质量检查使用原生 FP64；当前不提供 CUDA FP64 拟合。真实 MRI 和设备覆盖范围见[性能测试](performance.md)和[质量评估](validation.md)。

## 开发环境

```bash
python -m pip install -e '.[test]'
python -m pip wheel . --no-deps -w dist
```

测试命令和构建方法见[开发说明](development.md)。MRI 数据、模板图像和配准输出与安装软件包分开保存。
