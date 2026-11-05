# 安装说明

[English](../en/installation.md) · [简体中文](installation.md) · [文档](index.md)

## 环境要求

- Python 3.11 或更新版本。
- CPU 后端依赖 NumPy、NiBabel 和 Numba。
- Metal 后端需要 Apple Silicon macOS 和 MLX。
- CUDA 后端需要 NVIDIA GPU、兼容的驱动、CUDA 12.x 或 13.x 运行库及 CuPy。

发行包名、导入名和命令行入口均为 `fast_ants`。`validation/` 中的对照测试使用官方 ANTs。

## 安装 wheel

在项目根目录构建 v0.3.0 wheel。已发布的版本可从 [GitHub Releases](https://github.com/ayakacxy/Fast_ANTs/releases) 下载。

GPU 运行按后端选择可选依赖：

| 可选依赖 | 后端 | 依赖包 |
| --- | --- | --- |
| `metal` | Apple Metal FP32 | MLX |
| `cuda` | CUDA 12.x 的 CUDA FP32 | `cupy-cuda12x>=14,<15` |
| `cuda13` | CUDA 13.x 的 CUDA FP32 | `cupy-cuda13x>=14,<15` |

CPU 支持 FP64 和 FP32，Metal 与 CUDA 拟合使用 FP32，CUDA 最终场质量检查使用 FP64。同一环境安装与 CUDA 运行库对应的一种 CuPy。

### Windows PowerShell

创建环境、构建 wheel，并运行 CPU 配准：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip wheel . --no-deps -w dist
.\.venv\Scripts\python.exe -m pip install './dist/fast_ants-0.3.0-py3-none-any.whl'
.\.venv\Scripts\fast_ants.exe --version
.\.venv\Scripts\fast_ants.exe register fixed.nii.gz moving.nii.gz --output registration-cpu --backend cpu --precision fp64 --preset brain-syn --cpu-threads 16
```

CUDA 13.x 在同一环境中补充可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install './dist/fast_ants-0.3.0-py3-none-any.whl[cuda13]'
```

CPU 线程数按本机配置选择。Numba 和 CUDA 内核在首次使用时编译；缓存运行和首次运行耗时见 [Windows 性能测试](performance.md#windows-cuda-与-cpu16-线程)。

与 ANTs 对照时，解压 [ANTs 2.6.5 发布页](https://github.com/ANTsX/ANTs/releases/tag/v2.6.5)中的 Windows 压缩包，通过验证工具的 `--ants-registration` 参数传入 `antsRegistration.exe` 的路径。

### Linux 与 macOS（Bash）

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip wheel . --no-deps -w dist
python -m pip install ./dist/fast_ants-0.3.0-py3-none-any.whl
fast_ants --version
```

GPU 运行时，在 wheel 文件名后添加表中对应的可选依赖；例如 Apple Silicon macOS 使用 `fast_ants-0.3.0-py3-none-any.whl[metal]`。

## 从源码安装

在[源码树](https://github.com/ayakacxy/Fast_ANTs)中运行：

```bash
python -m pip install -e .
```

添加 `[metal]`、`[cuda]` 或 `[cuda13]` 安装对应 GPU 依赖，例如 `python -m pip install -e '.[cuda13]'`。完整源码压缩包包含测试和验证工具；Python 源码发行包包含软件包和文档。

## Conda 环境

[CPU](../../environments/cpu.yml) 和 [Metal](../../environments/metal.yml) 环境文件指定了各自的依赖：

```bash
conda env create -f environments/cpu.yml
conda activate fast-ants-cpu
python -m pip install -e .
```

Metal 使用 `environments/metal.yml`，激活 `fast-ants-metal` 后安装 `.[metal]`。[参考环境](../../environments/reference.yml)包含数值对照所需的 ANTs。

## CUDA 环境

[CUDA 环境](../../environments/cuda.yml)使用 Python 3.11 和 CUDA 12.x 的 CuPy 包：

```bash
conda env create -f environments/cuda.yml
conda activate ants
python -m pip install -e '.[cuda]'
```

CUDA 运行库可由已安装的 toolkit 或 CuPy 的 `[ctk]` 可选依赖提供，例如 CUDA 12.x 使用 `python -m pip install 'cupy-cuda12x[ctk]>=14,<15'`。存在多个 toolkit 时，使用 `CUDA_PATH` 选择安装目录。详见 [CuPy 安装指南](https://docs.cupy.dev/en/stable/install.html)。

## 开发环境

```bash
python -m pip install -e '.[test,build]'
```

测试和构建方法见[开发指南](development.md)。
