# 安装

[English](../en/installation.md) · [简体中文](installation.md) · [文档](index.md)

## 环境要求

- Python 3.11 或更新版本。
- CPU 后端依赖 NumPy、NiBabel 和 Numba。
- Metal 后端需要 Apple Silicon macOS 和 MLX。

CPU 和 GPU 配准实现在软件包中提供；`validation/` 中的对照工具使用官方 ANTs。

## 安装发行包

从 [v0.1.0 发布页](https://github.com/ayakacxy/Fast_ANTs/releases/tag/v0.1.0)下载 `fast_ants-0.1.0-py3-none-any.whl`。创建虚拟环境并安装 wheel：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./fast_ants-0.1.0-py3-none-any.whl
fast-ants --version
```

Windows PowerShell 中使用 `.venv\Scripts\Activate.ps1` 激活环境。

Apple Silicon macOS 启用 Metal 后端：

```bash
python -m pip install './fast_ants-0.1.0-py3-none-any.whl[metal]'
fast-ants register --help
```

Metal 后端使用 FP32 计算；CPU 后端支持 FP64 和 FP32，可独立于 MLX 加载。

## 从源码安装

克隆仓库，或解压发布页提供的完整源码 ZIP：

```bash
git clone https://github.com/ayakacxy/Fast_ANTs.git
cd Fast_ANTs
python -m pip install -e .
```

Metal 安装命令为 `python -m pip install -e '.[metal]'`。完整源码 ZIP 包含示例、测试和验证工具；Python 源码发行包包含可安装的软件包和文档。

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

## 开发环境

```bash
python -m pip install -e '.[test]'
python -m pip wheel . --no-deps -w dist
```

测试命令和构建方法见[开发说明](development.md)。MRI 数据、模板图像和配准输出与安装软件包分开保存。
