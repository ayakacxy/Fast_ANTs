# 安装

[English](../en/installation.md) · [简体中文](installation.md) · [文档](index.md)

## 环境要求

- Python 3.11 或更新版本。
- CPU 后端依赖 NumPy、NiBabel 和 Numba。
- Metal 后端需要 Apple Silicon macOS 和 MLX。

CPU 和 GPU 配准实现在软件包中提供；`validation/` 中的对照工具使用官方 ANTs。

## 使用 pip 安装

创建虚拟环境，在项目目录中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
fast-ants --help
```

Windows PowerShell 中使用 `.venv\Scripts\Activate.ps1` 激活环境。

Apple Silicon macOS 启用 Metal 后端：

```bash
python -m pip install -e '.[metal]'
fast-ants register --help
```

Metal 后端使用 FP32 计算；CPU 后端支持 FP64 和 FP32，可独立于 MLX 加载。

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
