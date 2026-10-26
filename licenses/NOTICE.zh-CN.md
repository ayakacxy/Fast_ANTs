# 第三方声明

[English](NOTICE.md) · [简体中文](NOTICE.zh-CN.md)

## Fast ANTs

Fast ANTs 采用 GNU General Public License version 3，许可证标识为
`GPL-3.0-only`，原文见 [LICENSE](../LICENSE)。下述上游代码保留原有版权、
许可条件和来源声明，其 Apache-2.0 许可证原文与项目许可证一并提供。

## ITK

Fast ANTs 的部分实现由 ITK 源码翻译并修改，版权归 NumFOCUS 所有，
采用 Apache License 2.0。许可证原文保留在 [LICENSE-ITK](LICENSE-ITK)。

主要参考版本为 ITK 提交
`4c49df934c316ff87dbbd971b89e63df6bbd1836`（5.4.7）。参考文件及校验值见
[ITK 源码清单](../docs/assets/provenance/itk-v5.4.7-sources.json)。

| 组件 | ITK 源码 | Python 实现 |
| --- | --- | --- |
| 高斯滤波 | `itkRecursiveGaussianImageFilter.hxx`、`itkRecursiveSeparableImageFilter.hxx`、`itkGaussianOperator.hxx` | `pipeline/gaussian.py`、CPU 滤波模块、Metal `filtering.py` 和 `gaussian_sections.py` |
| Mattes 互信息 | Mattes 度量及值／导数计算的 threader | CPU 和 Metal `metrics/mattes`、`pipeline/mattes32.py`、`pipeline/mattes64.py` |
| 局部互相关 | `itkANTSNeighborhoodCorrelationImageToImageMetricv4GetValueAndDerivativeThreader.hxx` | CPU 和 Metal `metrics/cc` |
| 位移场复合与求逆 | `itkComposeDisplacementFieldsImageFilter.hxx`、`itkInvertDisplacementFieldImageFilter.hxx` | CPU 和 Metal 位移场模块 |
| 标准 greedy SyN | `itkSyNImageRegistrationMethod.hxx` | CPU 和 Metal `syn.py`、`core/state.py` |
| 优化器 | 共轭梯度线搜索、物理位移参数尺度、窗口收敛判断 | `pipeline/optimizer32.py`、`pipeline/optimizer64.py`、`pipeline/convergence.py` |
| 线性变换 | Euler3DTransform、MatrixOffsetTransformBase、CompositeTransform | `core/transforms.py`、`pipeline/models32.py`、`pipeline/models64.py` |
| 初始化与图像几何 | ImageMomentsCalculator、ImageBase、NiftiImageIO | CPU 线性初始化、`core/types.py`、`io/images.py`、`io/transforms.py` |

实现路径均相对于 `src/fast_ants`。

翻译后的代码以 Python 数组接口、Numba 循环和自定义 Metal kernel 替代
C++ 模板、图像流水线和迭代器。主要修改包括并行计算、采样与度量计算融合、
中心化局部统计、分块导数直方图、缓冲区复用、局部坐标运算和独立 GPU 任务的
异步提交。FP32 递归高斯滤波采用两个二阶节及相邻像素差分；标量 FMA
使用原生硬件指令。
部分 CPU FP32 采样、直方图和邻域运算使用四路 LLVM 向量、可复用的仅值
工作区，以及缓存的全零背景梯度支持域。

## ANTs

图像预处理参考 ANTs `PreprocessImage` 及 ITK 的直方图端点和强度映射计算。
上游 Apache License 2.0 原文保留在 [LICENSE-ANTs](LICENSE-ANTs)，
复制自固定 ANTs 版本的 `COPYING.txt`。参考提交为
`38144917d78c20144f4a70e4a6e03c059aaefef8`。

ANTs 和 ITK 参考文件保留各自原有的版权与许可证声明。

## 小矩阵例程

`core/linalg.py` 将 LINPACK DSVDC 的双对角 SVD／QR 算法及其 BLAS 缩放范数、
Givens 旋转运算专用于 3×3 FP64 几何矩阵。算法参考为 ITK 5.4.1 包含的
原始 Netlib 例程，来源地址与校验值保留在
[数值源码清单](../docs/assets/provenance/itk-v5.4.1-numerical-sources.json)。

Python／Numba 实现采用从零开始的数组索引、专用标量 QR 运算和显式硬件 FMA。
VNL 仅供独立参考工具使用，不是配准运行依赖。

## 数据与依赖

MRI 和模板的来源说明分别保存在 `validation/sources` 及 `templates` 中，
数据的再分发条件独立于上述软件许可证。

NumPy、Numba、NiBabel 和可选的 MLX 保留各自的许可证。
