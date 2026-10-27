# 代码架构

[English](../en/architecture.md) · [简体中文](architecture.md) · [文档](index.md)

软件包将共享的图像几何、优化控制与 CPU、Metal、CUDA 算子分开，各后端内部按配准阶段组织。

## 源码结构

```text
src/fast_ants/
├── api.py, cli.py, progress.py
├── core/                 # Images, grids, transforms, configuration, results
├── io/                   # NIfTI and transform I/O
├── pipeline/             # Pyramids, sampling plans, linear models, optimizers
├── backends/
│   ├── cpu/
│   │   ├── common/
│   │   ├── fp64/
│   │   └── fp32/
│   ├── cuda/             # CuPy runtime, CUDA kernels, registration stages
│   └── metal/
│       ├── runtime.py, types.py, ops.py
│       ├── registration.py, linear.py, syn.py
│       ├── images.py, filtering.py, preparation.py
│       ├── metrics/mattes/
│       ├── metrics/cc/
│       └── fields/
├── quality/
└── presets/
```

| 模块 | 主要职责 |
| --- | --- |
| `core` | 网格几何、图像／场数据结构、变换、阶段选项和结果类型 |
| `io` | 标量图像、向量场、仿射变换与分块 NIfTI 写出 |
| `pipeline` | 共享金字塔、采样选择、FP64／FP32 线性模型和优化器 |
| `backends/cpu/fp64` | 双精度配准阶段与 Numba 算子 |
| `backends/cpu/fp32` | 单精度配准阶段与 Numba 算子 |
| `backends/cpu/common` | 共享的 CPU 几何与归约函数 |
| `backends/metal` | 单精度设备 kernel 与 GPU 调度 |
| `backends/cuda` | CuPy 设备数组、CUDA kernel、配准与最终 QA |
| `quality` | 图像／场质量指标与最终输出检查 |
| `progress` | 英文阶段与迭代日志 |
| `presets` | 随软件包分发的配准配置 |

## 配准流程

`api.py` 提供配准、读写和变换函数。`cli.py` 加载预设或用户配置，选择后端并保存运行信息。一次配准包括：

1. 读取图像几何与体素数据，初始化线性变换。
2. 准备图像金字塔，完成刚体／仿射阶段。
3. 在层间重采样 SyN 状态，更新四个位移场。
4. 复合最终正逆映射，重采样移动图像。
5. 计算输出质量指标，写出变换、图像与日志。

入口分别为 [CPU FP64 配准](../../src/fast_ants/backends/cpu/fp64/registration.py)、[CPU FP32 配准](../../src/fast_ants/backends/cpu/fp32/registration.py)、[Metal 配准](../../src/fast_ants/backends/metal/registration.py)和 [CUDA 配准](../../src/fast_ants/backends/cuda/registration.py)，其数据流对应[算法说明](algorithms.md)。

## 共享几何与优化器

`core/types.py` 保存图像与网格信息，`core/fields.py` 和 `core/state.py` 定义位移场与 SyN 状态。小型矩阵运算和标量算术位于 `core/linalg.py`、`core/arithmetic32.py` / `arithmetic64.py`。

CPU FP32、Metal 与 CUDA 共用 `pipeline` 中的 FP32 线性模型和优化器。变换状态、参数更新、直方图终结处理与收敛计算使用各自的精度设置，图像规模的运算由选定后端执行。

CPU 导入不会初始化 Metal 或 CUDA。各公共 GPU 入口在调用时加载实现，因此未安装 MLX 或 CuPy 也可使用 CPU 后端。

## CPU 算子

Numba 编译采样、有限差分、滤波、度量计算和位移场运算。算子按独立行、切片、样本块或滤波线并行，共享归约函数合并分块结果。

每个顺序执行的配准层级拥有自己的临时缓冲区，内部更新可重复使用这些缓冲；公共函数返回的数组和回调保留的结果仍可独立使用。

源码：[CPU FP64](../../src/fast_ants/backends/cpu/fp64)、[CPU FP32](../../src/fast_ants/backends/cpu/fp32)和[共享 CPU 函数](../../src/fast_ants/backends/cpu/common)。

## Metal 算子

[`MetalContext`](../../src/fast_ants/backends/metal/runtime.py) 管理自定义 kernel 编译、调度、设备数组、异步提交和主机／设备传输。[设备类型](../../src/fast_ants/backends/metal/types.py)将数组与网格关联，[`MetalOps`](../../src/fast_ants/backends/metal/ops.py)通过静态继承组合图像、CC、场运算和 SyN 方法。

Mattes MI 分为采样、直方图构造、导数和终结处理。CC 包括短窗口、共享内存分块和通用邻域实现。场模块实现复合、求逆和双侧逆场调度。

拟合期间图像和场保留在 GPU 上，主机处理小型几何／系数计算与优化器控制。最终数组下载后用于写出和 CPU 质量检查。彼此独立的 GPU 工作可以在主机回读之前提交；后续迭代仍等待决定下一步所必需的数值。

具体实现见[实现优化](optimizations.md)。

## CUDA 算子与配准

[`CUDAContext`](../../src/fast_ants/backends/cuda/runtime.py) 通过可选 CuPy 管理设备数组、自定义 kernel 编译与调度、同步及显式主机／设备传输。[设备类型](../../src/fast_ants/backends/cuda/types.py)将连续 FP32 图像和位移场与共享 `Grid` 几何关联。[`CUDAImageOps`](../../src/fast_ants/backends/cuda/images.py)提供图像上传／下载、零场、强度归一化、三线性／最近邻 pull 采样和锚定采样的物理梯度。

配准期间，GPU 上的图像和位移场使用 FP32。小型几何矩阵和原点差先在 CPU 上以 FP64 计算，再用于生成局部坐标。最终质量检查使用独立的 FP64 临时数组。`common.cuh` 中的 kernel 沿用 FP32 实现的采样范围和边界规则。CUDA 模块独立于 Metal，CPU 入口不会导入它们。

[`CUDAOps`](../../src/fast_ants/backends/cuda/ops.py)组合图像、场、CC 和 SyN 算子。高斯滤波、Mattes 采样／直方图／导数／终结处理、局部 CC、场复合和求逆均在 GPU 上执行。[`register_cuda`](../../src/fast_ants/backends/cuda/registration.py)将这些算子接入共享阶段配置、线性模型、优化器控制和多分辨率计划，密集图像及四场 SyN 状态保留在设备端。

小型几何矩阵、滤波系数、优化器和收敛判断在 CPU 上计算。最终位移场的质量检查在 GPU 上使用 FP64，计算精确分位数并检查有限值；程序将统计结果、最终场和重采样图像传回 CPU 后写出。CUDA 的 Mattes 直方图分块累加，再依次合并各块。浮点累加顺序可能影响末位数值和后续迭代，因此应在相同输入下比较最终对齐效果、形变和逆一致性，不要求结果逐位相同。

测试方法与后续验证、优化工作见[开发指南](development.md)和[开发计划](roadmap.md)。
