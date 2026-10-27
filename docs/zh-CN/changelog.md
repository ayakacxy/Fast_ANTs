# 更新日志

[English](../en/changelog.md) · [简体中文](changelog.md)

## 0.2.0

本版本在 v0.1.0 的 CPU 和 Apple Metal 配准流程基础上，增加 NVIDIA CUDA 计算。

### 新增功能

- 通过可选 CuPy 依赖提供 CUDA FP32 刚体、仿射和标准 greedy SyN 配准。
- 增加 `register_cuda` Python API，以及 `fast_ants register --backend cuda --precision fp32` 命令。
- CUDA 最终场 QA 使用原生 FP64，包含逆一致性、有限值检查和精确分位数统计；拟合仍使用 FP32。
- 命令行默认输出英文进度日志，Metal 和 CUDA 在交互终端按层显示 `tqdm` 进度条，`console.log` 保留纯文本诊断记录。
- `--quiet` 关闭进度和详细诊断，`--log-every` 控制 GPU 迭代日志频率。
- 提供名为 `ants` 的项目内 CUDA conda 环境；CPU 导入不依赖 CuPy 或 MLX。

### 实现与性能

- 图像与四场 SyN 状态保留在设备端，共享几何定义和主机优化控制。
- Mattes 度量使用 CUDA Graph 重放、固定分箱分组和仅值计算；控制参数通过锁页内存传递，减少传输和调度开销。
- 双向 CC 更新与逆场调度、合并访存的场滤波、可复用几何缓冲区。
- 原生 FP64 输出 QA 无须下载密集 QA 临时数组。
- 流式 NIfTI 输出默认使用 gzip level 0，可选 pigz；`FAST_ANTS_GZIP_LEVEL` 设置压缩级别。
- RTX 2080 Ti、双路 Xeon Gold 6230、32 线程下，256×256×150 原始 MRI 完整配准耗时中位数：CUDA FP32 8.24 s、CPU FP32 51.79 s、CPU FP64 85.58 s，取两个使用已有缓存的新进程测量值。
- 对所有实测后端及官方 ANTs 进行独立写出结果检查。详见[性能设置](performance.md)和[质量评估](validation.md)；v0.1.0 的 Apple M5 结果单独保留。

## 0.1.0

Fast ANTs 采用 GPL-3.0-only，附有 ANTs 和 ITK 的许可证与来源声明。

### 功能

- 三维刚体、仿射和标准 greedy SyN 配准。
- CPU FP64、CPU FP32 和 Apple Metal FP32 后端。
- Mattes 互信息、局部互相关、多分辨率图像金字塔和变换应用。
- `fast_ants register` 命令，支持 `brain-syn` 预设和 JSON 配置。
- NIfTI 图像与位移场读写、仿射变换读写及配准质量统计。
- 英文进度日志和 Python 配准 API。
- 中英文文档、示例和开发工具。

### 性能优化

- CPU 使用 Numba 多线程、采样与度量计算融合、可复用工作区。
- FP32 稠密样本映射、Parzen 直方图更新、导数通道和无掩膜局部 CC 统计使用四路 SIMD。
- 插值共享索引计算，梯度采样复用全零背景支持域缓存，线性 Jacobian 只计算有效项，CC 复用合并系数。
- Metal 使用设备驻留计算、流式互信息导数、中心化局部统计，以及独立任务的异步提交。
- 并行影像 I/O 和分块输出处理。
