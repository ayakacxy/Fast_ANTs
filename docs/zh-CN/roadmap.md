# 开发计划

[English](../en/roadmap.md) · [简体中文](roadmap.md) · [文档目录](index.md)

## 支持的功能

- 三维刚体、仿射和标准 greedy SyN 配准。
- 线性阶段使用 Mattes 互信息，SyN 使用局部互相关。
- CPU FP64、CPU FP32、Apple Metal FP32 和 CUDA FP32 计算。
- 图像金字塔、仿射与位移场读写、图像重采样和质量统计。
- 命令行与 Python 接口、JSON 预设和英文进度日志。

## NVIDIA CUDA

CUDA 后端通过 CuPy 实现 FP32 图像与场运算、Mattes MI、双向 CC 更新、四场 SyN 和原生 FP64 输出 QA。小型优化和几何控制仍在主机端执行。CUDA 依赖保持可选，设备不可用时明确报错。

后续工作包括补充 MRI／几何测试、分析内存和同步开销，以及在更多 NVIDIA 设备上验证。FP64 拟合需要独立实现和验证；最终 FP64 QA 不等于支持 FP64 拟合。

## 其他后续工作

- 补充不同配准方向、体素间距和采集设置的使用示例。
- 增加不同操作系统及依赖版本上的 CPU 自动化测试。
- 完善 CPU 与 Metal 后端的逐迭代日志。
- 分析度量计算、形变场操作、内存分配和 GPU 同步的开销。
- 补充预设与 API 示例，保持 CLI 和 Python 的参数处理一致。

Fast ANTs 专注于图像配准和变换应用，分割、皮层厚度等神经影像分析组件不在项目范围内。
