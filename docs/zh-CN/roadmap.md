# 开发计划

[English](../en/roadmap.md) · [简体中文](roadmap.md) · [文档](index.md)

## 支持的功能

- 三维刚体、仿射和标准 greedy SyN 配准。
- 线性阶段使用 Mattes 互信息，SyN 使用局部归一化互相关。
- CPU FP64、CPU FP32、Apple Metal FP32 和 CUDA FP32 计算。
- 图像金字塔、仿射与位移场读写、图像重采样和质量统计。
- 命令行与 Python 接口、JSON 预设和英文进度日志。

## NVIDIA CUDA

CUDA 支持 FP32 刚体、仿射和 SyN 配准，最终场质量检查使用 FP64。

Linux 与 Windows 实测结果见[性能测试](performance.md)和[测试与质量评估](validation.md)。

后续工作包括 FP64 拟合、更广泛的 MRI 和几何测试、更多 NVIDIA 设备验证，以及内存和同步开销分析。

## 其他后续工作

- 补充不同配准方向、体素间距和采集设置的使用示例。
- 增加不同操作系统及依赖版本上的 CPU 自动化测试。
- 完善 CPU 与 Metal 后端的逐迭代日志。
- 分析度量计算、位移场操作、内存分配和 GPU 同步的开销。
- 补充预设与 API 示例，保持 CLI 和 Python 的参数处理一致。
