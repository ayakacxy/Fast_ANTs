# 文档

[English](../en/index.md) · [简体中文](index.md) · [Fast ANTs](../../README.zh-CN.md)

Fast ANTs 面向 ANTs 的刚体、仿射和标准 greedy SyN 配准流程提供加速实现，支持 CPU FP64、CPU FP32 和 Metal FP32 后端。

## 开始使用

1. [安装](installation.md) CPU 软件包或 Metal 可选依赖。
2. 通过[命令行或 Python 接口](usage.md)运行配准。
3. 阅读[输入与输出](inputs-outputs.md)，了解图像坐标、变换方向和输出文件。
4. 按需要修改[配准参数](../../configs/README.zh-CN.md)。

## 参考文档

| 主题 | 内容 |
| --- | --- |
| [安装](installation.md) | 依赖、conda 环境、开发环境 |
| [使用说明](usage.md) | 命令、Python 接口、参数、日志 |
| [输入与输出](inputs-outputs.md) | NIfTI 几何、变换方向、文件格式、重采样 |
| [算法](algorithms.md) | 线性变换、Mattes 互信息、局部 CC、SyN |
| [优化实现](optimizations.md) | 并行内核、算子融合、内存复用、GPU 调度 |
| [代码结构](architecture.md) | 共享模块、CPU 后端、Metal 后端 |
| [测试与质量评估](validation.md) | 数值测试和配准质量指标 |
| [开发说明](development.md) | 测试、实现规范、软件包构建 |
| [开发计划](roadmap.md) | CUDA 支持及后续改进 |

参与开发请阅读[贡献指南](../../CONTRIBUTING.zh-CN.md)，实现来源与版权声明见 [NOTICE](../../NOTICE.zh-CN.md)。

- [性能测试](performance.md)
