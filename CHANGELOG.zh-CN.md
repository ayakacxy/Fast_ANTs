# 更新日志

[English](CHANGELOG.md) · [简体中文](CHANGELOG.zh-CN.md)

## 0.1.0 — 2026-10-26

首个开源版本，采用 GPL-3.0-only，保留 ANTs 和 ITK 的许可证与来源声明。

### 新增功能

- 三维刚体、仿射和标准 greedy SyN 配准。
- CPU FP64、CPU FP32 和 Apple Metal FP32 后端。
- Mattes 互信息、局部互相关、多分辨率图像金字塔和变换应用。
- `fast-ants register` 命令，支持 `brain-syn` 预设和 JSON 配置。
- NIfTI 图像与位移场读写、仿射变换读写及配准质量统计。
- 英文进度日志和 Python 配准 API。
- 中英文文档、示例和开发工具。

### 实现

- CPU 使用 Numba 并行循环、采样与度量融合、可复用工作区。
- Metal 使用设备驻留计算、流式互信息导数、中心化局部统计，以及独立任务的异步提交。
- 并行影像 I/O 和分块输出处理。
